from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch
import copy
import pycocotools.coco as coco
from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from dataset.dataset_factory import dataset_factory
import json
from statistics import mean 
import matplotlib.pyplot as plt
import numpy as np
import statistics


def std(test_list):
    mean = sum(test_list) / len(test_list) 
    variance = sum([((x - mean) ** 2) for x in test_list]) / len(test_list) 
    res = variance ** 0.5
    return res



class PrefetchDataset(torch.utils.data.Dataset):
  def __init__(self, opt, dataset):
    self.images = dataset.images
    self.load_image_func = dataset.coco.loadImgs
    self.img_dir = dataset.img_dir
    self.get_default_calib = dataset.get_default_calib
    self.opt = opt
    split_name='val'
    data_dir = os.path.join(opt.data_dir, 'nuscenes')
    ann_path = os.path.join(data_dir, 
        'annotations', '{}{}.json').format(opt.dataset_version, split_name)
    self.coco = coco.COCO(ann_path)


  def __getitem__(self, index):
    img_id = self.images[index]
    img_info = self.load_image_func(ids=[img_id])[0]
    img_path = os.path.join(self.img_dir, img_info['file_name'])
    image = cv2.imread(img_path)
    
    images, meta = {}, {}
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = copy.deepcopy(self.coco.loadAnns(ids=ann_ids))
    
    ret = {'images': images, 'image': image, 'meta': meta,'frame_id':img_info['frame_id'] }
    if 'frame_id' in img_info and img_info['frame_id'] == 1:
      ret['is_first_frame'] = 1
      ret['video_id'] = img_info['video_id']
    return img_id, ret,img_info,anns

  def __len__(self):
    return len(self.images)
def _coco_box_to_bbox(box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

def dist(a,b):
    return np.linalg.norm(a-b)


def calculate_occ_score(tracks):
    tot=0
    for track_id in tracks:
        tot+=tracks[track_id]['occ']
        
    return tot
def calculate_motion_score(tracks):
    motion=[]
    for track_id in tracks:
        if tracks[track_id]['num_frames']>1:
            motion.append(tracks[track_id]['motion']/(tracks[track_id]['num_frames']-1))   
    motion.sort()
    if len(motion)>10:
        motion=motion[-10:]
    if len(motion):
        return mean(motion)
    else:
        return 0.0
        
        
def extract_diff_splits(opt):

  Dataset = dataset_factory[opt.test_dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  
  split = 'val' 
  dataset = Dataset(opt, split)
  data = PrefetchDataset(opt, dataset)


  results = {}
  num_iters = data.__len__()
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'track']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
    
  difficulty_results={} 
  class_name = [
    'car', 'truck', 'bus', 'trailer', 
    'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle',
    'traffic_cone', 'barrier']
  num_categories = 10
  class_name = [
    'car', 'truck', 'bus', 'trailer', 
    'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle',
    'traffic_cone', 'barrier']
  cat_ids = {i + 1: i + 1 for i in range(num_categories)}
  _tracking_ignored_class = ['construction_vehicle', 'traffic_cone', 'barrier']
  current_video_id = None
  for ind in range(num_iters):
    img_id, pre_processed_images,img_info,anns=data.__getitem__(ind)

    
    
    sample_token = img_info['sample_token']
    sensor_id = img_info['sensor_id']
      
    if sensor_id !=1:
        continue
    if ('is_first_frame' in pre_processed_images):
      if current_video_id is not None:
        difficulty_results[current_video_id]['occ_score'] = calculate_occ_score(tracks)
        difficulty_results[current_video_id]['motion_score'] = calculate_motion_score(tracks)

      tracks_last_locations={}
      tracks={}
      frame_id = 0
      for video in dataset.coco.dataset['videos']:
        video_id = video['id']
        file_name = video['file_name']
        if pre_processed_images['video_id'] == video_id:
          current_video_id=video_id
          difficulty_results[video_id]={'occ_score':0,'motion_score':0,'final_score':0,'sample_tokens':[]}
    frame_id+=1
    difficulty_results[current_video_id]['sample_tokens'].append(sample_token)
    for ann in anns:
      cls_id = int(cat_ids[ann['category_id']])
      object_class =class_name[cls_id-1]
      if object_class in _tracking_ignored_class:
        continue

      bbox = _coco_box_to_bbox(ann['bbox'])
      ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
      track_id = ann['track_id']
      if track_id not in tracks:
        tracks[track_id] = {'last_pos': ct, 'occ':0,'num_frames':1,'motion':0,'last_frame':frame_id}
      else:
        if frame_id - tracks[track_id]['last_frame'] >1:
          tracks[track_id]['occ']+=frame_id - tracks[track_id]['last_frame']-1

        tracks[track_id]['motion']+=dist(ct,tracks[track_id]['last_pos'])
        tracks[track_id]['last_pos'] = ct
        tracks[track_id]['num_frames'] += 1
        tracks[track_id]['last_frame'] = frame_id
  

  occ_scores=[]
  motion_scores=[]
  final_scores=[]

  for video_id in difficulty_results:
    occ_scores.append(difficulty_results[video_id]['occ_score'])  
    motion_scores.append(difficulty_results[video_id]['motion_score'])  
    
  max_occ = max(occ_scores)
  max_motion = max(motion_scores)
    
  occ_scores=[]
  motion_scores=[]
  final_scores=[]
  for video_id in difficulty_results:
    difficulty_results[video_id]['occ_score']/= max_occ
    difficulty_results[video_id]['motion_score']/= max_motion
    occ_scores.append(difficulty_results[video_id]['occ_score'])  
    motion_scores.append(difficulty_results[video_id]['motion_score'])  
    

  for video_id in difficulty_results:
    
    difficulty_results[video_id]['final_score']= max(difficulty_results[video_id]['occ_score'], difficulty_results[video_id]['motion_score'])  
    
    final_scores.append(difficulty_results[video_id]['final_score'])  
    
  print('occ_scores ',occ_scores)
  print('motion_scores ',motion_scores)
  print('final_scores ',final_scores)
  print('mean of occ = ',statistics.mean(occ_scores))
  print('median of motion_scores = ',statistics.median(motion_scores))
  print('median of final_scores = ',statistics.median(final_scores))
  print('std of occ = ',std(occ_scores))
  print('std of motion_scores = ',std(motion_scores))
  print('std of final_scores = ',std(final_scores))

  occ_scores.sort()
  motion_scores.sort()
  final_scores.sort() 
    
    
  
    
  plt.hist(occ_scores, density=False, bins=50)  
  plt.ylabel('Count')
  plt.xlabel('occlusion score');
  plt.savefig('occ.pdf')
  plt.close()


  plt.hist(motion_scores, density=False, bins=50)  
  plt.ylabel('Count')
  plt.xlabel('displacement score');
  plt.savefig('mot.pdf')
  plt.close()
    
  plt.hist(final_scores, density=False, bins=50)  
  plt.ylabel('Count')
  plt.xlabel('mixed score');
  plt.savefig('mixed.pdf')  
  plt.close()

  thr1 =  0.05# occ_scores[100] 
  thr2 =  0.05# occ_scores[50] 
  tot=0
  with open('slipts/hard_videos_occ05.txt', 'w') as file:      
    for video_id in difficulty_results:
      if difficulty_results[video_id]['occ_score'] >thr1:
        tot+=1
        for sample_token in difficulty_results[video_id]['sample_tokens']:
          file.write(sample_token+'\n' )
  tot=0          
  with open('slipts/medium_videos_occ05.txt', 'w') as file:      
    for video_id in difficulty_results:
      if difficulty_results[video_id]['occ_score'] <=thr1 and difficulty_results[video_id]['occ_score'] >thr2:
        tot+=1
        for sample_token in difficulty_results[video_id]['sample_tokens']:
          file.write(sample_token+'\n' ) 
  tot=0      
  with open('slipts/easy_videos_occ05.txt', 'w') as file:      
    for video_id in difficulty_results:
      if difficulty_results[video_id]['occ_score'] <=thr2:
        tot+=1
        for sample_token in difficulty_results[video_id]['sample_tokens']:
          file.write(sample_token+'\n' ) 
 
  thr1=0.35
  thr2=0.195
  tot=0
  with open('slipts/hard_videos_mot3.txt', 'w') as file:      
    for video_id in difficulty_results:
      if difficulty_results[video_id]['motion_score'] >thr1:
        tot+=1
        for sample_token in difficulty_results[video_id]['sample_tokens']:
          file.write(sample_token+'\n' )
  tot=0          
  with open('slipts/medium_videos_mot3.txt', 'w') as file:      
    for video_id in difficulty_results:
      if difficulty_results[video_id]['motion_score'] <=thr1 and difficulty_results[video_id]['motion_score'] >thr2:
        tot+=1
        for sample_token in difficulty_results[video_id]['sample_tokens']:
          file.write(sample_token+'\n' ) 
  tot=0      
  with open('slipts/easy_videos_mot3.txt', 'w') as file:      
    for video_id in difficulty_results:
      if difficulty_results[video_id]['motion_score'] <=thr2:
        tot+=1
        for sample_token in difficulty_results[video_id]['sample_tokens']:
          file.write(sample_token+'\n' ) 


  thr1= 0.37
  thr2 = 0.2
  tot=0
  with open('slipts/hard_videos_final3.txt', 'w') as file:      
    for video_id in difficulty_results:
      
      if difficulty_results[video_id]['final_score'] >thr1:
        tot+=1
        for sample_token in difficulty_results[video_id]['sample_tokens']:
          file.write(sample_token+'\n' )
  tot=0 
  with open('slipts/medium_videos_final3.txt', 'w') as file:      
    for video_id in difficulty_results:
      if difficulty_results[video_id]['final_score'] <=thr1 and difficulty_results[video_id]['final_score'] >thr2:
        tot+=1
        for sample_token in difficulty_results[video_id]['sample_tokens']:
          file.write(sample_token+'\n' ) 
        
  tot=0      
  with open('slipts/easy_videos_final3.txt', 'w') as file:      
    for video_id in difficulty_results:
      if difficulty_results[video_id]['final_score'] <=thr2:
        tot+=1
        for sample_token in difficulty_results[video_id]['sample_tokens']:
          file.write(sample_token+'\n' ) 




    
    
    
def _to_list(results):
  for img_id in results:
    for t in range(len(results[img_id])):
      for k in results[img_id][t]:
        if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
          results[img_id][t][k] = results[img_id][t][k].tolist()
  return results

if __name__ == '__main__':
  opt = opts().parse()
  extract_diff_splits(opt)
