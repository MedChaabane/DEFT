import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from numba import jit
import copy


from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from utils import matching
from utils.tracking_utils.kalman_filter import KalmanFilter
from utils.tracking_utils.kalman_filter_lstm import KalmanFilterLSTM
# from utils.tracking_utils.utils import *
from utils.post_process import ctdet_post_process

from .basetrack import BaseTrack, TrackState
from utils.image import convert_detection
from opts import opts

Max_record_frame = 50
decay = 1.0
decay2=0.01
max_track_node = 50

class Node:
    """
    The Node is the basic element of a track. it contains the following information:
    1) extracted feature (it'll get removed when it isn't active
    2) box (a box (l, t, r, b)
    3) label (active label indicating keeping the features)
    4) detection, the formated box
    """

    def __init__(self, frame_index, id):
        self.frame_index = frame_index
        self.id = id

    def get_box(self, frame_index, recoder):
        return recoder.all_boxes[self.frame_index][self.id, :]


class FeatureRecorder:
    '''
    Record features and boxes every frame
    '''

    def __init__(self,dataset):
        self.max_record_frame = Max_record_frame
        self.all_frame_index = np.array([], dtype=int)
        self.all_features = {}
        self.all_boxes = {}
        self.all_similarity = {}
        self.dataset=dataset

    def update(self, model, frame_index, features, boxes):
        # if the coming frame in the new frame
        if frame_index not in self.all_frame_index:
            # if the recorder have reached the max_record_frame.
            if len(self.all_frame_index) == self.max_record_frame:
                del_frame = self.all_frame_index[0]
                del self.all_features[del_frame]
                del self.all_boxes[del_frame]
                del self.all_similarity[del_frame]
                self.all_frame_index = self.all_frame_index[1:]

            # add new item for all_frame_index, all_features and all_boxes. Besides, also add new similarity
            self.all_frame_index = np.append(self.all_frame_index, frame_index)
            self.all_features[frame_index] = features
            self.all_boxes[frame_index] = boxes

            self.all_similarity[frame_index] = {}
            for pre_index in self.all_frame_index[:-1]:
                # kitti : <5
                m_frame= 10
                if self.dataset == 'kitti_tracking':
                    m_frame=5
                elif self.dataset == 'nuscenes':
                    m_frame=3
                if frame_index - pre_index<m_frame:
                    delta = pow(decay, (frame_index - pre_index)/3.0)
                else:
                    delta = pow(decay2, (frame_index - pre_index)/3.0)
                pre_similarity = model.AFE.forward_stacker_features(self.all_features[pre_index], features, fill_up_column=False)
                self.all_similarity[frame_index][pre_index] = pre_similarity*delta



    def get_feature(self, frame_index, detection_index):
        '''
        get the feature by the specified frame index and detection index
        :param frame_index: start from 0
        :param detection_index: start from 0
        :return: the corresponding feature at frame index and detection index
        '''

        if frame_index in self.all_frame_index:
            features = self.all_features[frame_index]
            if len(features) == 0:
                return None
            if detection_index < len(features):
                return features[detection_index]

        return None

    def get_box(self, frame_index, detection_index):
        if frame_index in self.all_frame_index:
            boxes = self.all_boxes[frame_index]
            if len(boxes) == 0:
                return None

            if detection_index < len(boxes):
                return boxes[detection_index]
        return None

    def get_features(self, frame_index):
        if frame_index in self.all_frame_index:
            features = self.all_features[frame_index]
        else:
            return None
        if len(features) == 0:
            return None
        return features

    def get_boxes(self, frame_index):
        if frame_index in self.all_frame_index:
            boxes = self.all_boxes[frame_index]
        else:
            return None

        if len(boxes) == 0:
            return None
        return boxes    
opt = opts().parse()   
class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    shared_kalman_lstm=KalmanFilterLSTM(opt)
    def __init__(self, tlwh, score,node,recorder, buffer_size=30,use_lstm=True,opt=None,ddd_bbox=None,depth=None,org_ddd_box=None,classe=None,ddd_submission=None):
        

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.depth=depth
        self.score = score
        self.tracklet_len = 0
        self.org_ddd_box=org_ddd_box
        self.smooth_feat = None
        self.classe=classe
        self.ddd_submission=ddd_submission

        
        self.nodes = list()
        self.updated_frame=0
        self.add_node(node.frame_index, recorder, node)
        self.use_lstm=use_lstm
        self.opt=opt
        
        self.last_h=-1
        self.last_w=-1
        self.last_cx= 0
        self.last_cy=0
        self.first_time=True
        self.last_frame_id=-1
        self.last_cz=0
        self.last_l=-1
        self.last_rot_y=0
        
        
        self.future_predictions={}
        opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
        self.device=opt.device
        self.hn=torch.zeros(1, 1, 128).to(device=self.device).float()
        self.cn=torch.zeros(1, 1, 128).to(device=self.device).float()
        self.covariance   =np.eye(4, 4)
        self.observations=[]
        self.observations_tlwh=[]
        self.observations_tlwh.append( self._tlwh.copy())
        self.ddd_bbox=ddd_bbox
        self.observations_ddd_bboxes=[]
        self.dataset=opt.dataset

    

    def __del__(self):
        for n in self.nodes:
            del n

    def add_age(self):
        self.age += 1

    def reset_age(self):
        self.age = 0

    def add_node(self, frame_index, recorder, node):
        self.nodes.append(node)
        self.reset_age()
        return True

    def get_similarity(self, frame_index, recorder):
        similarity = []
        for n in self.nodes:
            f = n.frame_index
            id = n.id

            if frame_index - f >= max_track_node:
                continue

            similarity += [recorder.all_similarity[frame_index][f][id, :]]

        if len(similarity) == 0:
            return None
        a = np.array(similarity)
        if self.dataset == 'nuscenes':
            mm=2
        else:
            mm=4
        if a.shape[0] > mm:
            if a.shape[0] <= mm+1:
                a1 = a[:, : a.shape[1] - 1]
            else:
                a1 = a[a.shape[0] - mm :, : a.shape[1] - 1]
            if a.shape[0] <= mm+1:
                a2 = np.median(a[:, -1:], axis=0)
            else:
                a2 = np.median(a[a.shape[0] - mm :, -1:], axis=0)

            a1 = np.median(a1, axis=0)
            a = np.concatenate((a1, a2), axis=0)
        else:
            a = np.median(np.array(similarity), axis=0)

        return a

        

    def predict(self):
        if not self.use_lstm:
           
            mean_state = self.mean.copy()
            if self.state != TrackState.Tracked:
                mean_state[7] = 0
            self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
            
            
    def prediction_at_frame(self,frame_id):
        if self.dataset == 'nuscenes':
            max_fut=5
        else:
            max_fut=6
        if frame_id in   [self.frame_id + i for i in range(1,max_fut)]:
            return self.future_predictions[frame_id-self.frame_id]
        else:
            return self.future_predictions[max_fut-1]
        
    def prediction_at_frame_tlbr(self,frame_id):
        ret = self.prediction_at_frame(frame_id).copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        ret[2:] += ret[:2]
        
        return ret
        
    def ddd_prediction_at_frame(self,frame_id):
        if self.dataset == 'nuscenes':
            max_fut=5
        else:
            max_fut=6
        if frame_id in   [self.frame_id + i for i in range(1,max_fut)]:
            return self.future_predictions[frame_id-self.frame_id]
        else:
            return self.future_predictions[max_fut-1]    
        
        
        
    @staticmethod
    def multi_predict(stracks):
        
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        
        self.track_id = self.next_id()
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
       
        if self.use_lstm:
            
            
            
            self.kalman_filter = KalmanFilterLSTM(self.opt)
            if self.dataset == 'nuscenes':
                self.update_lstm_features_ddd(self.ddd_bbox)
                self.observations_tlwh.append(self._tlwh.copy())
            else:
                 self.update_lstm_features(self._tlwh)
           


        else:
            self.kalman_filter = kalman_filter
            self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

    def re_activate(self, new_track, frame_id, new_id=False):
        
        

#         self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
       
        self.nodes.append(new_track.nodes[-1])
        self.depth=new_track.depth
        self.org_ddd_box=new_track.org_ddd_box
        self.ddd_bbox=new_track.ddd_bbox
        self.ddd_submission=new_track.ddd_submission
        if self.use_lstm:
            if self.dataset == 'nuscenes':
                self.update_lstm_features_ddd(new_track.ddd_bbox)
                self.observations_tlwh.append(new_track.tlwh.copy())
            else:
                self.update_lstm_features(new_track.tlwh)
        else:
            self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
            
        
        if new_id:
            x=x5
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True):

        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.depth=new_track.depth
        
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        
        self.nodes.append(new_track.nodes[-1])
        self.org_ddd_box=new_track.org_ddd_box
        self.ddd_bbox=new_track.ddd_bbox
        self.ddd_submission=new_track.ddd_submission
        if self.use_lstm:
            
            if self.dataset == 'nuscenes':
                self.update_lstm_features_ddd(new_track.ddd_bbox)
                self.observations_tlwh.append(new_track.tlwh.copy())
            else:
                self.update_lstm_features(new_track.tlwh)
        else:
            self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
            
        


    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.use_lstm:
            ret = self.observations_tlwh[-1].copy()
        else:
            if self.mean is None:
                return self._tlwh.copy()
            ret = self.mean[:4].copy()
            ret[2] *= ret[3]
            ret[:2] -= ret[2:] / 2
        
        
        return ret
    def update_lstm_features(self,tlwh):
        
        self.observations_tlwh.append(tlwh.copy())
        self.observations.append(self.tlwh_to_xyah(tlwh).tolist())
        self.covariance = np.cov(np.asarray(self.observations).copy().T)
        tlwh = tlwh.copy()
        tlwh[:2] += tlwh[2:] / 2

        tlwh_list=tlwh.tolist()
        c_x,c_y,w,h = tlwh_list[0],tlwh_list[1],tlwh_list[2],tlwh_list[3]
        h_w_ratio=w/h
        
        new_features=[]
        if self.first_time:
            self.last_h=h
            self.last_w=w
            self.last_cx= c_x
            self.last_cy=c_y
            self.last_frame_id=self.frame_id
            self.first_time=False
            delta_h =0
            delta_w=0
            v_x=0
            v_y=0
            delta_cx=0
            delta_cy=0
        else:
            delta_h = h-self.last_h
            delta_w = w-self.last_w
            v_x = (c_x- self.last_cx)/(self.frame_id-self.last_frame_id)
            v_y = (c_y-self.last_cy)/(self.frame_id-self.last_frame_id)
            delta_cx=(c_x- self.last_cx)/(self.frame_id-self.last_frame_id)
            delta_cy=(c_y-self.last_cy)/(self.frame_id-self.last_frame_id)
            
        
            
            
            self.last_h=h
            self.last_w=w
            self.last_cx= c_x
            self.last_cy=c_y
            self.last_frame_id=self.frame_id
            
        new_features.append([ c_x,c_y,delta_cx,delta_cy, h, w,h_w_ratio,delta_h,delta_w,v_x,v_y].copy()) 
        new_features=np.array(new_features)    
        new_features =torch.from_numpy(new_features).unsqueeze(0)
        new_features=new_features.to(device=self.device).float()
        
        self.hn,self.cn,self.future_predictions = self.shared_kalman_lstm.predict(self.hn,self.cn,new_features)
            
        for key in self.future_predictions:
            self.future_predictions[key][:2]+= tlwh[:2]
            self.future_predictions[key][2]+= tlwh[3]
            self.future_predictions[key][3]+= tlwh[2]
            pred_h =self.future_predictions[key][2]
            pred_w =self.future_predictions[key][3]
            self.future_predictions[key][3] = pred_h
            self.future_predictions[key][2] = pred_w
            
            self.future_predictions[key][2] /= self.future_predictions[key][3]
    def update_lstm_features_ddd(self,ddd_box):
        
        self.observations_ddd_bboxes.append(ddd_box.copy())
        self.covariance = np.cov(np.asarray(self.observations_ddd_bboxes).copy().T)

        ddd_box_list=ddd_box.copy().tolist()
        h, w, l, c_x, c_y, c_z, rot_y= ddd_box_list[0], ddd_box_list[1], ddd_box_list[2], ddd_box_list[3], ddd_box_list[4], ddd_box_list[5],ddd_box_list[6]
        
        new_features=[]
        if self.first_time:
            
            
            self.last_h=h
            self.last_w=w
            self.last_l=l
            self.last_cx= c_x
            self.last_cy=c_y
            self.last_cz=c_z
            self.last_rot_y = rot_y
            self.last_frame_id=self.frame_id
            self.first_time=False
            delta_h =0
            delta_w=0
            delta_l=0
            v_x=0
            v_y=0
            v_z=0
            v_rot=0
            delta_cx=0
            delta_cy=0
            delta_cz = 0
            delta_rot=0
        else:
            delta_h = h-self.last_h
            delta_w = w-self.last_w
            delta_l = l-self.last_l
            
            v_x = (c_x- self.last_cx)/(self.frame_id-self.last_frame_id)
            v_y = (c_y-self.last_cy)/(self.frame_id-self.last_frame_id)
            v_z = (c_z-self.last_cz)/(self.frame_id-self.last_frame_id)
            v_rot = (rot_y-self.last_rot_y)/(self.frame_id-self.last_frame_id)
            
            delta_cx=(c_x- self.last_cx)
            delta_cy=(c_y-self.last_cy)
            delta_cz=(c_z-self.last_cz)
            delta_rot=rot_y-self.last_rot_y
            
            
            self.last_h=h
            self.last_w=w
            self.last_l=l
            self.last_cx= c_x
            self.last_cy=c_y
            self.last_cz=c_z
            self.last_rot_y = rot_y
            self.last_frame_id=self.frame_id
            
        new_features.append([ c_x, c_y, c_z,delta_cx,delta_cy,delta_cz, h, w,l, delta_h,delta_w,delta_l, v_x,v_y,v_z,rot_y,delta_rot,v_rot].copy()) 
        new_features=np.array(new_features)    
        new_features =torch.from_numpy(new_features).unsqueeze(0)
        new_features=new_features.to(device=self.device).float()
        
        self.hn,self.cn,self.future_predictions = self.shared_kalman_lstm.predict(self.hn,self.cn,new_features)
            
            
            
        for key in self.future_predictions:
            self.future_predictions[key][:3]+= ddd_box[3:6]
            self.future_predictions[key][3]+= ddd_box[-1]
            self.future_predictions[key]=    np.array([h, w, l]+self.future_predictions[key].tolist()) 

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret
    def forward_KF(self,frame_id):
        if self.updated_frame!=frame_id:
            self.mean, self.covariance = self.kalman_filter.predict( self.mean.copy(), self.covariance.copy())
            self.updated_frame=frame_id

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class Tracker(object):
    def __init__(self, opt, model,h=100,w=100,frame_rate=10):
        self.img_height = h
        self.img_width = w
        self.opt = opt
        self.dataset = opt.dataset
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        print('Creating model...')


        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        self.max_per_image = opt.max_object


        self.kalman_filter = KalmanFilter()
        self.recorder = FeatureRecorder(opt.dataset)
        self.model = model
        self.det_thresh = 0.0
        self.img_height = h
        self.img_width = w
        self.use_lstm = opt.lstm
        if  self.use_lstm:
            STrack.shared_kalman_lstm=KalmanFilterLSTM(opt)
            self.kalman_filter =KalmanFilterLSTM(opt)
    def get_similarity(self, frame_index,strack_pool,num_detections):
        
        
        ids = []
        similarity = []
        for t in strack_pool:
            s = t.get_similarity(frame_index, self.recorder)
          
            if s is None:
                
                s=[0.0]*(num_detections+1) 
            else:
                s=s.tolist()
            similarity += [s]

        similarity = np.array(similarity)

        track_num = similarity.shape[0]
        if track_num > 0:
            box_num = similarity.shape[1]
        else:
            box_num = 0

        if track_num == 0 :
            return np.array(similarity)

        return  np.array(similarity)

    def post_process(self, dets, meta):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.opt.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.opt.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def update(self, results,FeatureMaps,ddd_boxes=None,depths_by_class=None,ddd_org_boxes=None,submission = None, classe = None):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        output_stracks=[]



        if self.dataset == 'nuscenes':
            dets=np.array(results)
            ddd_boxes=np.array(ddd_boxes)
            depths=np.array(depths_by_class)
            ddd_org_boxes=np.array(ddd_org_boxes)
            if len(dets) > 0:
                nodes = [Node(self.frame_id, i) for i in range(dets.shape[0])]
                detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], node, 30,use_lstm=self.use_lstm, opt=self.opt , ddd_bbox = ddd_box,depth=depth[0],org_ddd_box=org_ddd_box,classe=classe,ddd_submission=ddd_submission) for
                              (tlbrs, node,ddd_box,depth,org_ddd_box,ddd_submission) in zip(dets[:, :5], nodes,ddd_boxes,depths,ddd_org_boxes,submission)]
                detection_org = np.copy(dets[:,:4])
                detection_centers = convert_detection(np.copy(detection_org),self.img_height, self.img_width)
                features = self.model.AFE.forward_feature_extracter(FeatureMaps, detection_centers)
                self.recorder.update(self.model, self.frame_id, features.data, detection_org)

            else:
                detections = []

                
            
            
            
        else:
            if self.dataset == 'kitti_tracking':

                dets = np.array(
          [det['bbox'].tolist() + [det['score']] for det in results if det['class'] == 2], np.float32)
            else:
                dets = np.array(
          [det['bbox'].tolist() + [det['score']] for det in results], np.float32)

            if len(dets) > 0:
                '''Detections'''
                nodes = [Node(self.frame_id, i) for i in range(dets.shape[0])]
                detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], node, 30,use_lstm=self.use_lstm, opt=self.opt ) for
                              (tlbrs, node) in zip(dets[:, :5], nodes)]
                detection_org = np.copy(dets[:,:4])
                detection_centers = convert_detection(np.copy(detection_org),self.img_height, self.img_width)
                if FeatureMaps[0].shape[0]==2:
                    ll=[]
                    for feayure_map in FeatureMaps:
                        ll.append(feayure_map[0].unsqueeze(0))
                    FeatureMaps=ll.copy()
                features = self.model.AFE.forward_feature_extracter(FeatureMaps, detection_centers)
                self.recorder.update(self.model, self.frame_id, features.data, detection_org)

            else:
                detections = []

            
        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = [] 
        for track in self.tracked_stracks:

            tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

            
            
        if not self.use_lstm:
            STrack.multi_predict(strack_pool)
        
        
        lll =len(detections)
        if self.dataset == 'nuscenes' and classe != 'pedestrian':
            
            strack_pool_old= [track for track in strack_pool if abs(track.frame_id - self.frame_id) >=3 ]
            strack_pool_new= [track for track in strack_pool if abs(track.frame_id - self.frame_id) <3 ]
            
            dists =  matching.iou_ddd_distance(strack_pool_new, detections,self.frame_id)
        
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.999)

            for itracked, idet in matches:
                track = strack_pool_new[itracked]
                output_stracks.append(track)
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    
            detections = [detections[i] for i in u_detection]
            r_tracked_stracks = [strack_pool_new[i] for i in u_track ]
            
            strack_pool=  joint_stracks(r_tracked_stracks, strack_pool_old)   
                    
        dists = np.zeros((len(strack_pool), len(detections)), dtype=np.float)
        if dists.size != 0:
            
            dists = self.get_similarity(self.frame_id,strack_pool,len(detections))
 
            dists=dists[:,:-1]
            if self.dataset == 'nuscenes' and classe != 'pedestrian':
                dists=dists[:,u_detection]
            dists=1-dists
            

        if self.dataset == 'nuscenes':
             dists = matching.fuse_motion_ddd(self.kalman_filter, dists, strack_pool, detections,frame_id=self.frame_id,classe_name = classe)
        else:
            dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections,frame_id=self.frame_id,use_lstm=self.use_lstm)
        matches, u_track, u_detection2 = matching.linear_assignment(dists, thresh=0.9)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            output_stracks.append(track)
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
        r_tracked_stracks = [strack_pool[i] for i in u_track ]
        detections = [detections[i] for i in u_detection2]
        if self.dataset == 'nuscenes' and len(detections)>0:
            
            
            dists = self.get_similarity(self.frame_id,r_tracked_stracks,lll)
            if dists.size != 0:
                dists=dists[:,:-1]
                if classe != 'pedestrian':
                    dists=dists[:,u_detection]
                dists=dists[:,u_detection2]
                dists=1-dists
                
                matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.9)
                for itracked, idet in matches:
                    track = r_tracked_stracks[itracked]
                    output_stracks.append(track)
                    det = detections[idet]
                    if track.state == TrackState.Tracked:
                        track.update(detections[idet], self.frame_id)
                        activated_starcks.append(track)
                    else:
                        track.re_activate(det, self.frame_id, new_id=False)
            
            
                detections = [detections[i] for i in u_detection]
                strack_pool = r_tracked_stracks
        elif self.dataset == 'kitti_tracking' and len(detections)>0:
            dists = self.get_similarity(self.frame_id,r_tracked_stracks,lll)
            if dists.size != 0:
                dists=dists[:,:-1]
                
                dists=dists[:,u_detection2]
                dists=1-dists
                
                matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.9)
                for itracked, idet in matches:
                    track = r_tracked_stracks[itracked]
                    output_stracks.append(track)
                    det = detections[idet]
                    if track.state == TrackState.Tracked:
                        track.update(detections[idet], self.frame_id)
                        activated_starcks.append(track)
                    else:
                        track.re_activate(det, self.frame_id, new_id=False)
            
            
                detections = [detections[i] for i in u_detection]
                strack_pool = r_tracked_stracks
                
                
      
    

        
        if self.dataset == 'kitti_tracking' or self.dataset == 'nuscenes':
            mm=6
            if self.dataset == 'nuscenes':
                mm=3
            r_tracked_stracks = [strack_pool[i] for i in u_track if abs(self.frame_id- strack_pool[i].frame_id)<mm]
            
             
        else:
            r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
            
        if self.dataset == 'nuscenes':
            dists = matching.iou_distance(r_tracked_stracks, detections,self.frame_id,use_prediction= False)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.0)
        else:
            dists = matching.iou_distance(r_tracked_stracks, detections,self.frame_id,use_prediction =self.use_lstm )
            
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.9)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            output_stracks.append(track)
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                
   
        for it in u_track:
            track = r_tracked_stracks[it]
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        detections = [detections[i] for i in u_detection]

        for track in detections:
            output_stracks.append(track)
            if track.score < self.det_thresh:

                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)


        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks,self.dataset == 'nuscenes')



        


        return output_stracks

# this function is taken from https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/tracker/multitracker.py
def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

# this function is taken from https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/tracker/multitracker.py
def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

# this function is taken from https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/tracker/multitracker.py
def remove_duplicate_stracks(stracksa, stracksb,ddd_tracking=False):
    if ddd_tracking:
        pdist = matching.iou_ddd_distance(stracksa, stracksb,use_prediction=False)
    else:
        pdist = matching.iou_distance(stracksa, stracksb,use_prediction=False)
    
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


