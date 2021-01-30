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

from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from dataset.dataset_factory import dataset_factory
from detector import Detector
from utils.image import plot_tracking, plot_tracking_ddd
import json


min_box_area = 20

_vehicles = ["car", "truck", "bus", "trailer", "construction_vehicle"]
_cycles = ["motorcycle", "bicycle"]
_pedestrians = ["pedestrian"]
attribute_to_id = {
    "": 0,
    "cycle.with_rider": 1,
    "cycle.without_rider": 2,
    "pedestrian.moving": 3,
    "pedestrian.standing": 4,
    "pedestrian.sitting_lying_down": 5,
    "vehicle.moving": 6,
    "vehicle.parked": 7,
    "vehicle.stopped": 8,
}
id_to_attribute = {v: k for k, v in attribute_to_id.items()}
nuscenes_att = np.zeros(8, np.float32)


class PrefetchDataset(torch.utils.data.Dataset):
    def __init__(self, opt, dataset, pre_process_func):
        self.images = dataset.images
        self.load_image_func = dataset.coco.loadImgs
        self.img_dir = dataset.img_dir
        self.pre_process_func = pre_process_func
        self.get_default_calib = dataset.get_default_calib
        self.opt = opt

    def __getitem__(self, index):
        img_id = self.images[index]
        img_info = self.load_image_func(ids=[img_id])[0]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = cv2.imread(img_path)

        images, meta = {}, {}
        for scale in opt.test_scales:
            input_meta = {}
            calib = (
                img_info["calib"]
                if "calib" in img_info
                else self.get_default_calib(image.shape[1], image.shape[0])
            )
            input_meta["calib"] = calib
            images[scale], meta[scale] = self.pre_process_func(image, scale, input_meta)
        ret = {
            "images": images,
            "image": image,
            "meta": meta,
            "frame_id": img_info["frame_id"],
        }
        if "frame_id" in img_info and img_info["frame_id"] == 1:
            ret["is_first_frame"] = 1
            ret["video_id"] = img_info["video_id"]
        return img_id, ret, img_info

    def __len__(self):
        return len(self.images)


def prefetch_test(opt):
    show_image = True
    if not opt.not_set_cuda_env:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus_str
    Dataset = dataset_factory[opt.test_dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    Logger(opt)

    split = "val" if not opt.trainval else "test"
    dataset = Dataset(opt, split)
    detector = Detector(opt)

    if opt.load_results != "":
        load_results = json.load(open(opt.load_results, "r"))
        for img_id in load_results:
            for k in range(len(load_results[img_id])):
                if load_results[img_id][k]["class"] - 1 in opt.ignore_loaded_cats:
                    load_results[img_id][k]["score"] = -1
    else:
        load_results = {}

    data_loader = torch.utils.data.DataLoader(
        PrefetchDataset(opt, dataset, detector.pre_process),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    results = {}
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    bar = Bar("{}".format(opt.exp_id), max=num_iters)
    time_stats = ["tot", "load", "pre", "net", "dec", "post", "merge", "track"]
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    if opt.use_loaded_results:
        for img_id in data_loader.dataset.images:
            results[img_id] = load_results["{}".format(img_id)]
        num_iters = 0
    final_results = []
    out_path = ""

    if opt.dataset == "nuscenes":
        ret = {
            "meta": {
                "use_camera": True,
                "use_lidar": False,
                "use_radar": False,
                "use_map": False,
                "use_external": False,
            },
            "results": {},
        }

    for ind, (img_id, pre_processed_images, img_info) in enumerate(data_loader):
        if ind >= num_iters:
            break

        if opt.dataset == "nuscenes":
            sample_token = img_info["sample_token"][0]
            sensor_id = img_info["sensor_id"].numpy().tolist()[0]

        if opt.tracking and ("is_first_frame" in pre_processed_images):
            if "{}".format(int(img_id.numpy().astype(np.int32)[0])) in load_results:
                pre_processed_images["meta"]["pre_dets"] = load_results[
                    "{}".format(int(img_id.numpy().astype(np.int32)[0]))
                ]
            else:
                print(
                    "No pre_dets for",
                    int(img_id.numpy().astype(np.int32)[0]),
                    ". Use empty initialization.",
                )
                pre_processed_images["meta"]["pre_dets"] = []
            if len(final_results) > 0 and not opt.dataset == "nuscenes":
                write_results(out_path, final_results, opt.dataset)
                final_results = []
            img0 = pre_processed_images["image"][0].numpy()
            h, w, _ = img0.shape
            detector.img_height = h
            detector.img_width = w
            if opt.dataset == "nuscenes":
                save_video_name = os.path.join(
                    opt.dataset + "_videos/",
                    "MOT"
                    + str(int(pre_processed_images["video_id"]))
                    + "_"
                    + str(int(img_info["sensor_id"]))
                    + ".avi",
                )
            elif opt.dataset == "kitti_tracking":
                save_video_name = os.path.join(
                    opt.dataset + "_videos/",
                    "KITTI_" + str(int(pre_processed_images["video_id"])) + ".avi",
                )
            else:
                save_video_name = os.path.join(
                    opt.dataset + "_videos/",
                    "MOT" + str(int(pre_processed_images["video_id"])) + ".avi",
                )
            results_dir = opt.dataset + "_results"
            if not os.path.exists(opt.dataset + "_videos/"):
                os.mkdir(opt.dataset + "_videos/")
            if not os.path.exists(results_dir):
                os.mkdir(results_dir)
            for video in dataset.coco.dataset["videos"]:
                video_id = video["id"]
                file_name = video["file_name"]
                if (
                    pre_processed_images["video_id"] == video_id
                    and not opt.dataset == "nuscenes"
                ):
                    out_path = os.path.join(results_dir, "{}.txt".format(file_name))
                    break

            detector.reset_tracking(opt)
            vw = cv2.VideoWriter(
                save_video_name, cv2.VideoWriter_fourcc("M", "J", "P", "G"), 10, (w, h)
            )
            print("Start tracking video", int(pre_processed_images["video_id"]))
        if opt.public_det:
            if "{}".format(int(img_id.numpy().astype(np.int32)[0])) in load_results:
                pre_processed_images["meta"]["cur_dets"] = load_results[
                    "{}".format(int(img_id.numpy().astype(np.int32)[0]))
                ]
            else:
                print("No cur_dets for", int(img_id.numpy().astype(np.int32)[0]))
                pre_processed_images["meta"]["cur_dets"] = []

        online_targets = detector.run(pre_processed_images, image_info=img_info)
        online_tlwhs = []
        online_ids = []
        online_ddd_boxes = []
        sample_results = []

        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            if tlwh[2] * tlwh[3] > min_box_area:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)

                if opt.dataset == "nuscenes":
                    online_ddd_boxes.append(t.org_ddd_box)
                    class_name = t.classe
                    if class_name in _cycles:
                        att = id_to_attribute[np.argmax(nuscenes_att[0:2]) + 1]
                    elif class_name in _pedestrians:
                        att = id_to_attribute[np.argmax(nuscenes_att[2:5]) + 3]
                    elif class_name in _vehicles:
                        att = id_to_attribute[np.argmax(nuscenes_att[5:8]) + 6]

                    ddd_box = t.ddd_bbox.copy()
                    ddd_box_submission = t.ddd_submission.tolist()
                    translation, size, rotation = (
                        ddd_box_submission[:3],
                        ddd_box_submission[3:6],
                        ddd_box_submission[6:],
                    )

                    result = {
                        "sample_token": sample_token,
                        "translation": translation,
                        "size": size,
                        "rotation": rotation,
                        "velocity": [0, 0],
                        "detection_name": t.classe,
                        "attribute_name": att,
                        "detection_score": t.score,
                        "tracking_name": t.classe,
                        "tracking_score": t.score,
                        "tracking_id": tid,
                        "sensor_id": sensor_id,
                        "det_id": -1,
                    }
                    sample_results.append(result.copy())

        if opt.dataset == "nuscenes":
            if sample_token in ret["results"]:

                ret["results"][sample_token] = (
                    ret["results"][sample_token] + sample_results
                )
            else:
                ret["results"][sample_token] = sample_results

        final_results.append(
            (pre_processed_images["frame_id"].cpu().item(), online_tlwhs, online_ids)
        )
        if show_image:
            img0 = pre_processed_images["image"][0].numpy()

            if opt.dataset == "nuscenes":
                online_im = plot_tracking_ddd(
                    img0,
                    online_tlwhs,
                    online_ddd_boxes,
                    online_ids,
                    frame_id=pre_processed_images["frame_id"],
                    calib=img_info["calib"],
                )
            else:
                online_im = plot_tracking(
                    img0,
                    online_tlwhs,
                    online_ids,
                    frame_id=pre_processed_images["frame_id"],
                )
            vw.write(online_im)

    if not opt.dataset == "nuscenes" and len(final_results) > 0:
        write_results(out_path, final_results, opt.dataset)
        final_results = []
    if opt.dataset == "nuscenes":
        for sample_token in ret["results"].keys():
            confs = sorted(
                [
                    (-d["detection_score"], ind)
                    for ind, d in enumerate(ret["results"][sample_token])
                ]
            )
            ret["results"][sample_token] = [
                ret["results"][sample_token][ind]
                for _, ind in confs[: min(500, len(confs))]
            ]

        json.dump(ret, open(results_dir + "/results.json", "w"))


def _to_list(results):
    for img_id in results:
        for t in range(len(results[img_id])):
            for k in results[img_id][t]:
                if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
                    results[img_id][t][k] = results[img_id][t][k].tolist()
    return results


def write_results(filename, results, data_type):
    if data_type == "mot":
        save_format = "{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n"
    elif data_type == "kitti_tracking":
        save_format = "{frame} {id} Car 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n"
    else:
        raise ValueError(data_type)

    with open(filename, "w") as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == "kitti_tracking":
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(
                    frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h
                )
                f.write(line)


if __name__ == "__main__":
    opt = opts().parse()

    prefetch_test(opt)
