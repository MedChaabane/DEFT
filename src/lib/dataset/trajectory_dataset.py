# ------------------------------------------------------------------------------
# Large part of this code is taken from https://github.com/xingyizhou/CenterTrack/
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import json
import cv2
import os
from collections import defaultdict

import pycocotools.coco as coco
import torch
import torch.utils.data as data

from utils.image import (
    flip,
    color_aug,
    show_matching_hanlded_rectangle,
    ResizeShuffleBoxes,
    FormatBoxes,
    ToTensor,
    ToPercentCoordinates,
)
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian
import copy
from numpy.random import randint
from functools import reduce
from pyquaternion import Quaternion


from nuscenes.utils.data_classes import Box

NUSCENES_TRACKING_NAMES = [
    "bicycle",
    "bus",
    "car",
    "motorcycle",
    "pedestrian",
    "trailer",
    "truck",
]


class TrajectoryDataset(data.Dataset):
    is_fusion_dataset = False
    default_resolution = None
    num_categories = None
    class_name = None
    # cat_ids: map from 'category_id' in the annotation files to 1..num_categories
    # Not using 0 because 0 is used for don't care region and ignore loss.
    cat_ids = None
    max_objs = None
    rest_focal_length = 1200
    num_joints = 17
    flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
    edges = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 4],
        [4, 6],
        [3, 5],
        [5, 6],
        [5, 7],
        [7, 9],
        [6, 8],
        [8, 10],
        [6, 12],
        [5, 11],
        [11, 12],
        [12, 14],
        [14, 16],
        [11, 13],
        [13, 15],
    ]
    mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(
        1, 1, 3
    )
    std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(
        1, 1, 3
    )
    _eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
    _eig_vec = np.array(
        [
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938],
        ],
        dtype=np.float32,
    )
    ignore_val = 1
    nuscenes_att_range = {
        0: [0, 1],
        1: [0, 1],
        2: [2, 3, 4],
        3: [2, 3, 4],
        4: [2, 3, 4],
        5: [5, 6, 7],
        6: [5, 6, 7],
        7: [5, 6, 7],
    }

    def __init__(self, opt=None, split=None, ann_path=None, img_dir=None):
        super(TrajectoryDataset, self).__init__()
        if opt is not None and split is not None:
            self.split = split
            self.opt = opt
            self._data_rng = np.random.RandomState(123)
        if ann_path is not None and img_dir is not None:
            print(
                "==> initializing {} data from {}, \n images from {} ...".format(
                    split, ann_path, img_dir
                )
            )
            self.coco = coco.COCO(ann_path)
            self.images = self.coco.getImgIds()
            if opt.tracking:
                if not ("videos" in self.coco.dataset):
                    self.fake_video_data()
                print("Creating video index!")
                self.video_to_images = defaultdict(list)
                for image in self.coco.dataset["images"]:
                    self.video_to_images[image["video_id"]].append(image)
            self.img_dir = img_dir
            self.max_object = opt.max_object
        self.valid_indexes = []
        self.non_valid_indexes = []
        self.gt_bboxes = {}
        self.gt_track_ids = {}
        self.possible_tracks = {}
        self.dataset = opt.dataset
        if self.dataset == "nuscenes":
            self.MAX_dis = 10
            self.MAX_dis_fut = 4
        else:

            self.MAX_dis = 10
            self.MAX_dis_fut = 5

    def __len__(self):
        return self.num_samples - self.MAX_dis_fut - 1

    def __getitem__(self, index):
        opt = self.opt
        if index < self.MAX_dis + 2 or index > self.__len__() - 2:
            return self.__getitem__(randint(12, self.__len__() - 2))
        if index not in self.possible_tracks:
            valid = False
            while not valid:
                while index in self.non_valid_indexes:
                    index = index + randint(-10, 10)
                    if index < self.MAX_dis + 2 or index > self.__len__() - 2:
                        return self.__getitem__(randint(12, self.__len__() - 2))

                img, anns, img_info, img_path = self._load_data(index)
                if "sensor_id" in img_info:
                    sensor_id = img_info["sensor_id"]

                if index not in self.valid_indexes:

                    video_id = img_info["video_id"]
                    img_infos = self.video_to_images[video_id]
                    frame_id = img_info["frame_id"]
                    img_ids = [
                        img_info["frame_id"]
                        for img_info in img_infos
                        if abs(img_info["frame_id"] - frame_id) <= self.MAX_dis
                        and (
                            not ("sensor_id" in img_info)
                            or img_info["sensor_id"] == sensor_id
                        )
                    ]

                    pre_frame = frame_id - self.MAX_dis
                    fut_frame = frame_id + self.MAX_dis_fut
                    if pre_frame in img_ids and fut_frame in img_ids:
                        valid = True
                    else:
                        self.non_valid_indexes.append(index)
                        index = index + randint(-10, 10)
                        if index < self.MAX_dis + 2 or index > self.__len__() - 2:
                            return self.__getitem__(randint(12, self.__len__() - 2))
                else:
                    valid = True

            if index not in self.valid_indexes:
                self.valid_indexes.append(index)

            for ind in range(index - self.MAX_dis, index + self.MAX_dis_fut + 1):
                if ind not in self.gt_bboxes:
                    img, anns, img_info, img_path = self._load_data(ind)
                    if "sensor_id" in img_info:
                        sensor_id = img_info["sensor_id"]
                    bboxes, track_ids, classes = self._get_bboxes(anns, img_info)
                    self.gt_bboxes[ind] = bboxes
                    self.gt_track_ids[ind] = track_ids
            ll = []
            for ind in range(index - 2, index + self.MAX_dis_fut + 1):
                ll.append(self.gt_track_ids[ind])

            possible_track_ids = list(reduce(set.intersection, [set(l_) for l_ in ll]))
        else:
            possible_track_ids = self.possible_tracks[index]
        if len(possible_track_ids) == 0:
            self.non_valid_indexes.append(index)
            index = index + randint(-10, 10)
            return self.__getitem__(index)
        else:
            self.possible_tracks[index] = possible_track_ids
            rand_id = np.random.choice(len(possible_track_ids))
            track_id = possible_track_ids[rand_id]

        if self.dataset == "nuscenes":

            t = -1
            last_t = -1
            last_h = -1
            last_w = -1
            last_l = -1
            last_rot_y = -1
            last_cx = 0
            last_cy = 0
            last_cz = 0
            delta_h = 0
            delta_w = 0
            delta_l = 0
            v_x = 0
            v_y = 0
            v_z = 0
            v_rot = 0
            delta_cx = 0
            delta_cy = 0
            delta_cz = 0
            delta_rot = 0

            input_traj = []
            for ind in range(index - self.MAX_dis, index + 1):
                t += 1
                bboxes = self.gt_bboxes[ind]
                track_ids = self.gt_track_ids[ind]
                if ind > index - 2 or randint(0, 10) < 9:
                    if track_id in track_ids:
                        track_id_index = track_ids.index(track_id)
                        bbox = bboxes[track_id_index]

                        h, w, l, c_x, c_y, c_z, rot_y = (
                            bbox[0],
                            bbox[1],
                            bbox[2],
                            bbox[3],
                            bbox[4],
                            bbox[5],
                            bbox[6],
                        )
                        if last_t > -1:
                            delta_h = h - last_h
                            delta_w = w - last_w
                            delta_l = l - last_l

                            v_x = (c_x - last_cx) / (t - last_t)
                            v_y = (c_y - last_cy) / (t - last_t)
                            v_z = (c_z - last_cz) / (t - last_t)
                            v_rot = (rot_y - last_rot_y) / (t - last_t)
                            delta_cx = c_x - last_cx
                            delta_cy = c_y - last_cy
                            delta_cz = c_z - last_cz
                            delta_rot = rot_y - last_rot_y

                        input_traj.append(
                            [
                                c_x,
                                c_y,
                                c_z,
                                delta_cx,
                                delta_cy,
                                delta_cz,
                                h,
                                w,
                                l,
                                delta_h,
                                delta_w,
                                delta_l,
                                v_x,
                                v_y,
                                v_z,
                                rot_y,
                                delta_rot,
                                v_rot,
                            ].copy()
                        )

                        last_t = t
                        last_h = h
                        last_w = w
                        last_l = l
                        last_cx = c_x
                        last_cy = c_y
                        last_cz = c_z
                        last_rot_y = rot_y

            output_traj = []
            for ind in range(index + 1, index + self.MAX_dis_fut + 1):
                bboxes = self.gt_bboxes[ind]
                track_ids = self.gt_track_ids[ind]
                track_id_index = track_ids.index(track_id)
                bbox = bboxes[track_id_index]
                h, w, l, c_x, c_y, c_z, rot_y = (
                    bbox[0],
                    bbox[1],
                    bbox[2],
                    bbox[3],
                    bbox[4],
                    bbox[5],
                    bbox[6],
                )
                delta_h = h - last_h
                delta_w = w - last_w
                delta_l = l - last_l
                delta_cx = c_x - last_cx
                delta_cy = c_y - last_cy
                delta_cz = c_z - last_cz
                delta_rot = rot_y - last_rot_y

                output_traj.append([delta_cx, delta_cy, delta_cz, delta_rot].copy())
        else:
            t = -1
            last_t = -1
            last_h = -1
            last_w = -1
            last_cx = 0
            last_cy = 0
            delta_h = 0
            delta_w = 0
            v_x = 0
            v_y = 0
            v_z = 0
            delta_cx = 0
            delta_cy = 0
            delta_cz = 0

            input_traj = []
            for ind in range(index - self.MAX_dis, index + 1):
                t += 1
                bboxes = self.gt_bboxes[ind]
                track_ids = self.gt_track_ids[ind]
                if ind > index - 2 or randint(0, 10) < 9:

                    if track_id in track_ids:
                        track_id_index = track_ids.index(track_id)
                        bbox = bboxes[track_id_index]
                        c_x, c_y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                        h_w_ratio = w / h
                        if last_t > -1:
                            delta_h = h - last_h
                            delta_w = w - last_w
                            v_x = (c_x - last_cx) / (t - last_t)
                            v_y = (c_y - last_cy) / (t - last_t)
                            delta_cx = (c_x - last_cx) / (t - last_t)
                            delta_cy = (c_y - last_cy) / (t - last_t)

                        input_traj.append(
                            [
                                c_x,
                                c_y,
                                delta_cx,
                                delta_cy,
                                h,
                                w,
                                h_w_ratio,
                                delta_h,
                                delta_w,
                                v_x,
                                v_y,
                            ].copy()
                        )
                        last_t = t
                        last_h = h
                        last_w = w
                        last_cx = c_x
                        last_cy = c_y

            output_traj = []
            for ind in range(index + 1, index + self.MAX_dis_fut + 1):
                bboxes = self.gt_bboxes[ind]
                track_ids = self.gt_track_ids[ind]
                track_id_index = track_ids.index(track_id)
                bbox = bboxes[track_id_index]
                c_x, c_y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                h_w_ratio = w / h
                delta_h = h - last_h
                delta_w = w - last_w
                delta_cx = c_x - last_cx
                delta_cy = c_y - last_cy

                output_traj.append([delta_cx, delta_cy, delta_h, delta_w].copy())

        input_traj = np.array(input_traj)
        output_traj = np.array(output_traj)

        return input_traj, output_traj

    def _get_bboxes(self, anns, image_info=None):
        bboxes, track_ids, classes = [], [], []
        if self.dataset == "nuscenes":
            for ann in anns:
                trans_matrix = np.array(image_info["trans_matrix"], np.float32)
                cls_id = int(self.cat_ids[ann["category_id"]])
                class_name = self.class_name[cls_id - 1]
                if class_name not in NUSCENES_TRACKING_NAMES:
                    continue
                center_location = ann["location"]
                wlh = ann["dim"]
                rotation = ann["rotation_y"]
                size = [float(wlh[1]), float(wlh[2]), float(wlh[0])]

                rot_cam = Quaternion(axis=[0, 1, 0], angle=rotation)
                loc = np.array(
                    [center_location[0], center_location[1], center_location[2]],
                    np.float32,
                ).copy()

                translation = np.dot(
                    trans_matrix,
                    np.array(
                        [
                            center_location[0],
                            center_location[1] - size[2] / 2,
                            center_location[2],
                            1,
                        ],
                        np.float32,
                    ),
                ).copy()
                box = Box(loc, size, rot_cam, name="2", token="1")
                box.translate(np.array([0, -box.wlh[2] / 2, 0]))
                box.rotate(Quaternion(image_info["cs_record_rot"]))
                box.translate(np.array(image_info["cs_record_trans"]))
                box.rotate(Quaternion(image_info["pose_record_rot"]))
                box.translate(np.array(image_info["pose_record_trans"]))
                rotation = box.orientation
                rotation = [
                    float(rotation.w),
                    float(rotation.x),
                    float(rotation.y),
                    float(rotation.z),
                ]
                q = Quaternion(rotation)
                angle = q.angle if q.axis[2] > 0 else -q.angle

                bboxes.append(
                    [
                        size[2],
                        size[0],
                        size[1],
                        box.center[0],
                        box.center[1],
                        box.center[2],
                        angle,
                    ].copy()
                )
                track_ids.append(ann["track_id"])
                classes.append(class_name)

        else:

            for ann in anns:
                cls_id = int(self.cat_ids[ann["category_id"]])
                if (
                    cls_id > self.opt.num_classes
                    or cls_id <= -99
                    or ("iscrowd" in ann and ann["iscrowd"] > 0)
                ):
                    continue
                bbox = self._coco_box_to_bbox(ann["bbox"])
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                if h > 0 and w > 0:
                    bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3]].copy())

                    track_ids.append(ann["track_id"] if "track_id" in ann else -1)

        return bboxes, track_ids, classes

    def _load_pre_data_AFE(self, video_id, frame_id, sensor_id=1):
        img_infos = self.video_to_images[video_id]

        if "train" in self.split:
            img_ids = [
                (img_info["id"], img_info["frame_id"])
                for img_info in img_infos
                if abs(img_info["frame_id"] - frame_id) < self.opt.max_frame_dist_AFE
                and (
                    not ("sensor_id" in img_info) or img_info["sensor_id"] == sensor_id
                )
            ]
        else:
            img_ids = [
                (img_info["id"], img_info["frame_id"])
                for img_info in img_infos
                if (img_info["frame_id"] - frame_id) == -1
                and (
                    not ("sensor_id" in img_info) or img_info["sensor_id"] == sensor_id
                )
            ]
            if len(img_ids) == 0:
                img_ids = [
                    (img_info["id"], img_info["frame_id"])
                    for img_info in img_infos
                    if (img_info["frame_id"] - frame_id) == 0
                    and (
                        not ("sensor_id" in img_info)
                        or img_info["sensor_id"] == sensor_id
                    )
                ]

        rand_id = np.random.choice(len(img_ids))
        img_id, pre_frame_id = img_ids[rand_id]
        frame_dist = abs(frame_id - pre_frame_id)

        img, anns, _, _ = self._load_image_anns(img_id, self.coco, self.img_dir)
        return img, anns, frame_dist

    def get_default_calib(self, width, height):
        calib = np.array(
            [
                [self.rest_focal_length, 0, width / 2, 0],
                [0, self.rest_focal_length, height / 2, 0],
                [0, 0, 1, 0],
            ]
        )
        return calib

    def _load_image_anns(self, img_id, coco, img_dir):
        img_info = coco.loadImgs(ids=[img_id])[0]
        file_name = img_info["file_name"]
        img_path = os.path.join(img_dir, file_name)
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = copy.deepcopy(coco.loadAnns(ids=ann_ids))
        img = cv2.imread(img_path)
        return img, anns, img_info, img_path

    def _load_data(self, index):
        coco = self.coco
        img_dir = self.img_dir
        img_id = self.images[index]
        img, anns, img_info, img_path = self._load_image_anns(img_id, coco, img_dir)

        return img, anns, img_info, img_path

    def _load_pre_data(self, video_id, frame_id, sensor_id=1):
        img_infos = self.video_to_images[video_id]

        if "train" in self.split:
            img_ids = [
                (img_info["id"], img_info["frame_id"])
                for img_info in img_infos
                if abs(img_info["frame_id"] - frame_id) < self.opt.max_frame_dist
                and (
                    not ("sensor_id" in img_info) or img_info["sensor_id"] == sensor_id
                )
            ]
        else:
            img_ids = [
                (img_info["id"], img_info["frame_id"])
                for img_info in img_infos
                if (img_info["frame_id"] - frame_id) == -1
                and (
                    not ("sensor_id" in img_info) or img_info["sensor_id"] == sensor_id
                )
            ]
            if len(img_ids) == 0:
                img_ids = [
                    (img_info["id"], img_info["frame_id"])
                    for img_info in img_infos
                    if (img_info["frame_id"] - frame_id) == 0
                    and (
                        not ("sensor_id" in img_info)
                        or img_info["sensor_id"] == sensor_id
                    )
                ]
        rand_id = np.random.choice(len(img_ids))
        img_id, pre_frame_id = img_ids[rand_id]
        frame_dist = abs(frame_id - pre_frame_id)

        img, anns, _, _ = self._load_image_anns(img_id, self.coco, self.img_dir)
        return img, anns, frame_dist

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def _get_aug_param(self, c, s, width, height, disturb=False):
        if (not self.opt.not_rand_crop) and not disturb:
            aug_s = np.random.choice(np.arange(0.6, 1.4, 0.1))
            w_border = self._get_border(128, width)
            h_border = self._get_border(128, height)
            c[0] = np.random.randint(low=w_border, high=width - w_border)
            c[1] = np.random.randint(low=h_border, high=height - h_border)
        else:
            sf = self.opt.scale
            cf = self.opt.shift
            if type(s) == float:
                s = [s, s]
            c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            aug_s = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

        if np.random.random() < self.opt.aug_rot:
            rf = self.opt.rotate
            rot = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
        else:
            rot = 0

        return c, aug_s, rot

    def _get_pre_dets(self, anns, trans_input, trans_output):
        hm_h, hm_w = self.opt.input_h, self.opt.input_w
        down_ratio = self.opt.down_ratio
        trans = trans_input
        reutrn_hm = self.opt.pre_hm
        pre_hm = np.zeros((1, hm_h, hm_w), dtype=np.float32) if reutrn_hm else None
        pre_cts, track_ids = [], []
        for ann in anns:
            cls_id = int(self.cat_ids[ann["category_id"]])
            if (
                cls_id > self.opt.num_classes
                or cls_id <= -99
                or ("iscrowd" in ann and ann["iscrowd"] > 0)
            ):
                continue
            bbox = self._coco_box_to_bbox(ann["bbox"])
            bbox[:2] = affine_transform(bbox[:2], trans)
            bbox[2:] = affine_transform(bbox[2:], trans)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, hm_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, hm_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            max_rad = 1
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                max_rad = max(max_rad, radius)
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32
                )
                ct0 = ct.copy()
                conf = 1

                ct[0] = ct[0] + np.random.randn() * self.opt.hm_disturb * w
                ct[1] = ct[1] + np.random.randn() * self.opt.hm_disturb * h
                conf = 1 if np.random.random() > self.opt.lost_disturb else 0

                ct_int = ct.astype(np.int32)
                if conf == 0:
                    pre_cts.append(ct / down_ratio)
                else:
                    pre_cts.append(ct0 / down_ratio)

                track_ids.append(ann["track_id"] if "track_id" in ann else -1)
                if reutrn_hm:
                    draw_umich_gaussian(pre_hm[0], ct_int, radius, k=conf)

                if np.random.random() < self.opt.fp_disturb and reutrn_hm:
                    ct2 = ct0.copy()
                    # Hard code heatmap disturb ratio, haven't tried other numbers.
                    ct2[0] = ct2[0] + np.random.randn() * 0.05 * w
                    ct2[1] = ct2[1] + np.random.randn() * 0.05 * h
                    ct2_int = ct2.astype(np.int32)
                    draw_umich_gaussian(pre_hm[0], ct2_int, radius, k=conf)

        return pre_hm, pre_cts, track_ids

    def _flip_anns(self, anns, width):
        for k in range(len(anns)):
            bbox = anns[k]["bbox"]
            anns[k]["bbox"] = [width - bbox[0] - 1 - bbox[2], bbox[1], bbox[2], bbox[3]]

            if "hps" in self.opt.heads and "keypoints" in anns[k]:
                keypoints = np.array(anns[k]["keypoints"], dtype=np.float32).reshape(
                    self.num_joints, 3
                )
                keypoints[:, 0] = width - keypoints[:, 0] - 1
                for e in self.flip_idx:
                    keypoints[e[0]], keypoints[e[1]] = (
                        keypoints[e[1]].copy(),
                        keypoints[e[0]].copy(),
                    )
                anns[k]["keypoints"] = keypoints.reshape(-1).tolist()

            if "rot" in self.opt.heads and "alpha" in anns[k]:
                anns[k]["alpha"] = (
                    np.pi - anns[k]["alpha"]
                    if anns[k]["alpha"] > 0
                    else -np.pi - anns[k]["alpha"]
                )

            if "amodel_offset" in self.opt.heads and "amodel_center" in anns[k]:
                anns[k]["amodel_center"][0] = width - anns[k]["amodel_center"][0] - 1

            if self.opt.velocity and "velocity" in anns[k]:
                anns[k]["velocity"] = [-10000, -10000, -10000]

        return anns

    def _init_ret(self, ret, gt_det):
        max_objs = self.max_objs * self.opt.dense_reg
        ret["hm"] = np.zeros(
            (self.opt.num_classes, self.opt.output_h, self.opt.output_w), np.float32
        )
        ret["ind"] = np.zeros((max_objs), dtype=np.int64)
        ret["cat"] = np.zeros((max_objs), dtype=np.int64)
        ret["mask"] = np.zeros((max_objs), dtype=np.float32)

        regression_head_dims = {
            "reg": 2,
            "wh": 2,
            "tracking": 2,
            "ltrb": 4,
            "ltrb_amodal": 4,
            "nuscenes_att": 8,
            "velocity": 3,
            "hps": self.num_joints * 2,
            "dep": 1,
            "dim": 3,
            "amodel_offset": 2,
        }

        for head in regression_head_dims:
            if head in self.opt.heads:
                ret[head] = np.zeros(
                    (max_objs, regression_head_dims[head]), dtype=np.float32
                )
                ret[head + "_mask"] = np.zeros(
                    (max_objs, regression_head_dims[head]), dtype=np.float32
                )
                gt_det[head] = []

        if "hm_hp" in self.opt.heads:
            num_joints = self.num_joints
            ret["hm_hp"] = np.zeros(
                (num_joints, self.opt.output_h, self.opt.output_w), dtype=np.float32
            )
            ret["hm_hp_mask"] = np.zeros((max_objs * num_joints), dtype=np.float32)
            ret["hp_offset"] = np.zeros((max_objs * num_joints, 2), dtype=np.float32)
            ret["hp_ind"] = np.zeros((max_objs * num_joints), dtype=np.int64)
            ret["hp_offset_mask"] = np.zeros(
                (max_objs * num_joints, 2), dtype=np.float32
            )
            ret["joint"] = np.zeros((max_objs * num_joints), dtype=np.int64)

        if "rot" in self.opt.heads:
            ret["rotbin"] = np.zeros((max_objs, 2), dtype=np.int64)
            ret["rotres"] = np.zeros((max_objs, 2), dtype=np.float32)
            ret["rot_mask"] = np.zeros((max_objs), dtype=np.float32)
            gt_det.update({"rot": []})

    def _get_calib(self, img_info, width, height):
        if "calib" in img_info:
            calib = np.array(img_info["calib"], dtype=np.float32)
        else:
            calib = np.array(
                [
                    [self.rest_focal_length, 0, width / 2, 0],
                    [0, self.rest_focal_length, height / 2, 0],
                    [0, 0, 1, 0],
                ]
            )
        return calib

    def _ignore_region(self, region, ignore_val=1):
        np.maximum(region, ignore_val, out=region)

    def _mask_ignore_or_crowd(self, ret, cls_id, bbox):
        # mask out crowd region, only rectangular mask is supported
        if cls_id == 0:  # ignore all classes
            self._ignore_region(
                ret["hm"][
                    :, int(bbox[1]) : int(bbox[3]) + 1, int(bbox[0]) : int(bbox[2]) + 1
                ]
            )
        else:
            # mask out one specific class
            self._ignore_region(
                ret["hm"][
                    abs(cls_id) - 1,
                    int(bbox[1]) : int(bbox[3]) + 1,
                    int(bbox[0]) : int(bbox[2]) + 1,
                ]
            )
        if ("hm_hp" in ret) and cls_id <= 1:
            self._ignore_region(
                ret["hm_hp"][
                    :, int(bbox[1]) : int(bbox[3]) + 1, int(bbox[0]) : int(bbox[2]) + 1
                ]
            )

    def _coco_box_to_bbox(self, box):
        bbox = np.array(
            [box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32
        )
        return bbox

    def _get_bbox_output(self, bbox, trans_output, height, width):
        bbox = self._coco_box_to_bbox(bbox).copy()

        rect = np.array(
            [
                [bbox[0], bbox[1]],
                [bbox[0], bbox[3]],
                [bbox[2], bbox[3]],
                [bbox[2], bbox[1]],
            ],
            dtype=np.float32,
        )
        for t in range(4):
            rect[t] = affine_transform(rect[t], trans_output)
        bbox[:2] = rect[:, 0].min(), rect[:, 1].min()
        bbox[2:] = rect[:, 0].max(), rect[:, 1].max()

        bbox_amodal = copy.deepcopy(bbox)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.opt.output_w - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.opt.output_h - 1)
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        return bbox, bbox_amodal

    def _add_instance(
        self,
        ret,
        gt_det,
        k,
        cls_id,
        bbox,
        bbox_amodal,
        ann,
        trans_output,
        aug_s,
        calib,
        pre_cts=None,
        track_ids=None,
    ):
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        if h <= 0 or w <= 0:
            return
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        ct = np.array(
            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32
        )
        ct_int = ct.astype(np.int32)
        ret["cat"][k] = cls_id - 1
        ret["mask"][k] = 1
        if "wh" in ret:
            ret["wh"][k] = 1.0 * w, 1.0 * h
            ret["wh_mask"][k] = 1
        ret["ind"][k] = ct_int[1] * self.opt.output_w + ct_int[0]
        ret["reg"][k] = ct - ct_int
        ret["reg_mask"][k] = 1
        draw_umich_gaussian(ret["hm"][cls_id - 1], ct_int, radius)

        gt_det["bboxes"].append(
            np.array(
                [ct[0] - w / 2, ct[1] - h / 2, ct[0] + w / 2, ct[1] + h / 2],
                dtype=np.float32,
            )
        )
        gt_det["scores"].append(1)
        gt_det["clses"].append(cls_id - 1)
        gt_det["cts"].append(ct)

        if "tracking" in self.opt.heads:

            if ann["track_id"] in track_ids:
                pre_ct = pre_cts[track_ids.index(ann["track_id"])]
                ret["tracking_mask"][k] = 1
                ret["tracking"][k] = 0 * (pre_ct - ct_int)

                gt_det["tracking"].append(ret["tracking"][k])
            else:
                gt_det["tracking"].append(np.zeros(2, np.float32))

        if "ltrb" in self.opt.heads:
            ret["ltrb"][k] = (
                bbox[0] - ct_int[0],
                bbox[1] - ct_int[1],
                bbox[2] - ct_int[0],
                bbox[3] - ct_int[1],
            )
            ret["ltrb_mask"][k] = 1

        if "ltrb_amodal" in self.opt.heads:
            ret["ltrb_amodal"][k] = (
                bbox_amodal[0] - ct_int[0],
                bbox_amodal[1] - ct_int[1],
                bbox_amodal[2] - ct_int[0],
                bbox_amodal[3] - ct_int[1],
            )
            ret["ltrb_amodal_mask"][k] = 1
            gt_det["ltrb_amodal"].append(bbox_amodal)

        if "nuscenes_att" in self.opt.heads:
            if ("attributes" in ann) and ann["attributes"] > 0:
                att = int(ann["attributes"] - 1)
                ret["nuscenes_att"][k][att] = 1
                ret["nuscenes_att_mask"][k][self.nuscenes_att_range[att]] = 1
            gt_det["nuscenes_att"].append(ret["nuscenes_att"][k])

        if "velocity" in self.opt.heads:
            if ("velocity" in ann) and min(ann["velocity"]) > -1000:
                ret["velocity"][k] = np.array(ann["velocity"], np.float32)[:3]
                ret["velocity_mask"][k] = 1
            gt_det["velocity"].append(ret["velocity"][k])

        if "hps" in self.opt.heads:
            self._add_hps(ret, k, ann, gt_det, trans_output, ct_int, bbox, h, w)

        if "rot" in self.opt.heads:
            self._add_rot(ret, ann, k, gt_det)

        if "dep" in self.opt.heads:
            if "depth" in ann:
                ret["dep_mask"][k] = 1
                ret["dep"][k] = ann["depth"] * aug_s
                gt_det["dep"].append(ret["dep"][k])
            else:
                gt_det["dep"].append(2)

        if "dim" in self.opt.heads:
            if "dim" in ann:
                ret["dim_mask"][k] = 1
                ret["dim"][k] = ann["dim"]
                gt_det["dim"].append(ret["dim"][k])
            else:
                gt_det["dim"].append([1, 1, 1])

        if "amodel_offset" in self.opt.heads:
            if "amodel_center" in ann:
                amodel_center = affine_transform(ann["amodel_center"], trans_output)
                ret["amodel_offset_mask"][k] = 1
                ret["amodel_offset"][k] = amodel_center - ct_int
                gt_det["amodel_offset"].append(ret["amodel_offset"][k])
            else:
                gt_det["amodel_offset"].append([0, 0])

    def _add_hps(self, ret, k, ann, gt_det, trans_output, ct_int, bbox, h, w):
        num_joints = self.num_joints
        pts = (
            np.array(ann["keypoints"], np.float32).reshape(num_joints, 3)
            if "keypoints" in ann
            else np.zeros((self.num_joints, 3), np.float32)
        )
        if self.opt.simple_radius > 0:
            hp_radius = int(simple_radius(h, w, min_overlap=self.opt.simple_radius))
        else:
            hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            hp_radius = max(0, int(hp_radius))

        for j in range(num_joints):
            pts[j, :2] = affine_transform(pts[j, :2], trans_output)
            if pts[j, 2] > 0:
                if (
                    pts[j, 0] >= 0
                    and pts[j, 0] < self.opt.output_w
                    and pts[j, 1] >= 0
                    and pts[j, 1] < self.opt.output_h
                ):
                    ret["hps"][k, j * 2 : j * 2 + 2] = pts[j, :2] - ct_int
                    ret["hps_mask"][k, j * 2 : j * 2 + 2] = 1
                    pt_int = pts[j, :2].astype(np.int32)
                    ret["hp_offset"][k * num_joints + j] = pts[j, :2] - pt_int
                    ret["hp_ind"][k * num_joints + j] = (
                        pt_int[1] * self.opt.output_w + pt_int[0]
                    )
                    ret["hp_offset_mask"][k * num_joints + j] = 1
                    ret["hm_hp_mask"][k * num_joints + j] = 1
                    ret["joint"][k * num_joints + j] = j
                    draw_umich_gaussian(ret["hm_hp"][j], pt_int, hp_radius)
                    if pts[j, 2] == 1:
                        ret["hm_hp"][j, pt_int[1], pt_int[0]] = self.ignore_val
                        ret["hp_offset_mask"][k * num_joints + j] = 0
                        ret["hm_hp_mask"][k * num_joints + j] = 0
                else:
                    pts[j, :2] *= 0
            else:
                pts[j, :2] *= 0
                self._ignore_region(
                    ret["hm_hp"][
                        j,
                        int(bbox[1]) : int(bbox[3]) + 1,
                        int(bbox[0]) : int(bbox[2]) + 1,
                    ]
                )
        gt_det["hps"].append(pts[:, :2].reshape(num_joints * 2))

    def _add_rot(self, ret, ann, k, gt_det):
        if "alpha" in ann:
            ret["rot_mask"][k] = 1
            alpha = ann["alpha"]
            if alpha < np.pi / 6.0 or alpha > 5 * np.pi / 6.0:
                ret["rotbin"][k, 0] = 1
                ret["rotres"][k, 0] = alpha - (-0.5 * np.pi)
            if alpha > -np.pi / 6.0 or alpha < -5 * np.pi / 6.0:
                ret["rotbin"][k, 1] = 1
                ret["rotres"][k, 1] = alpha - (0.5 * np.pi)
            gt_det["rot"].append(self._alpha_to_8(ann["alpha"]))
        else:
            gt_det["rot"].append(self._alpha_to_8(0))

    def _alpha_to_8(self, alpha):
        ret = [0, 0, 0, 1, 0, 0, 0, 1]
        if alpha < np.pi / 6.0 or alpha > 5 * np.pi / 6.0:
            r = alpha - (-0.5 * np.pi)
            ret[1] = 1
            ret[2], ret[3] = np.sin(r), np.cos(r)
        if alpha > -np.pi / 6.0 or alpha < -5 * np.pi / 6.0:
            r = alpha - (0.5 * np.pi)
            ret[5] = 1
            ret[6], ret[7] = np.sin(r), np.cos(r)
        return ret

    def _format_gt_det(self, gt_det):
        if len(gt_det["scores"]) == 0:
            gt_det = {
                "bboxes": np.array([[0, 0, 1, 1]], dtype=np.float32),
                "scores": np.array([1], dtype=np.float32),
                "clses": np.array([0], dtype=np.float32),
                "cts": np.array([[0, 0]], dtype=np.float32),
                "pre_cts": np.array([[0, 0]], dtype=np.float32),
                "tracking": np.array([[0, 0]], dtype=np.float32),
                "bboxes_amodal": np.array([[0, 0]], dtype=np.float32),
                "hps": np.zeros((1, 17, 2), dtype=np.float32),
            }
        gt_det = {k: np.array(gt_det[k], dtype=np.float32) for k in gt_det}
        return gt_det

    def fake_video_data(self):
        self.coco.dataset["videos"] = []
        for i in range(len(self.coco.dataset["images"])):
            img_id = self.coco.dataset["images"][i]["id"]
            self.coco.dataset["images"][i]["video_id"] = img_id
            self.coco.dataset["images"][i]["frame_id"] = 1
            self.coco.dataset["videos"].append({"id": img_id})

        if not ("annotations" in self.coco.dataset):
            return

        for i in range(len(self.coco.dataset["annotations"])):
            self.coco.dataset["annotations"][i]["track_id"] = i + 1
