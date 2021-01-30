# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou (https://github.com/xingyizhou/CenterTrack/blob/master/src/lib/utils/image.py)
# Then modified by Mohamed Chaabane
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils.ddd_utils import compute_box_3d, project_to_image, alpha2rot_y
from utils.ddd_utils import draw_box_3d, unproject_2d_to_3d
import numpy as np
import cv2
import random
import torch
def flip(img):
  return img[:, :, ::-1].copy()  

# @numba.jit(nopython=True, nogil=True)
def transform_preds_with_trans(coords, trans):
    # target_coords = np.concatenate(
    #   [coords, np.ones((coords.shape[0], 1), np.float32)], axis=1)
    target_coords = np.ones((coords.shape[0], 3), np.float32)
    target_coords[:, :2] = coords
    target_coords = np.dot(trans, target_coords.transpose()).transpose()
    return target_coords[:, :2]


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img,
                             trans,
                             (int(output_size[0]), int(output_size[1])),
                             flags=cv2.INTER_LINEAR)

    return dst_img

# @numba.jit(nopython=True, nogil=True)
def gaussian_radius(det_size, min_overlap=0.7):
  height, width = det_size

  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2
  return min(r1, r2, r3)


# @numba.jit(nopython=True, nogil=True)
def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    # y, x = np.arange(-m, m + 1).reshape(-1, 1), np.arange(-n, n + 1).reshape(1, -1)
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

# @numba.jit(nopython=True, nogil=True)
def draw_umich_gaussian(heatmap, center, radius, k=1):
  # import pdb; pdb.set_trace()
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
  
  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]
    
  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)
  # import pdb; pdb.set_trace()
  masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
  return heatmap

def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
  value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
  dim = value.shape[0]
  reg = np.ones((dim, diameter*2+1, diameter*2+1), dtype=np.float32) * value
  if is_offset and dim == 2:
    delta = np.arange(diameter*2+1) - radius
    reg[0] = reg[0] - delta.reshape(1, -1)
    reg[1] = reg[1] - delta.reshape(-1, 1)
  
  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]
    
  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
  masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom,
                             radius - left:radius + right]
  masked_reg = reg[:, radius - top:radius + bottom,
                      radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
    idx = (masked_gaussian >= masked_heatmap).reshape(
      1, masked_gaussian.shape[0], masked_gaussian.shape[1])
    masked_regmap = (1-idx) * masked_regmap + idx * masked_reg
  regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
  return regmap


def draw_msra_gaussian(heatmap, center, sigma):
  tmp_size = sigma * 3
  mu_x = int(center[0] + 0.5)
  mu_y = int(center[1] + 0.5)
  w, h = heatmap.shape[0], heatmap.shape[1]
  ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
  br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
  if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
    return heatmap
  size = 2 * tmp_size + 1
  x = np.arange(0, size, 1, np.float32)
  y = x[:, np.newaxis]
  x0 = y0 = size // 2
  g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
  g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
  g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
  img_x = max(0, ul[0]), min(br[0], h)
  img_y = max(0, ul[1]), min(br[1], w)
  heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
    g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
  return heatmap

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3, ))
    image += np.dot(eigvec, eigval * alpha)

def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2

def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])

def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha

def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)

def color_aug(data_rng, image, eig_val, eig_vec):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)
    lighting_(data_rng, image, 0.1, eig_val, eig_vec)
    
    
def show_matching_hanlded_rectangle(img_pre, img_next, boxes_pre, boxes_next, labels):
    img_p = img_pre.copy()
    img_n = img_next.copy()
    for box in boxes_pre[:, 0:4]:
        img_p = cv2.rectangle(img_p, tuple(box[:2].astype(int)), tuple((box[2:4]).astype(int)), (255, 0, 0), 2)

    for box in boxes_next[:, 0:4]:
        img_n = cv2.rectangle(img_n, tuple(box[:2].astype(int)), tuple((box[2:4]).astype(int)), (255, 0, 0), 2)

    h, w, c = img_p.shape
    h, w, c = img_n.shape
    img = np.concatenate([img_p, img_n], axis=0)
    rows, cols = np.nonzero(labels)
    for r, c in zip(rows, cols):
        box_p = boxes_pre[r, 0:4]
        box_n = boxes_next[c, 0:4]
        center_p = (box_p[:2] + box_p[2:4]) / 2.0
        center_n = (box_n[:2] + box_n[2:4]) / 2.0 + np.array([0, h])
        img = cv2.line(img, tuple(center_p.astype(int)), tuple(center_n.astype(int)),
                 ((int)(np.random.randn() * 255), (int)(np.random.randn() * 255), (int)(np.random.randn() * 255)),
                 2)
    return img

def ResizeShuffleBoxes( max_object, boxes_pre, boxes_next, labels):

    

    resize_f = lambda boxes: \
            (boxes.shape[0],
             np.vstack((
                 boxes,
                 np.full(
                     (max_object - len(boxes),
                      boxes.shape[1]),
                     np.inf
                 ))))
       
    size_pre, boxes_pre = resize_f(boxes_pre)
    size_next, boxes_next = resize_f(boxes_next)

    indexes_pre = np.arange(max_object)
    indexes_next = np.arange(max_object)
    np.random.shuffle(indexes_pre)
    np.random.shuffle(indexes_next)

    boxes_pre = boxes_pre[indexes_pre, :]
    boxes_next = boxes_next[indexes_next, :]

    labels = labels[indexes_pre, :]
    labels = labels[:, indexes_next]

    mask_pre = indexes_pre < size_pre
    mask_next = indexes_next < size_next

    # add false object label
    false_object_pre = (labels.sum(1) == 0).astype(float)      
    false_object_pre[np.logical_not(mask_pre)] = 0.0

    false_object_next = (labels.sum(0) == 0).astype(float) 
    false_object_next[np.logical_not(mask_next)] = 0.0

    false_object_pre = np.expand_dims(false_object_pre, axis=1)
    labels = np.concatenate((labels, false_object_pre), axis=1) #60x61

    false_object_next = np.append(false_object_next, [0])
    false_object_next = np.expand_dims(false_object_next, axis=0)
    labels = np.concatenate((labels, false_object_next), axis=0)  # 60x61

    mask_pre = np.append(mask_pre, [True])  # 61
    mask_next = np.append(mask_next, [True]) # 61
    return [boxes_pre, mask_pre], \
           [boxes_next, mask_next], \
           labels


def FormatBoxes( boxes_pre, boxes_next, labels ):
        # convert the center to [-1, 1]
    f = lambda boxes: np.expand_dims(
        np.expand_dims(
            (boxes[:, :2] + boxes[:, 2:]) - 1,
            axis=1
        ),
        axis=1
    )
   

    # remove inf
    boxes_pre[0] = f(boxes_pre[0])
    boxes_pre[0][boxes_pre[0] == np.inf] = 1.5

    boxes_next[0] = f(boxes_next[0])
    boxes_next[0][boxes_next[0] == np.inf] = 1.5

    return boxes_pre, boxes_next, labels

    

def ToTensor(boxes_pre, boxes_next, labels
        ):
        boxes_pre[0] = torch.from_numpy(boxes_pre[0].astype(float)).float()
        boxes_pre[1] = torch.from_numpy(boxes_pre[1].astype(np.uint8)).unsqueeze(0)

        boxes_next[0] = torch.from_numpy(boxes_next[0].astype(float)).float()
        boxes_next[1] = torch.from_numpy(boxes_next[1].astype(np.uint8)).unsqueeze(0)

        labels = torch.from_numpy(labels).unsqueeze(0)

        return  boxes_pre[0], boxes_pre[1], boxes_next[0], boxes_next[1], labels   
    
    
def ToPercentCoordinates(boxes_pre, boxes_next,img):
    height, width, channels = img.shape
    boxes_pre[:, 0] /= width
    boxes_pre[:, 2] /= width
    boxes_pre[:, 1] /= height
    boxes_pre[:, 3] /= height

    boxes_next[:, 0] /= width
    boxes_next[:, 2] /= width
    boxes_next[:, 1] /= height
    boxes_next[:, 3] /= height
    
    return boxes_pre, boxes_next

def convert_detection(detection,h,w):
    '''
    transform the current detection center to [-1, 1]
    :param detection: detection
    :return: translated detection
    '''
    # get the center, and format it in (-1, 1)
    detection[:, 2] -= detection[:, 0]
    detection[:, 3] -= detection[:, 1]
    detection[:, 0] /= w
    detection[:, 2] /= w
    detection[:, 1] /= h
    detection[:, 3] /= h
    center = (2 * detection[:, 0:2] + detection[:, 2:4]) - 1.0
    center = torch.from_numpy(center.astype(float)).float()
    center.unsqueeze_(0)
    center.unsqueeze_(2)
    center.unsqueeze_(3)

    if torch.cuda.is_available():
        return center.cuda()
    return center

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = max(1, image.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(image.shape[1] / 500.))

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        _line_thickness = 1 if obj_id <= 0 else line_thickness
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im

def plot_tracking_ddd(image, tlwhs, ddd_boxes, obj_ids, scores=None, frame_id=0, fps=0., ids2=None,calib=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = max(1, image.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(image.shape[1] / 500.))

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(ddd_boxes)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)
    
    
    for i,box3d in enumerate(ddd_boxes):
        tlwh = tlwhs[i]
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        
        
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        _line_thickness = 1 if obj_id <= 0 else line_thickness
        color = get_color(abs(obj_id))
        dim = box3d[:3]
        loc= box3d[3:-1]
        rot= box3d[-1]
        
        box_3d = compute_box_3d(dim, loc, rot)
        box_2d = project_to_image(box_3d, calib)
        im = draw_box_3d(im, box_2d,c=color,same_color=True)
        cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    
    return im
