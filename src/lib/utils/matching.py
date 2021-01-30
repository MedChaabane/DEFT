


import lap
import numpy as np
import scipy
from cython_bbox import bbox_overlaps as bbox_ious
from scipy.spatial.distance import cdist
from utils.tracking_utils import kalman_filter
import os.path, copy
from scipy.spatial import ConvexHull

# This function is taken from : https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/tracker/matching.py
def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b

# This function is taken from : https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/tracker/matching.py
def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

# This function is taken from : https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/tracker/matching.py
def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious

# Part of this function is taken from : https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/tracker/matching.py
def iou_distance(atracks, btracks,frame_id=0,use_prediction=True):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        if use_prediction:
        
            atlbrs = [track.prediction_at_frame_tlbr(frame_id) for track in atracks]
        else:
            atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def iou_ddd_distance(atracks, btracks,frame_id=0,use_prediction=True):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        
        atlbrs = [convert_3dbox_to_8corner(track.ddd_bbox) for track in atracks]
        btlbrs = [convert_3dbox_to_8corner(track.ddd_bbox) for track in btracks]
        
    iou_matrix = np.zeros((len(atlbrs),len(btlbrs)),dtype=np.float32)
    if iou_matrix.size == 0:
        return iou_matrix
    for d,det in enumerate(btlbrs):
        for t,trk in enumerate(atlbrs):
            iou_matrix[t,d] = iou3d(det,trk)[0]             # det: 8 x 3, trk: 8 x 3
    iou_matrix = 1 - iou_matrix

    return iou_matrix

def poly_area(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c
def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0  
    
    
def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.
   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**
   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)
def convert_3dbox_to_8corner(bbox3d_input):
 
    # compute rotational matrix around yaw axis
    bbox3d = copy.copy(bbox3d_input)
    # transform to kitti format first
    bbox3d_nuscenes = copy.copy(bbox3d)
    # kitti:    [x,  y,  z,  a, l, w, h]
    bbox3d[0] =  bbox3d_nuscenes[3]
    bbox3d[1] = bbox3d_nuscenes[4]
    bbox3d[2] = bbox3d_nuscenes[5]
    bbox3d[3] = bbox3d_nuscenes[-1]
    bbox3d[4] =  bbox3d_nuscenes[2]
    bbox3d[5] =  bbox3d_nuscenes[1]
    bbox3d[6] =  bbox3d_nuscenes[0]
    

    R = roty(bbox3d[3])    

    # 3d bounding box dimensions
    l = bbox3d[4]
    w = bbox3d[5]
    h = bbox3d[6]
    
    # 3d bounding box corners
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [0,0,0,0,-h,-h,-h,-h];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    
    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + bbox3d[0]
    corners_3d[1,:] = corners_3d[1,:] + bbox3d[1]
    corners_3d[2,:] = corners_3d[2,:] + bbox3d[2]
    
    return np.transpose(corners_3d)
     
def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])

def iou3d(corners1, corners2):
    ''' Compute 3D bounding box IoU.
    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])
    inter_vol = inter_area * max(0.0, ymax-ymin)

    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)

    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=True):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > (gating_threshold+10)] = np.inf
    return cost_matrix

# Part of this function is taken from : https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/tracker/matching.py
def fuse_motion(kf, cost_matrix, tracks, detections,frame_id,use_lstm=True, only_position=True, lambda_=0.9):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    if not use_lstm:
        measurements = np.asarray([det.to_xyah() for det in detections])
        for row, track in enumerate(tracks):
            gating_distance = kf.gating_distance(
                track.mean, track.covariance, measurements, only_position, metric='maha')
            cost_matrix[row, gating_distance > 5.0*gating_threshold] = np.inf
            cost_matrix[row] = lambda_ * cost_matrix[row] + 0.05*(1 - lambda_) * gating_distance
    else:
        measurements = np.asarray([det.to_xyah() for det in detections])
        for row, track in enumerate(tracks):
            if len(track.observations)>=300:
                gating_distance = kf.gating_distance(
                    track.prediction_at_frame(frame_id), track.covariance, measurements, only_position, metric='maha')
                cost_matrix[row, gating_distance > 5.0*gating_threshold] = np.inf
                cost_matrix[row] = lambda_ * cost_matrix[row] + 0.05*(1 - lambda_) * gating_distance
            else:
                gating_distance = kf.gating_distance(
                    track.prediction_at_frame(frame_id), track.covariance, measurements, only_position, metric='gaussian')
                cost_matrix[row, gating_distance > 50] = np.inf
                cost_matrix[row] = lambda_ * cost_matrix[row] + 0.0005*(1 - lambda_) * gating_distance
                
    return cost_matrix


def fuse_motion_ddd(kf, cost_matrix, tracks, detections,frame_id,use_lstm=True, only_position=False, lambda_=0.9,use_prediction=False,classe_name=None):
    if cost_matrix.size == 0:
        return cost_matrix
    
    

    gating_dim = 7
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.ddd_bbox for det in detections])
    for row, track in enumerate(tracks):

        if use_prediction:
            gating_distance = kf.gating_distance(
            track.ddd_prediction_at_frame(frame_id), track.covariance, measurements, only_position, metric='gaussian')
        else:
            gating_distance = kf.gating_distance(
            track.ddd_bbox, track.covariance, measurements, only_position, metric='gaussian')

        thr = 0.2 * track.depth
        if classe_name=='pedestrian':
            if thr<5:
                thr = 5
        else:

            if thr<10:
                thr = 10
        cost_matrix[row, gating_distance > thr] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + 0.001 * gating_distance
                
    return cost_matrix