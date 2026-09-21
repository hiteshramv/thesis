from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Mapping

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

from datastructures.lidar_detections import Object3dList, Object3d
from datastructures.calibration_manager import CalibrationManager

import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def _stats(x: np.ndarray) -> str:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        return "n=0"
    return (f"n={x.size} min={x.min():.3f} p25={np.percentile(x,25):.3f} "
            f"med={np.median(x):.3f} p75={np.percentile(x,75):.3f} max={x.max():.3f} mean={x.mean():.3f}")
# ============================================================
# Helpers: bbox + scores
# ============================================================

def xywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    return (float(cx - w / 2), float(cy - h / 2), float(cx + w / 2), float(cy + h / 2))


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return float(inter / denom)


def size_ratio(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    return float(min(area_a, area_b) / max(area_a, area_b))


def aspect_product(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    wa = max(1e-6, a[2] - a[0])
    ha = max(1e-6, a[3] - a[1])
    wb = max(1e-6, b[2] - b[0])
    hb = max(1e-6, b[3] - b[1])
    ra = ha / wa
    rb = hb / wb
    return float(ra * rb)


def normalized_center_distance(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """
    d = 1 - (||centerA-centerB|| / diag_of_enclosing_box)
    """
    ca = np.array([(a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5], dtype=np.float32)
    cb = np.array([(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5], dtype=np.float32)

    rho = float(np.linalg.norm(ca - cb))

    enc_min = np.array([min(a[0], b[0]), min(a[1], b[1])], dtype=np.float32)
    enc_max = np.array([max(a[2], b[2]), max(a[3], b[3])], dtype=np.float32)
    c = float(np.linalg.norm(enc_max - enc_min))

    if c <= 1e-6:
        return 0.0

    return float(1.0 - (rho / c))


def sdiou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    # SDIoU = IoU + (aspect_product * size_ratio * normalized_center_distance)
    return float(iou_xyxy(a, b) + aspect_product(a, b) * size_ratio(a, b) * normalized_center_distance(a, b))


def box_area_xyxy(b: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = b
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


# ============================================================
# Projection: 3D bbox (ouster/world) -> camera pixels
# ============================================================

def compute_3d_bbox_corners_centered(
    cx: float, cy: float, cz: float,
    yaw: float,
    sx: float, sy: float, sz: float
) -> np.ndarray:
    """
    Build 8 corners of 3D box around center (cx,cy,cz) with size (sx,sy,sz).
    Yaw rotation around +Z.
    Returns (8,3) float32 in LiDAR/world frame.
    """
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5

    corners = np.array([
        [ hx,  hy,  hz],
        [ hx, -hy,  hz],
        [-hx, -hy,  hz],
        [-hx,  hy,  hz],
        [ hx,  hy, -hz],
        [ hx, -hy, -hz],
        [-hx, -hy, -hz],
        [-hx,  hy, -hz],
    ], dtype=np.float32)

    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    Rz = np.array([
        [ c, -s, 0.0],
        [ s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    rotated = (Rz @ corners.T).T
    rotated[:, 0] += float(cx)
    rotated[:, 1] += float(cy)
    rotated[:, 2] += float(cz)
    return rotated


class MultiCamProjector:
    """
    Uses CalibrationManager + your calibration_data format.

    Expected:
      - intrinsics: 3x3
      - distortion: [k1,k2,p1,p2,k3]
      - extrinsic:  3x4  (Ouster -> Camera), i.e. X_cam = R * X_ouster + t
      - image size: width/height from calibration_data
    """

    def __init__(self, calibration_data: dict):
        self.cm = CalibrationManager(calibration_data)
        self.calib = calibration_data

    def _img_size(self, cam_id: int) -> Tuple[int, int]:
        cam_key = f"camera_{cam_id:02d}"
        w = int(self.calib["cameras"][cam_key]["image_width"])
        h = int(self.calib["cameras"][cam_key]["image_height"])
        return w, h

    def project_object_to_xyxy(self, obj3d: Object3d, cam_id: int) -> Optional[Tuple[float, float, float, float]]:
        K = self.cm.get_intrinsic(cam_id)                   
        dist = self.cm.get_distortion(cam_id).reshape(-1)   
        Ext = self.cm.get_extrinsic(cam_id)                 
        w_img, h_img = self._img_size(cam_id)

        R = Ext[:, :3].astype(np.float32)                   
        t = Ext[:, 3].astype(np.float32).reshape(3, 1)      

        cx = float(obj3d.position.x)
        cy = float(obj3d.position.y)
        cz = float(obj3d.position.z)

        sx = float(obj3d.size.x)
        sy = float(obj3d.size.y)
        sz = float(obj3d.size.z)

        yaw = 0.0
        if hasattr(obj3d, "yaw_angle") and obj3d.yaw_angle is not None:
            yaw = float(obj3d.yaw_angle)

        if sx <= 1e-3 or sy <= 1e-3 or sz <= 1e-3:
            return None

        corners_ouster = compute_3d_bbox_corners_centered(cx, cy, cz, yaw, sx, sy, sz)

        X = corners_ouster.T.astype(np.float32)   
        corners_cam = (R @ X) + t                 
        corners_cam = corners_cam.T               

        if np.all(corners_cam[:, 2] <= 1e-6):
            return None

        pts = corners_cam.reshape(-1, 1, 3).astype(np.float32)

        #rvec, _ = cv2.Rodrigues(R)
        #tvec = t.reshape(3, 1)

        rvec = np.zeros((3, 1), dtype=np.float32)
        tvec = np.zeros((3, 1), dtype=np.float32)

        img_pts, _ = cv2.projectPoints(pts, rvec, tvec, K.astype(np.float32), dist.astype(np.float32))
        img_pts = img_pts.reshape(-1, 2)

        valid = corners_cam[:, 2] > 1e-6
        img_pts_valid = img_pts[valid]
        if img_pts_valid.shape[0] < 2:
            return None

        x1 = float(np.min(img_pts_valid[:, 0]))
        y1 = float(np.min(img_pts_valid[:, 1]))
        x2 = float(np.max(img_pts_valid[:, 0]))
        y2 = float(np.max(img_pts_valid[:, 1]))

        if x2 < 0 or y2 < 0 or x1 > w_img or y1 > h_img:
            return None

        x1 = max(0.0, min(x1, float(w_img - 1)))
        y1 = max(0.0, min(y1, float(h_img - 1)))
        x2 = max(0.0, min(x2, float(w_img - 1)))
        y2 = max(0.0, min(y2, float(h_img - 1)))

        if (x2 - x1) < 2.0 or (y2 - y1) < 2.0:
            return None

        return (x1, y1, x2, y2)


# ============================================================
# Association result structures
# ============================================================

@dataclass
class CamAssocResult:
    camera_id: int
    matches: List[Tuple[int, int, float]]   # (lidar_idx, det_idx, score)
    unmatched_lidar: List[int]
    unmatched_cam: List[int]


# ============================================================
# Sensor Fusion (Multi-camera)
# ============================================================

class SensorFusion:
    """
    Measurement-level fusion:
      - Associate LiDAR 3D objects with YOLO 2D detections using Hungarian on (SD)IoU
      - If the same LiDAR object matches across multiple cameras:
          choose the camera where projected 3D bbox area is largest
          tie-breaker: higher IoU/SDIoU score
      - Fuse semantics: update LiDAR Object3d.class_name & category_confidence from YOLO

    Input:
      - lidar_list: Object3dList
      - yolo_by_cam: Dict[int, ImageDetectionList]

    Output:
      - fused_lidar_list: Object3dList
      - assoc_debug: Dict[int, CamAssocResult]
    """

    def __init__(
        self,
        projector: MultiCamProjector,
        score_fn: str = "iou",   # "iou" or "sdiou"
        threshold: float = 0.4,
        resolve_global_unique_lidar: bool = True,
    ):
        if score_fn not in ("iou", "sdiou"):
            raise ValueError("score_fn must be 'iou' or 'sdiou'")
        self.projector = projector
        self.score_fn = score_fn
        self.threshold = float(threshold)
        self.resolve_global_unique_lidar = bool(resolve_global_unique_lidar)

    def _score(self, a_xyxy, b_xyxy) -> float:
        return iou_xyxy(a_xyxy, b_xyxy) if self.score_fn == "iou" else sdiou(a_xyxy, b_xyxy)

    def associate_one_camera(self, lidar_objects: List[Object3d], det_list) -> CamAssocResult:
        logger = logging.getLogger("fusion")
        cam_id = int(det_list.camera_id)
        dets = det_list.detections

        nL = len(lidar_objects)
        nD = len(dets)

        if nL == 0 or nD == 0:
            return CamAssocResult(cam_id, [], list(range(nL)), list(range(nD)))

        det_boxes = [xywh_to_xyxy(d.x, d.y, d.w, d.h) for d in dets]
        score_mat = np.zeros((nL, nD), dtype=np.float32)

        for li, obj in enumerate(lidar_objects):
            proj = self.projector.project_object_to_xyxy(obj, cam_id)
            if proj is None:
                continue
            for di, db in enumerate(det_boxes):
                score_mat[li, di] = self._score(proj, db)

        row, col = linear_sum_assignment(-score_mat)  # maximize score
        matches: List[Tuple[int, int, float]] = []
        used_l = set()
        used_d = set()

        for r, c in zip(row, col):
            s = float(score_mat[r, c])
            if s >= self.threshold:
                matches.append((int(r), int(c), s))
                used_l.add(int(r))
                used_d.add(int(c))
    

        unmatched_lidar = [i for i in range(nL) if i not in used_l]
        unmatched_cam = [j for j in range(nD) if j not in used_d]

        return CamAssocResult(cam_id, matches, unmatched_lidar, unmatched_cam)

    def fuse_object3d_list(
        self,
        lidar_list: Object3dList,
        yolo_by_cam: Dict[int, object],
    ) -> Tuple[Object3dList, Dict[int, CamAssocResult], Dict[int, Dict[str, Any]]]:

        lidar_objects = list(lidar_list.objects_3d) 
        assoc_debug: Dict[int, CamAssocResult] = {}

        candidates: List[Tuple[float, float, int, int, int, Tuple[float, float, float, float]]] = []

        for cam_id, det_list in yolo_by_cam.items():
            res = self.associate_one_camera(lidar_objects, det_list)
            assoc_debug[int(cam_id)] = res

            det_boxes = [xywh_to_xyxy(d.x, d.y, d.w, d.h) for d in det_list.detections]

            for (li, di, s) in res.matches:
                proj = self.projector.project_object_to_xyxy(lidar_objects[li], int(cam_id))
                if proj is None:
                    continue

                det_xyxy = det_boxes[int(di)]
                area = box_area_xyxy(proj)

                candidates.append((
                    float(area), float(s),
                    int(cam_id), int(li), int(di),
                    proj, det_xyxy
                ))

        chosen: List[Tuple[int, int, int, float]] = []  # (cam_id, lidar_idx, det_idx, score)

        if self.resolve_global_unique_lidar:
            candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

            used_lidar = set()
            used_det = set()  # (cam_id, det_idx)

            for area, s, cam_id, li, di, _proj, _det_xyxy in candidates:
                if li in used_lidar:
                    continue
                if (cam_id, di) in used_det:
                    continue
                chosen.append((cam_id, li, di, s))
                used_lidar.add(li)
                used_det.add((cam_id, di))
        else:
            chosen = [(cam_id, li, di, s) for (area, s, cam_id, li, di, _proj, _det_xyxy) in candidates]

        fusion_meta: Dict[int, Dict[str, Any]] = {}

        for cam_id, li, di, s in chosen:
            det_list = yolo_by_cam[cam_id]
            det = det_list.detections[int(di)]

            det_xyxy = xywh_to_xyxy(det.x, det.y, det.w, det.h)
            proj = self.projector.project_object_to_xyxy(lidar_objects[int(li)], int(cam_id))
            if proj is None:
                continue

            area = box_area_xyxy(proj)
            conf = float(det.score) * float(s)

            fusion_meta[int(li)] = {
                "cam_id": int(cam_id),
                "det_idx": int(di),
                "score": float(s),
                "conf": float(conf),
                "proj_xyxy": (float(proj[0]), float(proj[1]), float(proj[2]), float(proj[3])),
                "det_xyxy": (float(det_xyxy[0]), float(det_xyxy[1]), float(det_xyxy[2]), float(det_xyxy[3])),
                "proj_area": float(area),
                "class_name": getattr(det, "class_name", None),
                "det_score": float(getattr(det, "score", 0.0) or 0.0),
            }


        for cam_id, li, di, s in chosen:
            det_list = yolo_by_cam[cam_id]
            det = det_list.detections[di]
            obj = lidar_objects[li]

            #if obj.class_name == det.class_name:
            #    obj.class_name = det.class_name
            
            obj.class_name = det.class_name

            # confidence update: keep max(existing, det.score * match_score)
            old = float(getattr(obj, "category_confidence", 0.0) or 0.0)
            new = float(det.score) * float(s)
            obj.category_confidence = float(max(old, new))

        fused = Object3dList(time_stamp=lidar_list.time_stamp, objects_3d=lidar_objects)
        return fused, assoc_debug, fusion_meta