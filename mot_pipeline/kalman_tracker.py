from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Mapping
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import numpy as np

from motion_models import make_kf, set_kf_dt, correct_new_angle_and_diff , set_kf_q
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def _stats(x: np.ndarray) -> str:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        return "n=0"
    return (f"n={x.size} min={x.min():.3f} p25={np.percentile(x,25):.3f} "
            f"med={np.median(x):.3f} p75={np.percentile(x,75):.3f} max={x.max():.3f} mean={x.mean():.3f}")

@dataclass
class TrackOutput:
    track_id: int
    state: np.ndarray
    age: int
    hits: int
    time_since_update: int
    class_name: str = "unknown"
    last_bbox2d_by_cam: Optional[Dict[int, Tuple[float, float, float, float]]] = None


# ============================================================
# Geometry: 2D IoU
# ============================================================

def iou2d_xyxy(a: Tuple[float, float, float, float],
              b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else float(inter / union)


def area_xyxy(bb: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bb
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


# ============================================================
# Geometry: 3D IoU
# ============================================================

def _rect_corners_bev(cx: float, cy: float, l: float, w: float, yaw: float) -> np.ndarray:
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    dx, dy = l / 2.0, w / 2.0
    corners = np.array([[ dx,  dy],
                        [ dx, -dy],
                        [-dx, -dy],
                        [-dx,  dy]], dtype=np.float32)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    rot = corners @ R.T
    rot[:, 0] += cx
    rot[:, 1] += cy
    return rot


def _bev_intersection_area(boxA: Tuple[float, float, float, float, float],
                           boxB: Tuple[float, float, float, float, float]) -> float:
    
    ax, ay, al, aw, ayaw = boxA
    bx, by, bl, bw, byaw = boxB
    ax1, ay1, ax2, ay2 = ax - al/2.0, ay - aw/2.0, ax + al/2.0, ay + aw/2.0
    bx1, by1, bx2, by2 = bx - bl/2.0, by - bw/2.0, bx + bl/2.0, by + bw/2.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return float(max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1))


def scaled_dist(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    
    
    ax, ay, az = map(float, a["center_xyz"])
    al, aw, ah = map(float, a["dims_lwh"])
    ayaw = float(a["yaw"])

    bx, by, bz = map(float, b["center_xyz"])
    bl, bw, bh = map(float, b["dims_lwh"])
    byaw = float(b["yaw"])

    rho_a = np.array([ax, ay, az, ah, aw, al], dtype=np.float32)
    rho_b = np.array([bx, by, bz, bh, bw, bl], dtype=np.float32)
    dist = float(np.linalg.norm(rho_a - rho_b))

    _, angle_diff = correct_new_angle_and_diff(ayaw, byaw)

    angle_diff = float(abs(angle_diff))
    if angle_diff > np.pi / 2:
        angle_diff = np.pi - angle_diff 
    yaw_weight = 0.2
    alpha = 1.0 + yaw_weight*(1.0 - float(np.cos(angle_diff)))
    return dist * alpha


def iou3d_oriented(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """
    a/b dict:
      center_xyz: (x,y,z)
      dims_lwh:   (l,w,h)
      yaw:        radians
    """
    ax, ay, az = map(float, a["center_xyz"])
    al, aw, ah = map(float, a["dims_lwh"])
    ayaw = float(a["yaw"])

    bx, by, bz = map(float, b["center_xyz"])
    bl, bw, bh = map(float, b["dims_lwh"])
    byaw = float(b["yaw"])

    if al <= 1e-6 or aw <= 1e-6 or ah <= 1e-6:
        return 0.0
    if bl <= 1e-6 or bw <= 1e-6 or bh <= 1e-6:
        return 0.0

    inter_area = _bev_intersection_area((ax, ay, al, aw, ayaw), (bx, by, bl, bw, byaw))
    if inter_area <= 0.0:
        return 0.0

    a_zmin, a_zmax = az - ah/2.0, az + ah/2.0
    b_zmin, b_zmax = bz - bh/2.0, bz + bh/2.0
    z_inter = max(0.0, min(a_zmax, b_zmax) - max(a_zmin, b_zmin))
    if z_inter <= 0.0:
        return 0.0

    inter_vol = inter_area * z_inter
    vol_a = al * aw * ah
    vol_b = bl * bw * bh
    union = vol_a + vol_b - inter_vol
    return 0.0 if union <= 0.0 else float(inter_vol / union)


# ============================================================
# Data association 
# ============================================================


def perform_association_from_similarity_hungarian(
    n_dets: int,
    n_trks: int,
    sim: np.ndarray,
    threshold: float = 0.0
) -> Tuple[np.ndarray, List[int], List[int]]:
    

    if n_trks == 0:
        return np.empty((0, 2), dtype=int), list(range(n_dets)), []
    if n_dets == 0:
        return np.empty((0, 2), dtype=int), [], list(range(n_trks))

    BIG = 1e6
    cost = 1.0 - sim                 
    cost[sim < threshold] = BIG

    row_ind, col_ind = linear_sum_assignment(cost)

    matched = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] >= BIG:
            continue
        matched.append([int(r), int(c)])

    matched = np.asarray(matched, dtype=int)

    unmatched_d = [i for i in range(n_dets) if i not in matched[:, 0]]
    unmatched_t = [j for j in range(n_trks) if j not in matched[:, 1]]

    return matched, unmatched_d, unmatched_t



def greedy_match(cost_matrix: np.ndarray) -> np.ndarray:
    
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int)

    n_d, n_t = cost_matrix.shape
    flat = cost_matrix.reshape(-1)
    order = np.argsort(flat)
    pairs = np.stack([order // n_t, order % n_t], axis=1).astype(int, copy=False)

    used_d, used_t = set(), set()
    matched = []
    for di, ti in pairs:
        if di in used_d or ti in used_t:
            continue
        used_d.add(int(di))
        used_t.add(int(ti))
        matched.append([int(di), int(ti)])

    return np.asarray(matched, dtype=int)


def perform_association_from_similarity(n_dets: int,
                                       n_trks: int,
                                       sim: np.ndarray) -> Tuple[np.ndarray, List[int], List[int]]:

    if n_trks == 0:
        return np.empty((0, 2), dtype=int), list(range(n_dets)), []
    if n_dets == 0:
        return np.empty((0, 2), dtype=int), [], list(range(n_trks))

    cost = -sim
    matched = greedy_match(cost)

    unmatched_d = [i for i in range(n_dets) if i not in matched[:, 0]] if len(matched) else list(range(n_dets))
    unmatched_t = [j for j in range(n_trks) if j not in matched[:, 1]] if len(matched) else list(range(n_trks))
    return matched, unmatched_d, unmatched_t


def filter_matches(matched: np.ndarray,
                   unmatched_d: List[int],
                   unmatched_t: List[int],
                   sim: np.ndarray,
                   threshold: float) -> Tuple[np.ndarray, List[int], List[int]]:
    keep = []
    for di, ti in matched:
        if sim[int(di), int(ti)] < threshold:
            unmatched_d.append(int(di))
            unmatched_t.append(int(ti))
        else:
            keep.append([int(di), int(ti)])

    keep = np.asarray(keep, dtype=int) if keep else np.empty((0, 2), dtype=int)
    return keep, unmatched_d, unmatched_t


# ============================================================
# Track
# ============================================================

class Track:
    _count = 0

    def __init__(self, obj3d: Any, ts: float, motion_model: str):
        self.id = Track._count
        Track._count += 1

        self.age_total = 1
        self.hits = 1
        self.time_since_update = 0
        self.time_since_3d_update = 0
        self.time_since_2d_update = 0
        self.class_name = getattr(obj3d, "class_name", "unknown")
        self.last_ts = float(ts)
        self.kf = make_kf(motion_model)
        z = self._obj3d_to_z7(obj3d)
        self.kf.x[:7] = z.reshape(7, 1)
        self.last_bbox2d_by_cam: Dict[int, Tuple[float, float, float, float]] = {}
        self._pred_bbox2d_cache: Dict[int, Optional[Tuple[float, float, float, float]]] = {}

    def reset_for_new_frame(self):
        self.age_total += 1
        self.time_since_update += 1
        self.time_since_3d_update += 1
        self.time_since_2d_update += 1
        self._pred_bbox2d_cache = {}

    def predict_motion(self, ts: float):
        dt = max(1e-3, float(ts - self.last_ts))
        self.last_ts = float(ts)

        set_kf_dt(self.kf, dt)
        set_kf_q(self.kf, dt)
        self.kf.predict()

    def update_with_3d(self, obj3d: Any):
        z = self._obj3d_to_z7(obj3d)

        new_cls = getattr(obj3d, "class_name", None)
        if new_cls not in (None, "", "unknown"):
            self.class_name = new_cls

        new_yaw = float(z[3])
        new_yaw_corr, _ = correct_new_angle_and_diff(float(self.kf.x[3]), new_yaw)
        z[3] = new_yaw_corr

        self.kf.update(z.reshape(self.kf.dim_z, 1))

        self.hits += 1
        self.time_since_update = 0
        self.time_since_3d_update = 0


    def update_with_2d(self, cam_id: int, xyxy: Tuple[float, float, float, float]):
        self.last_bbox2d_by_cam[int(cam_id)] = tuple(map(float, xyxy))
        self.time_since_update = 0
        self.time_since_2d_update = 0

    def state(self) -> np.ndarray:
        return self.kf.x.reshape(-1).astype(np.float32)

    def center_xyz(self) -> Tuple[float, float, float]:
        x = self.kf.x.reshape(-1)
        return float(x[0]), float(x[1]), float(x[2])

    def yaw_lwh(self) -> Tuple[float, Tuple[float, float, float]]:
        x = self.kf.x.reshape(-1)
        yaw = float(x[3])
        lwh = (float(x[4]), float(x[5]), float(x[6]))
        return yaw, lwh

    def predicted_box3d_dict(self) -> Dict[str, Any]:
        cx, cy, cz = self.center_xyz()
        yaw, (l, w, h) = self.yaw_lwh()
        return {"center_xyz": (cx, cy, cz), "dims_lwh": (l, w, h), "yaw": yaw}

    def projected_bbox2d_in_cam(self, cam_id: int, projector: Any) -> Optional[Tuple[float, float, float, float]]:
        cam_id = int(cam_id)
        if cam_id in self._pred_bbox2d_cache:
            return self._pred_bbox2d_cache[cam_id]

        best = self.last_bbox2d_by_cam.get(cam_id, None)

        try:
            obj3d_like = self._make_obj3d_like_for_projector()
            proj = projector.project_object_to_xyxy(obj3d_like, cam_id)
            if proj is not None:
                best = (float(proj[0]), float(proj[1]), float(proj[2]), float(proj[3]))
        except Exception:
            pass

        self._pred_bbox2d_cache[cam_id] = best
        return best

    def _make_obj3d_like_for_projector(self) -> Any:

        class _Vec:
            def __init__(self, x, y, z):
                self.x, self.y, self.z = x, y, z

        class _Obj:
            pass

        cx, cy, cz = self.center_xyz()
        yaw, (l, w, h) = self.yaw_lwh()

        o = _Obj()
        o.position = _Vec(cx, cy, cz)
        o.size = _Vec(l, w, h)
        o.yaw_angle = yaw
        return o

    @staticmethod
    def _obj3d_to_z7(obj3d: Any) -> np.ndarray:
    
        yaw = float(getattr(obj3d, "yaw_angle", 0.0)) if getattr(obj3d, "yaw_angle", None) is not None else 0.0
        return np.array([
            float(obj3d.position.x),
            float(obj3d.position.y),
            float(obj3d.position.z),
            yaw,
            float(obj3d.size.x),
            float(obj3d.size.y),
            float(obj3d.size.z),
        ], dtype=np.float32)

# ============================================================
# Two-stage Track Manager
# ============================================================

class KalmanMultiObjectTracker:
    """
    Stage-1: 3D association (3D IoU) between 3D detections and existing tracks
    Stage-2: 2D association (2D IoU) between leftover tracks and 2D-only detections (multi-cam)
             + multicam heuristic: choose cam where projected 3D area is largest.

    Lifecycle:
      - births from unmatched 3D dets
      - delete if time_since_update > max_age
    """

    def __init__(self,
                 motion_model: str,
                 max_age: int = 4,
                 min_hits: int = 3,
                 dist3d_thresh: float = 3,
                 iou2d_thresh: float = 0.3):

        self.motion_model = motion_model
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.dist3d_thresh = float(dist3d_thresh)
        self.iou2d_thresh = float(iou2d_thresh)

        self.tracks: List[Track] = []
        self.frame_count = 0


    @staticmethod
    def _obj3d_to_boxdict(obj3d: Any) -> Dict[str, Any]:
        yaw = float(getattr(obj3d, "yaw_angle", 0.0)) if getattr(obj3d, "yaw_angle", None) is not None else 0.0
        return {
            "center_xyz": (float(obj3d.position.x), float(obj3d.position.y), float(obj3d.position.z)),
            "dims_lwh": (float(obj3d.size.x), float(obj3d.size.y), float(obj3d.size.z)),
            "yaw": yaw,
        }


    @staticmethod
    def _imdet_to_xyxy(det: Any) -> Optional[Tuple[float, float, float, float]]:

        if hasattr(det, "xyxy"):
            v = getattr(det, "xyxy")
            if v is not None:
                return (float(v[0]), float(v[1]), float(v[2]), float(v[3]))

        if hasattr(det, "bbox_xyxy"):
            v = getattr(det, "bbox_xyxy")
            if v is not None:
                return (float(v[0]), float(v[1]), float(v[2]), float(v[3]))

        if all(hasattr(det, k) for k in ["x1", "y1", "x2", "y2"]):
            return (float(det.x1), float(det.y1), float(det.x2), float(det.y2))

        if all(hasattr(det, k) for k in ["x", "y", "w", "h"]):
            x, y, w, h = float(det.x), float(det.y), float(det.w), float(det.h)
            return (x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0)

        if hasattr(det, "bbox"):
            bb = getattr(det, "bbox")
            if bb is None or len(bb) != 4:
                return None
            x, y, w, h = map(float, bb)
            return (x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0)

        return None

    def update_two_stage(self,
                        dets3d: List[Any],
                        yolo_by_cam: Dict[int, Any],
                        assoc_debug: Dict[int, Any],
                        projector: Any,
                        t: float,
                        fusion_meta: Optional[Dict[int, Dict[str, Any]]] = None) -> List[TrackOutput]:

        logger = logging.getLogger("mot3d")
        self.frame_count += 1
        t = float(t)

        # reset + predict
        for trk in self.tracks:
            trk.reset_for_new_frame()
            trk.predict_motion(t)

        # if no tracks: create from all 3D detections
        if len(self.tracks) == 0:
            for obj in dets3d:
                self.tracks.append(Track(obj, t, motion_model=self.motion_model))
            self._prune()
            return self._outputs()

        # ----------------------------------------------------
        # Stage-1: scaled distance cost
        # ----------------------------------------------------
        track_boxes = [trk.predicted_box3d_dict() for trk in self.tracks]
        det_boxes = [self._obj3d_to_boxdict(obj) for obj in dets3d]

        T, D = len(track_boxes), len(det_boxes)
        cost3d = np.zeros((D, T), dtype=np.float32)  # (dets, tracks) lower is better

        for di in range(D):
            for ti in range(T):
                cost3d[di, ti] = float(scaled_dist(det_boxes[di], track_boxes[ti]))

        vals = cost3d[np.isfinite(cost3d)]

        BIG = 1e6
        cost_for_hungarian = cost3d.copy()
        cost_for_hungarian[cost3d > self.dist3d_thresh] = BIG

        row_ind, col_ind = linear_sum_assignment(cost_for_hungarian)

        matched = []
        for r, c in zip(row_ind, col_ind):
            if cost_for_hungarian[r, c] >= BIG:
                continue
            matched.append([int(r), int(c)])
        matched = np.asarray(matched, dtype=int)

        u_det = [i for i in range(D) if i not in matched[:, 0]] if len(matched) else list(range(D))
        u_trk = [j for j in range(T) if j not in matched[:, 1]] if len(matched) else list(range(T))

        for di, ti in matched:
            di = int(di)
            ti = int(ti)

            self.tracks[ti].update_with_3d(dets3d[di])

            if fusion_meta is not None:
                meta = fusion_meta.get(di, None)
                if meta is not None:
                    cam_id = int(meta["cam_id"])
                    det_xyxy = meta["det_xyxy"]
                    self.tracks[ti].update_with_2d(cam_id, det_xyxy)

        leftover_track_indices = list(u_trk)

        # ----------------------------------------------------
        # Stage-2: 2D IoU association
        # ----------------------------------------------------
        logger2 = logging.getLogger("mot2d")

        candidate_matches: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        track_proj_xyxy_by_cam: Dict[Tuple[int, int], Tuple[float, float, float, float]] = {}
        leftover_tracks = [self.tracks[i] for i in leftover_track_indices]

        for cam_id, res in assoc_debug.items():
            cam_id = int(cam_id)
            det_list = yolo_by_cam.get(cam_id, None)
            if det_list is None:
                continue

            unmatched_cam = list(getattr(res, "unmatched_cam", []))
            if len(unmatched_cam) == 0 or len(leftover_tracks) == 0:
                continue

            det_xyxys: List[Tuple[float, float, float, float]] = []
            valid_det_map: List[int] = []
            for idx in unmatched_cam:
                det = det_list.detections[int(idx)]
                xyxy = self._imdet_to_xyxy(det)
                if xyxy is None:
                    continue
                det_xyxys.append(xyxy)
                valid_det_map.append(int(idx))

            if len(det_xyxys) == 0:
                continue

            trk_xyxys: List[Tuple[float, float, float, float]] = []
            valid_trk_map: List[int] = []
            for local_ti, trk in enumerate(leftover_tracks):
                proj = trk.projected_bbox2d_in_cam(cam_id, projector)
                if proj is None:
                    continue
                trk_xyxys.append(proj)
                valid_trk_map.append(local_ti)
                track_proj_xyxy_by_cam[(local_ti, cam_id)] = proj

            if len(trk_xyxys) == 0:
                continue

            DD, TT = len(det_xyxys), len(trk_xyxys)
            sim2d = np.zeros((DD, TT), dtype=np.float32)
            for di in range(DD):
                for ti in range(TT):
                    sim2d[di, ti] = float(iou2d_xyxy(det_xyxys[di], trk_xyxys[ti]))

            vals2 = sim2d[sim2d > 0]
            best_proj_trk = sim2d.max(axis=0) if sim2d.size else np.array([])

            m2, u2d_det, u2d_trk = perform_association_from_similarity(DD, TT, sim2d)


            m2_filt, _, _ = filter_matches(m2, u2d_det, u2d_trk, sim2d, self.iou2d_thresh)
            m2 = m2_filt

            for di, ti in m2:
                local_track_i = valid_trk_map[int(ti)]
                det_local_i = int(di)
                candidate_matches[local_track_i].append((cam_id, det_local_i))

        chosen_matches: Dict[int, Tuple[int, int]] = {}
        for local_track_i, cand_list in candidate_matches.items():
            if len(cand_list) == 1:
                chosen_matches[local_track_i] = cand_list[0]
                continue

            best = None
            best_area = -1.0
            for cam_id, det_local_i in cand_list:
                proj = track_proj_xyxy_by_cam.get((local_track_i, cam_id), None)
                if proj is None:
                    continue
                a = area_xyxy(proj)
                if a > best_area:
                    best_area = a
                    best = (cam_id, det_local_i)
            if best is not None:
                chosen_matches[local_track_i] = best

        for local_track_i, (cam_id, det_local_i) in chosen_matches.items():
            trk = leftover_tracks[local_track_i]
            det_list = yolo_by_cam[int(cam_id)]
            unmatched_cam = list(getattr(assoc_debug[int(cam_id)], "unmatched_cam", []))

            det_xyxys = []
            det_indices = []
            for idx in unmatched_cam:
                det = det_list.detections[int(idx)]
                xyxy = self._imdet_to_xyxy(det)
                if xyxy is None:
                    continue
                det_xyxys.append(xyxy)
                det_indices.append(int(idx))

            if 0 <= det_local_i < len(det_xyxys):
                trk.update_with_2d(int(cam_id), det_xyxys[det_local_i])

        # ----------------------------------------------------
        # Births: create new tracks from unmatched 3D detections
        # ----------------------------------------------------
        for di in u_det:
            self.tracks.append(Track(dets3d[int(di)], t, motion_model=self.motion_model))

        # lifecycle
        self._prune()
        return self._outputs()


    def _prune(self):
        self.tracks = [trk for trk in self.tracks if trk.time_since_update <= self.max_age]

    def _outputs(self) -> List[TrackOutput]:
        outs: List[TrackOutput] = []
        for trk in self.tracks:
            if trk.hits < self.min_hits:   # and trk.age_total <= self.min_hits:
                continue
            outs.append(
                TrackOutput(
                    track_id=trk.id,
                    state=trk.state(),
                    age=trk.age_total,
                    hits=trk.hits,
                    class_name=getattr(trk, "class_name", "unknown"),
                    time_since_update=trk.time_since_update,
                    last_bbox2d_by_cam=dict(trk.last_bbox2d_by_cam),
                )
            )
        return outs