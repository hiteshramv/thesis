# MEASUREMENT-ONLY association utilities

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable, Set
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

from datastructures.track_object_3d import TrackObject3dList, TrackObject3d
from datastructures.lidar_detections import Object3d
from datastructures.image_detection import ImageDetectionList, ImageDetection
from datastructures.calibration_manager import CalibrationManager
from datastructures.PcdHelper import PcdHelper


@dataclass
class ClassConfirmation:
    track_id: int
    camera_id: int
    det_index: int
    sdiou: float
    confirmed_class: str
    prev_class: str

@dataclass
class MergeResult:
    camera_id: int
    det_index: int
    track_ids_merged: List[int]
    new_track_object: TrackObject3d  # merged LiDAR box (axis-aligned union for robustness)
    sdiou_values: List[float]

@dataclass
class MeasurementAssociationOutput:
    class_confirmations: List[ClassConfirmation]
    merges: List[MergeResult]
    unmatched_tracks_by_cam: Dict[int, List[int]]
    unmatched_detections_by_cam: Dict[int, List[int]]

pcdhelper = PcdHelper()

# ---------------------- Config helpers ----------------------

_DEFAULT_ALLOWED_CLASSES: Set[str] = {
    "car", "van", "truck", "bus", "motorcycle"
}

def _name_ok(name: Optional[str], allowed: Set[str]) -> bool:
    if not name:
        return False
    return name.lower() in allowed

def _within_lr_roi(obj: Object3d, halfwidth_m: float) -> bool:
    """
    Keep objects with |Y| <= halfwidth_m (± left/right band).
    NOTE: Assumes world Y is lateral (left/right). If your lateral axis is X,
    just switch to abs(obj.position.x).
    """
    try:
        y = float(obj.position.y)
        return abs(y) <= float(halfwidth_m)
    except Exception:
        return True  # if unsure, don't drop it silently


# ---------------------- Public API ----------------------

def run_measurement_association_sdiou(
    tracks: TrackObject3dList,
    detections: List[ImageDetectionList],
    image_shapes_by_cam: Dict[int, Tuple[int, int]],
    calib: CalibrationManager,
    *,
    min_sdiou: float = 0.30,
    large_box_px_area: int = 14000,
    class_map: Optional[Dict[str, List[str]]] = None,
    update_class_on_confirm: bool = True,
    perform_merging: bool = True,
    # NEW: filters
    allowed_classes: Optional[Iterable[str]] = None,
    roi_left_right_m: float = 50.0
) -> MeasurementAssociationOutput:
    """
    Measurement-only association with SDIoU, optional class overwrite, and large-object LiDAR merge.
    Now supports:
      - allowed_classes: only consider these classes (LiDAR & 2D).
      - roi_left_right_m: only consider LiDAR objects with |Y| <= ROI (± band).
    """
    if allowed_classes is None:
        allowed_cls_set = set(_DEFAULT_ALLOWED_CLASSES)
    else:
        allowed_cls_set = {c.lower() for c in allowed_classes}

    per_cam_matches, per_cam_unmatched_tracks, per_cam_unmatched_dets = \
        _associate_per_camera_sdiou(
            tracks, detections, image_shapes_by_cam, calib, min_sdiou,
            allowed_cls_set, roi_left_right_m
        )

    confirmations: List[ClassConfirmation] = []
    merges: List[MergeResult] = []

    # ---- Best match per track across cameras ----
    best_for_track: Dict[int, Tuple[int, int, float]] = {}  # track_id -> (cam_id, det_idx, sdiou)
    for cam_id, matches in per_cam_matches.items():
        for (track_id, det_idx, sdiou) in matches:
            prev = best_for_track.get(track_id)
            if prev is None or sdiou > prev[2]:
                best_for_track[track_id] = (cam_id, det_idx, sdiou)

    # ---- Class confirmation (force-update when SDIoU high but classes differ) ----
    dets_by_cam: Dict[int, ImageDetectionList] = {dl.camera_id: dl for dl in detections}
    for track_id, (cam_id, det_idx, sdiou) in best_for_track.items():
        det_list = dets_by_cam.get(cam_id)
        if det_list is None or det_idx >= len(det_list.detections):
            continue

        det = det_list.detections[det_idx]
        cam_cls = (det.class_name or "").lower()
        # filter out camera classes not in allowed set as well
        if not _name_ok(cam_cls, allowed_cls_set):
            continue

        tobj = _get_track_by_id(tracks, track_id)
        if tobj is None:
            continue
        prev_cls = (tobj.object_3d.class_name or "").lower()

        # NEW: if SDIoU passes and classes differ -> force-change to camera class
        force_update = (bool(cam_cls) and sdiou >= min_sdiou and cam_cls != prev_cls)

        # Keep compatibility allowance as a softer condition (still respects allowed set through cam_cls check above)
        compatible = _class_is_compatible(prev_cls, cam_cls, class_map)

        # Final decided class
        new_cls = cam_cls if (force_update or compatible) else prev_cls

        confirmations.append(ClassConfirmation(
            track_id=track_id,
            camera_id=cam_id,
            det_index=det_idx,
            sdiou=sdiou,
            confirmed_class=new_cls,
            prev_class=prev_cls
        ))

        # Apply change if enabled and effective
        if update_class_on_confirm and new_cls != prev_cls and new_cls:
            tobj.object_3d.class_name = new_cls

    # ---- Large-object merging ----
    if perform_merging:
        for cam_id, matches in per_cam_matches.items():
            if not matches:
                continue
            H, W = image_shapes_by_cam[cam_id]
            det_list = dets_by_cam.get(cam_id)
            if det_list is None:
                continue
            buckets: Dict[int, List[Tuple[int, float]]] = {}
            for (track_id, det_idx, sdiou) in matches:
                buckets.setdefault(det_idx, []).append((track_id, sdiou))
            for det_idx, items in buckets.items():
                if len(items) < 2:
                    continue
                bx = det_list.detections[det_idx]
                # Only consider merging when the 2D box is large enough
                if bx.w * bx.h < large_box_px_area:
                    continue
                merge_ids = [tid for (tid, _) in items]
                sdiou_vals = [s for (_, s) in items]
                objs = [_get_track_by_id(tracks, tid).object_3d
                        for tid in merge_ids if _get_track_by_id(tracks, tid) is not None]
                # Also filter objs by allowed class + ROI just to be safe
                objs = [o for o in objs if _name_ok(o.class_name, allowed_cls_set) and _within_lr_roi(o, roi_left_right_m)]
                if len(objs) < 2:
                    continue
                merged_obj = _merge_lidar_objects_axis_aligned(objs)
                keep_id = min(merge_ids)
                merged_track = TrackObject3d(track_id=keep_id, object_3d=merged_obj)
                merges.append(MergeResult(
                    camera_id=cam_id, det_index=det_idx,
                    track_ids_merged=merge_ids,
                    new_track_object=merged_track,
                    sdiou_values=sdiou_vals
                ))
                _apply_inplace_merge(tracks, keep_id=keep_id, merged=merged_track,
                                     remove_ids=[tid for tid in merge_ids if tid != keep_id])

    return MeasurementAssociationOutput(
        class_confirmations=confirmations,
        merges=merges,
        unmatched_tracks_by_cam=per_cam_unmatched_tracks,
        unmatched_detections_by_cam=per_cam_unmatched_dets
    )


def _associate_per_camera_sdiou(
    tracks: TrackObject3dList,
    detections: List[ImageDetectionList],
    image_shapes_by_cam: Dict[int, Tuple[int, int]],
    calib: CalibrationManager,
    min_sdiou: float,
    allowed_classes: Set[str],
    roi_left_right_m: float
) -> Tuple[Dict[int, List[Tuple[int,int,float]]], Dict[int, List[int]], Dict[int, List[int]]]:
    per_cam_matches: Dict[int, List[Tuple[int,int,float]]] = {}
    unmatched_tracks_by_cam: Dict[int, List[int]] = {}
    unmatched_dets_by_cam: Dict[int, List[int]] = {}

    for det_list in detections:
        cam_id = det_list.camera_id
        H, W = image_shapes_by_cam[cam_id]

        # Only use tracks that pass class + ROI filters
        track_ids, lidar_xyxy = _project_tracks_to_xyxy(
            tracks, cam_id, calib, (H, W),
            allowed_classes=allowed_classes,
            roi_left_right_m=roi_left_right_m
        )

        # Also filter 2D dets by allowed class
        cam_xyxy = []
        for d in det_list.detections:
            if _name_ok(d.class_name, allowed_classes):
                cam_xyxy.append(_det_xyxy(d))

        if len(track_ids) == 0 or len(cam_xyxy) == 0:
            per_cam_matches[cam_id] = []
            unmatched_tracks_by_cam[cam_id] = track_ids[:]  # empty or filtered
            unmatched_dets_by_cam[cam_id] = list(range(len(cam_xyxy)))
            continue

        N, M = len(lidar_xyxy), len(cam_xyxy)
        sdiou_mat = np.zeros((N, M), dtype=np.float32)
        for i in range(N):
            for j in range(M):
                sdiou_mat[i, j] = _sdiou_xyxy(lidar_xyxy[i], cam_xyxy[j])

        rows, cols = linear_sum_assignment(-sdiou_mat)

        matches, used_rows, used_cols = [], set(), set()
        for r, c in zip(rows, cols):
            s = float(sdiou_mat[r, c])
            if s >= min_sdiou:
                matches.append((track_ids[r], c, s))
                used_rows.add(r); used_cols.add(c)

        per_cam_matches[cam_id] = matches
        unmatched_tracks_by_cam[cam_id] = [track_ids[i] for i in range(N) if i not in used_rows]
        unmatched_dets_by_cam[cam_id] = [j for j in range(M) if j not in used_cols]

    return per_cam_matches, unmatched_tracks_by_cam, unmatched_dets_by_cam


# ==================== Geometry & Helpers ====================

def _project_tracks_to_xyxy(
    tracks: TrackObject3dList,
    cam_id: int,
    calib: CalibrationManager,
    image_shape: tuple,
    *,
    allowed_classes: Set[str],
    roi_left_right_m: float
):
    # Expect exactly your formats:
    K = calib.get_intrinsic(cam_id)      # 3x3
    E = calib.get_extrinsic(cam_id)      # 3x4 in your schema

    # if someone returns 4x4 by mistake, coerce to 3x4
    if E.shape == (4, 4):
        E = np.hstack([E[:3, :3], E[:3, 3:4]])  # 3x4

    track_ids, boxes = [], []
    for tobj in tracks.track_objects_3d:
        o = tobj.object_3d
        # Filter by class
        if not _name_ok(o.class_name, allowed_classes):
            continue
        # Filter by left/right ROI band
        if not _within_lr_roi(o, roi_left_right_m):
            continue

        box = project_single_3d_box_xyxy_using_helper(o, K, E, image_shape)
        if box is None:
            continue
        track_ids.append(int(tobj.track_id))
        boxes.append(box)
    return track_ids, boxes


def project_single_3d_box_xyxy_using_helper(
    obj: Object3d,             # Object3d
    K: np.ndarray,             # 3x3
    E: np.ndarray,             # 3x4
    image_shape: tuple         # (rows, cols) or (rows, cols, channels)
):
    # sizes
    l, w, h = float(obj.size.x), float(obj.size.y), float(obj.size.z)
    # NOTE: your dataset appears to store cz as TOP; subtract 2.9 and use centered Z.
    cx, cy, cz = float(obj.position.x), float(obj.position.y), float(obj.position.z) - 2.9
    yaw = float(obj.yaw_angle)

    # 8 local corners (centered box, z: bottom = -h/2, top = +h/2)
    xs = np.array([+l/2, +l/2, -l/2, -l/2, +l/2, +l/2, -l/2, -l/2], dtype=np.float64)
    ys = np.array([+w/2, -w/2, -w/2, +w/2, +w/2, -w/2, -w/2, +w/2], dtype=np.float64)
    zs = np.array([-h/2, -h/2, -h/2, -h/2, +h/2, +h/2, +h/2, +h/2], dtype=np.float64)

    c, s = math.cos(yaw), math.sin(yaw)
    X = xs * c - ys * s + cx
    Y = xs * s + ys * c + cy
    Z = zs + cz
    corners = np.vstack([X, Y, Z]).T  # (8,3)

    # Ensure image_shape is (rows, cols, channels) for PcdHelper (it only uses rows/cols anyway)
    if len(image_shape) == 2:
        H, W = image_shape
        img_shape3 = (H, W, 3)
    else:
        img_shape3 = image_shape

    uvw = pcdhelper.project_points_uv(corners, img_shape3, K, E)
    if uvw is None or uvw.size == 0:
        return None

    u1, v1 = float(np.min(uvw["u"])), float(np.min(uvw["v"]))
    u2, v2 = float(np.max(uvw["u"])), float(np.max(uvw["v"]))

    # sanity: ensure at least 1px in size
    if (u2 - u1) < 1.0 or (v2 - v1) < 1.0:
        return None
    return (u1, v1, u2, v2)


def _det_xyxy(d: ImageDetection) -> Tuple[float,float,float,float]:
    x1 = d.x - d.w / 2.0
    y1 = d.y - d.h / 2.0
    x2 = d.x + d.w / 2.0
    y2 = d.y + d.h / 2.0
    return (x1, y1, x2, y2)


# -------- SDIoU (2D) --------

def _sdiou_xyxy(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float], lam: float = 0.5) -> float:
    """ SDIoU = IoU - DIoU_penalty - λ * scale_mismatch ; clamped to [0,1] """
    iou = _iou_xyxy(a, b)
    diou_pen = _diou_penalty(a, b)
    scale_pen = _scale_penalty(a, b)
    sdiou = iou - diou_pen - lam * scale_pen
    return max(0.0, min(1.0, sdiou))

def _iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    ua = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    ub = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = ua + ub - inter + 1e-9
    return float(inter / union)

def _diou_penalty(a, b) -> float:
    acx = 0.5 * (a[0] + a[2]); acy = 0.5 * (a[1] + a[3])
    bcx = 0.5 * (b[0] + b[2]); bcy = 0.5 * (b[1] + b[3])
    ex1, ey1 = min(a[0], b[0]), min(a[1], b[1])
    ex2, ey2 = max(a[2], b[2]), max(a[3], b[3])
    c = math.hypot(ex2 - ex1, ey2 - ey1) + 1e-9
    rho2 = (acx - bcx)**2 + (acy - bcy)**2
    return float(rho2 / (c * c))

def _scale_penalty(a, b) -> float:
    aw = max(1e-6, a[2] - a[0]); ah = max(1e-6, a[3] - a[1])
    bw = max(1e-6, b[2] - b[0]); bh = max(1e-6, b[3] - b[1])
    return ((abs(aw - bw) / max(aw, bw)) ** 2 + (abs(ah - bh) / max(ah, bh)) ** 2)


# -------- Class + Merge helpers --------

def _class_is_compatible(prev_cls: str, cam_cls: str, class_map: Optional[Dict[str, Iterable[str]]]) -> bool:
    if not cam_cls:
        return False
    if class_map is None or prev_cls == "" or prev_cls == cam_cls:
        return True
    allowed = set(class_map.get(prev_cls, []))
    return cam_cls in allowed

def _get_track_by_id(tracks: TrackObject3dList, tid: int) -> Optional[TrackObject3d]:
    for t in tracks.track_objects_3d:
        if int(t.track_id) == int(tid):
            return t
    return None

def _apply_inplace_merge(tracks: TrackObject3dList, keep_id: int, merged: TrackObject3d, remove_ids: List[int]) -> None:
    # replace keep_id object, drop others
    new_list: List[TrackObject3d] = []
    replaced = False
    for t in tracks.track_objects_3d:
        if int(t.track_id) in remove_ids:
            continue
        if int(t.track_id) == int(keep_id):
            new_list.append(merged)
            replaced = True
        else:
            new_list.append(t)
    if not replaced:
        new_list.append(merged)
    tracks.track_objects_3d = new_list

def _merge_lidar_objects_axis_aligned(objs: List[Object3d]) -> Object3d:
    """
    Conservative merge: world-axis-aligned union of 3D boxes (ignores yaw).
    This is robust when multiple smaller LiDAR boxes belong to one large object.
    """
    mins = np.array([+np.inf, +np.inf, +np.inf], dtype=np.float64)
    maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    for o in objs:
        l, w, h = float(o.size.x), float(o.size.y), float(o.size.z)
        cx, cy, cz = float(o.position.x), float(o.position.y), float(o.position.z)
        yaw = float(o.yaw_angle)

        xs = np.array([+l/2, +l/2, -l/2, -l/2, +l/2, +l/2, -l/2, -l/2], dtype=np.float64)
        ys = np.array([+w/2, -w/2, -w/2, +w/2, +w/2, -w/2, -w/2, +w/2], dtype=np.float64)
        zs = np.array([-h/2, -h/2, -h/2, -h/2, +h/2, +h/2, +h/2, +h/2], dtype=np.float64)

        c, s = math.cos(yaw), math.sin(yaw)
        X = xs * c - ys * s + cx
        Y = xs * s + ys * c + cy
        Z = zs + cz

        mins = np.minimum(mins, np.array([X.min(), Y.min(), Z.min()]))
        maxs = np.maximum(maxs, np.array([X.max(), Y.max(), Z.max()]))

    size = maxs - mins
    center = 0.5 * (mins + maxs)

    merged = Object3d(
        position=type(objs[0].position)(center[0], center[1], center[2]),
        size=type(objs[0].size)(size[0], size[1], size[2]),
        orientation=objs[0].orientation,
        speed=objs[0].speed,
        yaw_angle=0.0,                     # conservative (axis-aligned)
        class_name=objs[0].class_name,     # may be overwritten earlier by confirmation
        category_confidence=objs[0].category_confidence,
        existence_confidence=objs[0].existence_confidence
    )
    return merged



"""
# MEASUREMENT-ONLY association utilities

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

from datastructures.track_object_3d import TrackObject3dList, TrackObject3d
from datastructures.lidar_detections import Object3d
from datastructures.image_detection import ImageDetectionList, ImageDetection
from datastructures.calibration_manager import CalibrationManager
from datastructures.PcdHelper import PcdHelper


@dataclass
class ClassConfirmation:
    track_id: int
    camera_id: int
    det_index: int
    sdiou: float
    confirmed_class: str
    prev_class: str

@dataclass
class MergeResult:
    camera_id: int
    det_index: int
    track_ids_merged: List[int]
    new_track_object: TrackObject3d  # merged LiDAR box (axis-aligned union for robustness)
    sdiou_values: List[float]

@dataclass
class MeasurementAssociationOutput:
    class_confirmations: List[ClassConfirmation]
    merges: List[MergeResult]
    unmatched_tracks_by_cam: Dict[int, List[int]]
    unmatched_detections_by_cam: Dict[int, List[int]]

pcdhelper = PcdHelper()

def run_measurement_association_sdiou(
    tracks: TrackObject3dList,
    detections: List[ImageDetectionList],
    image_shapes_by_cam: Dict[int, Tuple[int, int]],
    calib: CalibrationManager,
    *,
    min_sdiou: float = 0.30,
    large_box_px_area: int = 14000,
    class_map: Optional[Dict[str, List[str]]] = None,
    update_class_on_confirm: bool = True,
    perform_merging: bool = True
) -> MeasurementAssociationOutput:
    per_cam_matches, per_cam_unmatched_tracks, per_cam_unmatched_dets = \
        _associate_per_camera_sdiou(tracks, detections, image_shapes_by_cam, calib, min_sdiou)

    confirmations: List[ClassConfirmation] = []
    merges: List[MergeResult] = []

    # ---- Best match per track across cameras ----
    best_for_track: Dict[int, Tuple[int, int, float]] = {}  # track_id -> (cam_id, det_idx, sdiou)
    for cam_id, matches in per_cam_matches.items():
        for (track_id, det_idx, sdiou) in matches:
            prev = best_for_track.get(track_id)
            if prev is None or sdiou > prev[2]:
                best_for_track[track_id] = (cam_id, det_idx, sdiou)

    # ---- Class confirmation (force-update when SDIoU high but classes differ) ----
    dets_by_cam: Dict[int, ImageDetectionList] = {dl.camera_id: dl for dl in detections}
    for track_id, (cam_id, det_idx, sdiou) in best_for_track.items():
        det_list = dets_by_cam.get(cam_id)
        if det_list is None or det_idx >= len(det_list.detections):
            continue

        det = det_list.detections[det_idx]
        cam_cls = (det.class_name or "").lower()

        tobj = _get_track_by_id(tracks, track_id)
        if tobj is None:
            continue
        prev_cls = (tobj.object_3d.class_name or "").lower()

        # NEW: if SDIoU passes and classes differ -> force-change to camera class
        force_update = (bool(cam_cls) and sdiou >= min_sdiou and cam_cls != prev_cls)

        # Keep compatibility allowance as a softer condition
        compatible = _class_is_compatible(prev_cls, cam_cls, class_map)

        # Final decided class
        new_cls = cam_cls if (force_update or compatible) else prev_cls

        confirmations.append(ClassConfirmation(
            track_id=track_id,
            camera_id=cam_id,
            det_index=det_idx,
            sdiou=sdiou,
            confirmed_class=new_cls,
            prev_class=prev_cls
        ))

        # Apply change if enabled and effective
        if update_class_on_confirm and new_cls != prev_cls and new_cls:
            tobj.object_3d.class_name = new_cls

    # ---- Large-object merging ----
    if perform_merging:
        for cam_id, matches in per_cam_matches.items():
            if not matches:
                continue
            H, W = image_shapes_by_cam[cam_id]
            det_list = dets_by_cam.get(cam_id)
            if det_list is None:
                continue
            buckets: Dict[int, List[Tuple[int, float]]] = {}
            for (track_id, det_idx, sdiou) in matches:
                buckets.setdefault(det_idx, []).append((track_id, sdiou))
            for det_idx, items in buckets.items():
                if len(items) < 2:
                    continue
                bx = det_list.detections[det_idx]
                if bx.w * bx.h < large_box_px_area:
                    continue
                merge_ids = [tid for (tid, _) in items]
                sdiou_vals = [s for (_, s) in items]
                objs = [_get_track_by_id(tracks, tid).object_3d
                        for tid in merge_ids if _get_track_by_id(tracks, tid) is not None]
                if len(objs) < 2:
                    continue
                merged_obj = _merge_lidar_objects_axis_aligned(objs)
                keep_id = min(merge_ids)
                merged_track = TrackObject3d(track_id=keep_id, object_3d=merged_obj)
                merges.append(MergeResult(
                    camera_id=cam_id, det_index=det_idx,
                    track_ids_merged=merge_ids,
                    new_track_object=merged_track,
                    sdiou_values=sdiou_vals
                ))
                _apply_inplace_merge(tracks, keep_id=keep_id, merged=merged_track,
                                     remove_ids=[tid for tid in merge_ids if tid != keep_id])

    return MeasurementAssociationOutput(
        class_confirmations=confirmations,
        merges=merges,
        unmatched_tracks_by_cam=per_cam_unmatched_tracks,
        unmatched_detections_by_cam=per_cam_unmatched_dets
    )


def _associate_per_camera_sdiou(
    tracks: TrackObject3dList,
    detections: List[ImageDetectionList],
    image_shapes_by_cam: Dict[int, Tuple[int, int]],
    calib: CalibrationManager,
    min_sdiou: float
) -> Tuple[Dict[int, List[Tuple[int,int,float]]], Dict[int, List[int]], Dict[int, List[int]]]:
    per_cam_matches: Dict[int, List[Tuple[int,int,float]]] = {}
    unmatched_tracks_by_cam: Dict[int, List[int]] = {}
    unmatched_dets_by_cam: Dict[int, List[int]] = {}

    for det_list in detections:
        cam_id = det_list.camera_id
        H, W = image_shapes_by_cam[cam_id]
        track_ids, lidar_xyxy = _project_tracks_to_xyxy(tracks, cam_id, calib, (H, W))
        cam_xyxy = [_det_xyxy(d) for d in det_list.detections]

        if len(track_ids) == 0 or len(cam_xyxy) == 0:
            per_cam_matches[cam_id] = []
            unmatched_tracks_by_cam[cam_id] = track_ids[:]
            unmatched_dets_by_cam[cam_id] = list(range(len(cam_xyxy)))
            continue

        N, M = len(lidar_xyxy), len(cam_xyxy)
        sdiou_mat = np.zeros((N, M), dtype=np.float32)
        for i in range(N):
            for j in range(M):
                sdiou_mat[i, j] = _sdiou_xyxy(lidar_xyxy[i], cam_xyxy[j])

        rows, cols = linear_sum_assignment(-sdiou_mat)

        matches, used_rows, used_cols = [], set(), set()
        for r, c in zip(rows, cols):
            s = float(sdiou_mat[r, c])
            if s >= min_sdiou:
                matches.append((track_ids[r], c, s))
                used_rows.add(r); used_cols.add(c)

        per_cam_matches[cam_id] = matches
        unmatched_tracks_by_cam[cam_id] = [track_ids[i] for i in range(N) if i not in used_rows]
        unmatched_dets_by_cam[cam_id] = [j for j in range(M) if j not in used_cols]

    return per_cam_matches, unmatched_tracks_by_cam, unmatched_dets_by_cam


# ==================== Geometry & Helpers ====================

def _project_tracks_to_xyxy(
    tracks,
    cam_id: int,
    calib,
    image_shape: tuple,
):
    # Expect exactly your formats:
    K = calib.get_intrinsic(cam_id)      # 3x3
    E = calib.get_extrinsic(cam_id)      # 3x4 in your schema

    # if someone returns 4x4 by mistake, coerce to 3x4
    if E.shape == (4, 4):
        E = np.hstack([E[:3, :3], E[:3, 3:4]])  # 3x4

    track_ids, boxes = [], []
    for tobj in tracks.track_objects_3d:
        box = project_single_3d_box_xyxy_using_helper(tobj.object_3d, K, E, image_shape)
        if box is None:
            continue
        track_ids.append(int(tobj.track_id))
        boxes.append(box)
    return track_ids, boxes


def project_single_3d_box_xyxy_using_helper(
    obj,                       # Object3d
    K: np.ndarray,             # 3x3
    E: np.ndarray,             # 3x4
    image_shape: tuple         # (rows, cols, channels)
):
    # sizes
    l, w, h = float(obj.size.x), float(obj.size.y), float(obj.size.z)
    # NOTE: your dataset appears to store cz as TOP; subtract 2.9 and use centered Z.
    cx, cy, cz = float(obj.position.x), float(obj.position.y), float(obj.position.z) - 2.9
    yaw = float(obj.yaw_angle)

    # 8 local corners (centered box, z: bottom = -h/2, top = +h/2)
    xs = np.array([+l/2, +l/2, -l/2, -l/2, +l/2, +l/2, -l/2, -l/2], dtype=np.float64)
    ys = np.array([+w/2, -w/2, -w/2, +w/2, +w/2, -w/2, -w/2, +w/2], dtype=np.float64)
    zs = np.array([-h/2, -h/2, -h/2, -h/2, +h/2, +h/2, +h/2, +h/2], dtype=np.float64)

    c, s = math.cos(yaw), math.sin(yaw)
    X = xs * c - ys * s + cx
    Y = xs * s + ys * c + cy
    Z = zs + cz
    corners = np.vstack([X, Y, Z]).T  # (8,3)

    uvw = pcdhelper.project_points_uv(corners, image_shape, K, E)
    if uvw is None or uvw.size == 0:
        return None

    u1, v1 = float(np.min(uvw["u"])), float(np.min(uvw["v"]))
    u2, v2 = float(np.max(uvw["u"])), float(np.max(uvw["v"]))

    # sanity: ensure at least 1px in size
    if (u2 - u1) < 1.0 or (v2 - v1) < 1.0:
        return None
    return (u1, v1, u2, v2)


def _det_xyxy(d: ImageDetection) -> Tuple[float,float,float,float]:
    x1 = d.x - d.w / 2.0
    y1 = d.y - d.h / 2.0
    x2 = d.x + d.w / 2.0
    y2 = d.y + d.h / 2.0
    return (x1, y1, x2, y2)


# -------- SDIoU (2D) --------

def _sdiou_xyxy(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float], lam: float = 0.5) -> float:
    # SDIoU = IoU - DIoU_penalty - λ * scale_mismatch ; clamped to [0,1] 
    iou = _iou_xyxy(a, b)
    diou_pen = _diou_penalty(a, b)
    scale_pen = _scale_penalty(a, b)
    sdiou = iou - diou_pen - lam * scale_pen
    return max(0.0, min(1.0, sdiou))

def _iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    ua = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    ub = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = ua + ub - inter + 1e-9
    return float(inter / union)

def _diou_penalty(a, b) -> float:
    acx = 0.5 * (a[0] + a[2]); acy = 0.5 * (a[1] + a[3])
    bcx = 0.5 * (b[0] + b[2]); bcy = 0.5 * (b[1] + b[3])
    ex1, ey1 = min(a[0], b[0]), min(a[1], b[1])
    ex2, ey2 = max(a[2], b[2]), max(a[3], b[3])
    c = math.hypot(ex2 - ex1, ey2 - ey1) + 1e-9
    rho2 = (acx - bcx)**2 + (acy - bcy)**2
    return float(rho2 / (c * c))

def _scale_penalty(a, b) -> float:
    aw = max(1e-6, a[2] - a[0]); ah = max(1e-6, a[3] - a[1])
    bw = max(1e-6, b[2] - b[0]); bh = max(1e-6, b[3] - b[1])
    return ((abs(aw - bw) / max(aw, bw)) ** 2 + (abs(ah - bh) / max(ah, bh)) ** 2)


# -------- Class + Merge helpers --------

def _class_is_compatible(prev_cls: str, cam_cls: str, class_map: Optional[Dict[str, Iterable[str]]]) -> bool:
    if not cam_cls:
        return False
    if class_map is None or prev_cls == "" or prev_cls == cam_cls:
        return True
    allowed = set(class_map.get(prev_cls, []))
    return cam_cls in allowed

def _get_track_by_id(tracks: TrackObject3dList, tid: int) -> Optional[TrackObject3d]:
    for t in tracks.track_objects_3d:
        if int(t.track_id) == int(tid):
            return t
    return None

def _apply_inplace_merge(tracks: TrackObject3dList, keep_id: int, merged: TrackObject3d, remove_ids: List[int]) -> None:
    # replace keep_id object, drop others
    new_list: List[TrackObject3d] = []
    replaced = False
    for t in tracks.track_objects_3d:
        if int(t.track_id) in remove_ids:
            continue
        if int(t.track_id) == int(keep_id):
            new_list.append(merged)
            replaced = True
        else:
            new_list.append(t)
    if not replaced:
        new_list.append(merged)
    tracks.track_objects_3d = new_list

def _merge_lidar_objects_axis_aligned(objs: List[Object3d]) -> Object3d:
    
    #Conservative merge: world-axis-aligned union of 3D boxes (ignores yaw).
    #This is robust when multiple smaller LiDAR boxes belong to one large object.
    
    mins = np.array([+np.inf, +np.inf, +np.inf], dtype=np.float64)
    maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    for o in objs:
        l, w, h = float(o.size.x), float(o.size.y), float(o.size.z)
        cx, cy, cz = float(o.position.x), float(o.position.y), float(o.position.z)
        yaw = float(o.yaw_angle)

        xs = np.array([+l/2, +l/2, -l/2, -l/2, +l/2, +l/2, -l/2, -l/2], dtype=np.float64)
        ys = np.array([+w/2, -w/2, -w/2, +w/2, +w/2, -w/2, -w/2, +w/2], dtype=np.float64)
        zs = np.array([-h/2, -h/2, -h/2, -h/2, +h/2, +h/2, +h/2, +h/2], dtype=np.float64)

        c, s = math.cos(yaw), math.sin(yaw)
        X = xs * c - ys * s + cx
        Y = xs * s + ys * c + cy
        Z = zs + cz

        mins = np.minimum(mins, np.array([X.min(), Y.min(), Z.min()]))
        maxs = np.maximum(maxs, np.array([X.max(), Y.max(), Z.max()]))

    size = maxs - mins
    center = 0.5 * (mins + maxs)

    merged = Object3d(
        position=type(objs[0].position)(center[0], center[1], center[2]),
        size=type(objs[0].size)(size[0], size[1], size[2]),
        orientation=objs[0].orientation,
        speed=objs[0].speed,
        yaw_angle=0.0,                     # conservative (axis-aligned)
        class_name=objs[0].class_name,     # may be overwritten earlier by confirmation
        category_confidence=objs[0].category_confidence,
        existence_confidence=objs[0].existence_confidence
    )
    return merged

"""

"""
# MEASUREMENT-ONLY association utilities

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

from datastructures.track_object_3d import TrackObject3dList, TrackObject3d
from datastructures.lidar_detections import Object3d
from datastructures.image_detection import ImageDetectionList, ImageDetection
from datastructures.calibration_manager import CalibrationManager
from datastructures.PcdHelper import PcdHelper


@dataclass
class ClassConfirmation:
    track_id: int
    camera_id: int
    det_index: int
    sdiou: float
    confirmed_class: str
    prev_class: str

@dataclass
class MergeResult:
    camera_id: int
    det_index: int
    track_ids_merged: List[int]
    new_track_object: TrackObject3d  # merged LiDAR box (axis-aligned union for robustness)
    sdiou_values: List[float]

@dataclass
class MeasurementAssociationOutput:
    class_confirmations: List[ClassConfirmation]
    merges: List[MergeResult]
    unmatched_tracks_by_cam: Dict[int, List[int]]
    unmatched_detections_by_cam: Dict[int, List[int]]

pcdhelper = PcdHelper()

def run_measurement_association_sdiou(
    tracks: TrackObject3dList,
    detections: List[ImageDetectionList],
    image_shapes_by_cam: Dict[int, Tuple[int, int]],
    calib: CalibrationManager,
    *,
    min_sdiou: float = 0.30,
    large_box_px_area: int = 1400,
    class_map: Optional[Dict[str, List[str]]] = None,
    update_class_on_confirm: bool = True,
    perform_merging: bool = True
) -> MeasurementAssociationOutput:
    per_cam_matches, per_cam_unmatched_tracks, per_cam_unmatched_dets = \
        _associate_per_camera_sdiou(tracks, detections, image_shapes_by_cam, calib, min_sdiou)

    confirmations: List[ClassConfirmation] = []
    merges: List[MergeResult] = []

    # ---- Best match per track across cameras ----
    best_for_track: Dict[int, Tuple[int, int, float]] = {}  # track_id -> (cam_id, det_idx, sdiou)
    for cam_id, matches in per_cam_matches.items():
        for (track_id, det_idx, sdiou) in matches:
            prev = best_for_track.get(track_id)
            if prev is None or sdiou > prev[2]:
                best_for_track[track_id] = (cam_id, det_idx, sdiou)

    # ---- Class confirmation ----
    # Build quick lookup: cam_id -> detections list
    dets_by_cam: Dict[int, ImageDetectionList] = {dl.camera_id: dl for dl in detections}
    for track_id, (cam_id, det_idx, sdiou) in best_for_track.items():
        det_list = dets_by_cam.get(cam_id)
        if det_list is None or det_idx >= len(det_list.detections):
            continue
        det = det_list.detections[det_idx]
        cam_cls = (det.class_name or "").lower()
        tobj = _get_track_by_id(tracks, track_id)
        if tobj is None:
            continue
        prev_cls = (tobj.object_3d.class_name or "").lower()
        accept = _class_is_compatible(prev_cls, cam_cls, class_map)
        confirmations.append(ClassConfirmation(
            track_id=track_id, camera_id=cam_id, det_index=det_idx,
            sdiou=sdiou, confirmed_class=cam_cls if accept else prev_cls, prev_class=prev_cls
        ))
        if update_class_on_confirm and accept and cam_cls:
            tobj.object_3d.class_name = cam_cls

    # ---- Large-object merging ----
    if perform_merging:
        for cam_id, matches in per_cam_matches.items():
            if not matches:
                continue
            H, W = image_shapes_by_cam[cam_id]
            det_list = dets_by_cam.get(cam_id)
            if det_list is None:
                continue
            buckets: Dict[int, List[Tuple[int, float]]] = {}
            for (track_id, det_idx, sdiou) in matches:
                buckets.setdefault(det_idx, []).append((track_id, sdiou))
            for det_idx, items in buckets.items():
                if len(items) < 2:
                    continue
                bx = det_list.detections[det_idx]
                if bx.w * bx.h < large_box_px_area:
                    continue
                merge_ids = [tid for (tid, _) in items]
                sdiou_vals = [s for (_, s) in items]
                objs = [_get_track_by_id(tracks, tid).object_3d
                        for tid in merge_ids if _get_track_by_id(tracks, tid) is not None]
                if len(objs) < 2:
                    continue
                merged_obj = _merge_lidar_objects_axis_aligned(objs)
                keep_id = min(merge_ids)
                merged_track = TrackObject3d(track_id=keep_id, object_3d=merged_obj)
                merges.append(MergeResult(
                    camera_id=cam_id, det_index=det_idx,
                    track_ids_merged=merge_ids,
                    new_track_object=merged_track,
                    sdiou_values=sdiou_vals
                ))
                _apply_inplace_merge(tracks, keep_id=keep_id, merged=merged_track,
                                     remove_ids=[tid for tid in merge_ids if tid != keep_id])

    return MeasurementAssociationOutput(
        class_confirmations=confirmations,
        merges=merges,
        unmatched_tracks_by_cam=per_cam_unmatched_tracks,
        unmatched_detections_by_cam=per_cam_unmatched_dets
    )


def _associate_per_camera_sdiou(
    tracks: TrackObject3dList,
    detections: List[ImageDetectionList],
    image_shapes_by_cam: Dict[int, Tuple[int, int]],
    calib: CalibrationManager,
    min_sdiou: float
) -> Tuple[Dict[int, List[Tuple[int,int,float]]], Dict[int, List[int]], Dict[int, List[int]]]:
    per_cam_matches: Dict[int, List[Tuple[int,int,float]]] = {}
    unmatched_tracks_by_cam: Dict[int, List[int]] = {}
    unmatched_dets_by_cam: Dict[int, List[int]] = {}

    for det_list in detections:
        cam_id = det_list.camera_id
        H, W = image_shapes_by_cam[cam_id]
        track_ids, lidar_xyxy = _project_tracks_to_xyxy(tracks, cam_id, calib, (H, W))
        cam_xyxy = [_det_xyxy(d) for d in det_list.detections]

        if len(track_ids) == 0 or len(cam_xyxy) == 0:
            per_cam_matches[cam_id] = []
            unmatched_tracks_by_cam[cam_id] = track_ids[:]
            unmatched_dets_by_cam[cam_id] = list(range(len(cam_xyxy)))
            continue

        N, M = len(lidar_xyxy), len(cam_xyxy)
        sdiou_mat = np.zeros((N, M), dtype=np.float32)
        for i in range(N):
            for j in range(M):
                sdiou_mat[i, j] = _sdiou_xyxy(lidar_xyxy[i], cam_xyxy[j])

        rows, cols = linear_sum_assignment(-sdiou_mat)

        matches, used_rows, used_cols = [], set(), set()
        for r, c in zip(rows, cols):
            s = float(sdiou_mat[r, c])
            if s >= min_sdiou:
                matches.append((track_ids[r], c, s))
                used_rows.add(r); used_cols.add(c)

        per_cam_matches[cam_id] = matches
        unmatched_tracks_by_cam[cam_id] = [track_ids[i] for i in range(N) if i not in used_rows]
        unmatched_dets_by_cam[cam_id] = [j for j in range(M) if j not in used_cols]

    return per_cam_matches, unmatched_tracks_by_cam, unmatched_dets_by_cam


# ==================== Geometry & Helpers ====================

def _project_tracks_to_xyxy(
    tracks,
    cam_id: int,
    calib,
    image_shape: tuple,
):
    # Expect exactly your formats:
    K = calib.get_intrinsic(cam_id)      # 3x3
    E = calib.get_extrinsic(cam_id)      # 3x4 in your schema

    # if someone returns 4x4 by mistake, coerce to 3x4
    if E.shape == (4, 4):
        E = np.hstack([E[:3, :3], E[:3, 3:4]])  # 3x4

    track_ids, boxes = [], []
    for tobj in tracks.track_objects_3d:
        box = project_single_3d_box_xyxy_using_helper(tobj.object_3d, K, E, image_shape)
        if box is None:
            continue
        track_ids.append(int(tobj.track_id))
        boxes.append(box)
    return track_ids, boxes


def project_single_3d_box_xyxy_using_helper(
    obj,                       # Object3d
    K: np.ndarray,             # 3x3
    E: np.ndarray,             # 3x4
    image_shape: tuple         # (rows, cols, channels)
):
    # sizes
    l, w, h = float(obj.size.x), float(obj.size.y), float(obj.size.z)
    cx, cy, cz = float(obj.position.x), float(obj.position.y), float(obj.position.z)-2.9
    yaw = float(obj.yaw_angle)

    # 8 local corners (centered box, z: bottom = -h/2, top = +h/2)
    xs = np.array([+l/2, +l/2, -l/2, -l/2, +l/2, +l/2, -l/2, -l/2], dtype=np.float64)
    ys = np.array([+w/2, -w/2, -w/2, +w/2, +w/2, -w/2, -w/2, +w/2], dtype=np.float64)
    zs = np.array([-h/2, -h/2, -h/2, -h/2, +h/2, +h/2, +h/2, +h/2], dtype=np.float64)

    c, s = math.cos(yaw), math.sin(yaw)
    X = xs * c - ys * s + cx
    Y = xs * s + ys * c + cy
    Z = zs + cz
    corners = np.vstack([X, Y, Z]).T  # (8,3)

    uvw = pcdhelper.project_points_uv(corners, image_shape, K, E)
    if uvw is None or uvw.size == 0:
        return None

    u1, v1 = float(np.min(uvw["u"])), float(np.min(uvw["v"]))
    u2, v2 = float(np.max(uvw["u"])), float(np.max(uvw["v"]))

    # sanity: ensure at least 1px in size
    if (u2 - u1) < 1.0 or (v2 - v1) < 1.0:
        return None
    return (u1, v1, u2, v2)

# Corrected function for vehicle_pipeline.py
def _det_xyxy(d: ImageDetection) -> Tuple[float,float,float,float]:
    x1 = d.x - d.w / 2.0
    y1 = d.y - d.h / 2.0
    x2 = d.x + d.w / 2.0
    y2 = d.y + d.h / 2.0
    return (x1, y1, x2, y2)

# -------- SDIoU (2D) --------

def _sdiou_xyxy(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float], lam: float = 0.5) -> float:
    # SDIoU = IoU - DIoU_penalty - λ * scale_mismatch ; clamped to [0,1]
    iou = _iou_xyxy(a, b)
    diou_pen = _diou_penalty(a, b)
    scale_pen = _scale_penalty(a, b)
    sdiou = iou - diou_pen - lam * scale_pen
    return max(0.0, min(1.0, sdiou))

def _iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    ua = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    ub = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = ua + ub - inter + 1e-9
    return float(inter / union)

def _diou_penalty(a, b) -> float:
    acx = 0.5 * (a[0] + a[2]); acy = 0.5 * (a[1] + a[3])
    bcx = 0.5 * (b[0] + b[2]); bcy = 0.5 * (b[1] + b[3])
    ex1, ey1 = min(a[0], b[0]), min(a[1], b[1])
    ex2, ey2 = max(a[2], b[2]), max(a[3], b[3])
    c = math.hypot(ex2 - ex1, ey2 - ey1) + 1e-9
    rho2 = (acx - bcx)**2 + (acy - bcy)**2
    return float(rho2 / (c * c))

def _scale_penalty(a, b) -> float:
    aw = max(1e-6, a[2] - a[0]); ah = max(1e-6, a[3] - a[1])
    bw = max(1e-6, b[2] - b[0]); bh = max(1e-6, b[3] - b[1])
    return ((abs(aw - bw) / max(aw, bw)) ** 2 + (abs(ah - bh) / max(ah, bh)) ** 2)

# -------- Class + Merge helpers --------

def _class_is_compatible(prev_cls: str, cam_cls: str, class_map: Optional[Dict[str, Iterable[str]]]) -> bool:
    if not cam_cls:
        return False
    if class_map is None or prev_cls == "" or prev_cls == cam_cls:
        return True
    allowed = set(class_map.get(prev_cls, []))
    return cam_cls in allowed

def _get_track_by_id(tracks: TrackObject3dList, tid: int) -> Optional[TrackObject3d]:
    for t in tracks.track_objects_3d:
        if int(t.track_id) == int(tid):
            return t
    return None

def _apply_inplace_merge(tracks: TrackObject3dList, keep_id: int, merged: TrackObject3d, remove_ids: List[int]) -> None:
    # replace keep_id object, drop others
    new_list: List[TrackObject3d] = []
    replaced = False
    for t in tracks.track_objects_3d:
        if int(t.track_id) in remove_ids:
            continue
        if int(t.track_id) == int(keep_id):
            new_list.append(merged)
            replaced = True
        else:
            new_list.append(t)
    if not replaced:
        new_list.append(merged)
    tracks.track_objects_3d = new_list

def _merge_lidar_objects_axis_aligned(objs: List[Object3d]) -> Object3d:
    
    #Conservative merge: world-axis-aligned union of 3D boxes (ignores yaw).
    #This is robust when multiple smaller LiDAR boxes belong to one large object.
    
    # Gather all 8 corners for each box (in world), ignore yaw when union-ing
    mins = np.array([+np.inf, +np.inf, +np.inf], dtype=np.float64)
    maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    for o in objs:
        l, w, h = float(o.size.x), float(o.size.y), float(o.size.z)
        cx, cy, cz = float(o.position.x), float(o.position.y), float(o.position.z)
        yaw = float(o.yaw_angle)

        xs = np.array([+l/2, +l/2, -l/2, -l/2, +l/2, +l/2, -l/2, -l/2], dtype=np.float64)
        ys = np.array([+w/2, -w/2, -w/2, +w/2, +w/2, -w/2, -w/2, +w/2], dtype=np.float64)
        zs = np.array([-h/2, -h/2, -h/2, -h/2, +h/2, +h/2, +h/2, +h/2], dtype=np.float64)

        c, s = math.cos(yaw), math.sin(yaw)
        X = xs * c - ys * s + cx
        Y = xs * s + ys * c + cy
        Z = zs + cz

        mins = np.minimum(mins, np.array([X.min(), Y.min(), Z.min()]))
        maxs = np.maximum(maxs, np.array([X.max(), Y.max(), Z.max()]))

    # Axis-aligned merged box (yaw=0 for safety; you can set yaw to the dominant box if needed)
    size = maxs - mins
    center = 0.5 * (mins + maxs)

    merged = Object3d(
        position=type(objs[0].position)(center[0], center[1], center[2]),
        size=type(objs[0].size)(size[0], size[1], size[2]),
        orientation=objs[0].orientation,   # keep some orientation fields if you use them
        speed=objs[0].speed,
        yaw_angle=0.0,                     # conservative (axis-aligned)
        class_name=objs[0].class_name,     # keep class from first (class may be overwritten by confirmation earlier)
        category_confidence=objs[0].category_confidence,
        existence_confidence=objs[0].existence_confidence
    )
    return merged
"""