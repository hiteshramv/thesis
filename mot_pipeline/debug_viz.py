import cv2
import math
import numpy as np
from typing import Dict, Tuple, List, Optional

# If your helper is a module-level function:
from datastructures.PcdHelper import PcdHelper
# If it's a @staticmethod on a class, use:
# from datastructures.PcdHelper import PcdHelper

# Types
from datastructures.image import ImageData
from datastructures.image_detection import ImageDetectionList
from datastructures.track_object_3d import TrackObject3dList, TrackObject3d
from datastructures.calibration_manager import CalibrationManager

# Colors (BGR)
C_YELLOW  = (0, 225, 255)
C_CYAN    = (255, 225, 0)
C_GREEN   = (0, 200, 0)
C_MAGENTA = (255, 0, 255)
C_WHITE   = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

pcdhelper = PcdHelper()

def _put_text(img, text, org, color=C_WHITE, scale=0.5, thickness=1):
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def _coerce_3x4(E):
    E = np.asarray(E, dtype=np.float64)
    if E.shape == (3, 4):
        return E
    if E.shape == (4, 4):
        return np.hstack([E[:3, :3], E[:3, 3:4]])
    raise ValueError(f"Extrinsic must be 3x4 or 4x4, got {E.shape}")


def _project_box_corners_uv(tobj: TrackObject3d, K, E, image_shape):
    """Return (N,2) uv array for the 8 corners that fall in the image."""
    obj = tobj.object_3d
    l, w, h = float(obj.size.x), float(obj.size.y), float(obj.size.z)
    cx, cy, cz = float(obj.position.x), float(obj.position.y), float(obj.position.z)-2.9
    yaw = float(obj.yaw_angle)

    #print(f"Projecting T{tobj.track_id}: cz={cz}, h={h}")

    xs = np.array([+l/2, +l/2, -l/2, -l/2, +l/2, +l/2, -l/2, -l/2], dtype=np.float64)
    ys = np.array([+w/2, -w/2, -w/2, +w/2, +w/2, -w/2, -w/2, +w/2], dtype=np.float64)
    # NEW (assumes cz is TOP)
    zs = np.array([-h, -h, -h, -h, 0, 0, 0, 0], dtype=np.float64)
    #zs = np.array([-h/2, -h/2, -h/2, -h/2, +h/2, +h/2, +h/2, +h/2], dtype=np.float64)

    c, s = math.cos(yaw), math.sin(yaw)
    X = xs * c - ys * s + cx
    Y = xs * s + ys * c + cy
    Z = zs + cz
    corners = np.vstack([X, Y, Z]).T  # (8,3)

    # If your helper is a staticmethod: uvw = PcdHelper.project_points_uv(corners, (H,W,3), K, E)
    H, W = image_shape[:2]
    uvw = pcdhelper.project_points_uv(corners, (H, W, 3), K, E)
    if uvw is None or uvw.size == 0:
        return None
    return np.vstack([uvw["u"], uvw["v"]]).T  # (n,2)


def _project_box_xyxy(tobj: TrackObject3d, K, E, image_shape):
    uv = _project_box_corners_uv(tobj, K, E, image_shape)
    if uv is None or uv.shape[0] == 0:
        return None
    u1, v1 = float(np.min(uv[:, 0])), float(np.min(uv[:, 1]))
    u2, v2 = float(np.max(uv[:, 0])), float(np.max(uv[:, 1]))
    if (u2 - u1) < 1.0 or (v2 - v1) < 1.0:
        return None
    return (u1, v1, u2, v2)


def draw_overlays_for_camera(
    image: np.ndarray,
    cam_id: int,
    det_list: Optional[ImageDetectionList],
    track_list: TrackObject3dList,
    calib: CalibrationManager,
    assoc_out,  # MeasurementAssociationOutput
    draw_wireframe_3d: bool = False
) -> np.ndarray:
    """
    Returns an annotated copy showing:
      - camera 2D boxes (yellow)
      - projected LiDAR boxes (cyan)
      - associations (green) from assoc_out.class_confirmations (SDIoU + link)
      - merged LiDAR (magenta outline on kept box)
    """
    out = image.copy()
    H, W = out.shape[:2]

    # 1) 2D camera boxes
    if det_list is not None:
        for j, det in enumerate(det_list.detections):
            x1, y1, x2, y2 = int(det.x-(det.w/2)), int(det.y-(det.h/2)), int(det.x+(det.w/2)), int(det.y+(det.h/2))
            cv2.rectangle(out, (x1, y1), (x2, y2), C_YELLOW, 2)
            label = det.class_name or ""
            if hasattr(det, "score") and det.score is not None:
                label += f" {det.score:.2f}"
            _put_text(out, label, (x1, max(12, y1 - 5)), C_YELLOW, 0.45)

    # Camera intrinsics/extrinsics
    K = calib.get_intrinsic(cam_id)
    E = _coerce_3x4(calib.get_extrinsic(cam_id))

    # 2) projected LiDAR boxes
    for tobj in track_list.track_objects_3d:
        xyxy = _project_box_xyxy(tobj, K, E, (H, W))
        if xyxy is None:
            continue
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), C_CYAN, 2)
        _put_text(out, f"T{int(tobj.track_id)} {(tobj.object_3d.class_name or '')}",
                  (x1, min(H - 5, y2 + 14)), C_CYAN, 0.5)

        if draw_wireframe_3d:
            uv = _project_box_corners_uv(tobj, K, E, (H, W))
            if uv is not None and uv.shape[0] >= 8:
                edges = [(0,1),(1,2),(2,3),(3,0),
                         (4,5),(5,6),(6,7),(7,4),
                         (0,4),(1,5),(2,6),(3,7)]
                for i, j in edges:
                    u1, v1 = int(uv[i, 0]), int(uv[i, 1])
                    u2, v2 = int(uv[j, 0]), int(uv[j, 1])
                    cv2.line(out, (u1, v1), (u2, v2), (180, 180, 255), 2)

    # 3) associations from class_confirmations (track_id, camera_id, det_index, sdiou)
    # Group per camera and draw links + SDIoU
    confs = [c for c in getattr(assoc_out, "class_confirmations", []) if int(c.camera_id) == int(cam_id)]
    if det_list is not None and confs:
        for c in confs:
            if c.det_index >= len(det_list.detections):
                continue
            d = det_list.detections[c.det_index]
            cx = int(d.x + 0.5 * d.w)
            cy = int(d.y + 0.5 * d.h)

            # projected LiDAR center
            tobj = next((t for t in track_list.track_objects_3d if int(t.track_id) == int(c.track_id)), None)
            if tobj is None:
                continue
            xyxy = _project_box_xyxy(tobj, K, E, (H, W))
            if xyxy is None:
                continue
            lx = int(0.5 * (xyxy[0] + xyxy[2]))
            ly = int(0.5 * (xyxy[1] + xyxy[3]))

            cv2.line(out, (lx, ly), (cx, cy), C_GREEN, 2)
            _put_text(out, f"SDIoU={c.sdiou:.2f}", (min(lx, cx) + 4, min(ly, cy) - 6), C_GREEN, 0.5)

            # class update hint
            prev_cls = c.prev_class or ""
            new_cls = c.confirmed_class or ""
            if new_cls and new_cls != prev_cls:
                _put_text(out, f"{prev_cls}->{new_cls}", (min(lx, cx) + 4, min(ly, cy) - 22), C_GREEN, 0.5)

    # 4) merged LiDAR boxes — highlight the kept box (magenta)
    for m in getattr(assoc_out, "merges", []):
        if int(m.camera_id) != int(cam_id):
            continue
        kept_id = int(m.new_track_object.track_id)
        tobj = next((t for t in track_list.track_objects_3d if int(t.track_id) == kept_id), None)
        if tobj is None:
            continue
        xyxy = _project_box_xyxy(tobj, K, E, (H, W))
        if xyxy is None:
            continue
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), C_MAGENTA, 2)
        _put_text(out, "MERGED", (x1, max(12, y1 - 5)), C_MAGENTA, 0.55)

    # quick legend
    _put_text(out, "2D det: yellow | LiDAR proj: cyan | assoc: green | merged: magenta", (10, 20), C_WHITE, 0.5)
    return out


def render_overlays_for_all_cams(
    images: List[ImageData],
    detections: List[ImageDetectionList],
    track_list: TrackObject3dList,
    calib: CalibrationManager,
    assoc_out,
    show: bool = False,
    out_dir: Optional[str] = None,
    frame_id: Optional[int] = None,
    draw_wireframe_3d: bool = False
) -> List[np.ndarray]:
    """
    Returns canvases for each input image (same order).
    Optionally shows with cv2.imshow and/or saves to out_dir.
    """
    # quick lookup for detections per cam
    det_by_cam: Dict[int, ImageDetectionList] = {int(d.camera_id): d for d in detections}
    canvases: List[np.ndarray] = []

    for idx, img in enumerate(images):
        if img is None:
            canvases.append(None)
            continue
        cam_id = int(img.camera_id)
        det_list = det_by_cam.get(cam_id, None)
        canvas = draw_overlays_for_camera(
            image=img.data,
            cam_id=cam_id,
            det_list=det_list,
            track_list=track_list,
            calib=calib,
            assoc_out=assoc_out,
            draw_wireframe_3d=draw_wireframe_3d
        )
        canvases.append(canvas)

        if out_dir is not None:
            import os
            os.makedirs(out_dir, exist_ok=True)
            name = f"cam{cam_id:02d}_{(frame_id if frame_id is not None else 0):06d}.jpg"
            cv2.imwrite(os.path.join(out_dir, name), canvas)

        if show:
            cv2.imshow(f"INANI cam{cam_id}", canvas)
    if show:
        cv2.waitKey(1)

    return canvases
