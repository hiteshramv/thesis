import os
import cv2
import numpy as np
from datetime import datetime

# -------------------------------------------------
# Fixed BGR colors per class name
# -------------------------------------------------
CLASS_COLORS = {
    "background": (128, 128, 128),   # gray
    "adult":      (36, 255, 12),     # light green
    "child":      (0, 165, 255),     # orange
    "bicycle":    (255, 0, 0),       # blue
    "motorcycle": (255, 255, 0),     # cyan
    "car":        (0, 255, 0),       # green
    "van":        (0, 255, 255),     # yellow
    "truck":      (0, 128, 255),     # amber
    "bus":        (0, 0, 255),       # red
}


def _put_label(img, text, org, color, font_scale=0.6, thickness=2):
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = org
    x = max(0, x)
    y = max(th + baseline + 2, y)  # ensure we don't go above image
    # black filled rectangle as text background
    cv2.rectangle(img, (x, y - th - baseline - 2), (x + tw + 4, y + 2), (0, 0, 0), -1)
    # the actual text
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def show(
    images,
    detections,
    window_height=640,
    save_images=False,
    show_hstack=False,
    save_hstack=False,
    save_dir="visualization_frames",
    window_name="All Cameras",
):
    """
    Visualize all cameras in a single window (optional) and save each camera image individually.

    Parameters
    ----------
    images : list
        List of image objects with at least `.data` (numpy array BGR),
        and ideally `.camera_id` and `.time_stamp` attributes.
    detections : list
        List aligned with `images`. Each item has `.detections`, which is an iterable
        of detection objects with attributes: x, y, x, h, class_name, score, tracker_id (optional).
    window_height : int
        Display height for each panel in the hstack (display only).
    save_images : bool
        If True, saves each annotated camera image individually (full-res).
    show_hstack : bool
        If True, shows a single window with hstack visualization for quick monitoring.
    save_hstack : bool
        If True, also saves the hstack image (for debugging/overview). By default False.
    save_dir : str
        Base output directory for saved images.
    window_name : str
        Title for the visualization window.
    """
    os.makedirs(save_dir, exist_ok=True)
    vis_panels = []

    # Fallback timestamp used when an image lacks a time_stamp
    fallback_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")

    # Process each image + its detections
    for img_obj, det_list in zip(images, detections):
        # Full-resolution copy for saving
        full = img_obj.data.copy()
        cam_id = getattr(img_obj, "camera_id", "X")
        ts = getattr(img_obj, "time_stamp", None) or fallback_ts

        # Draw all detections (per-class colors)
        for det in det_list.detections:
            x1, y1, x2, y2 = int(det.x-(det.w/2)), int(det.y-(det.h/2)), int(det.x+(det.w/2)), int(det.y+(det.h/2))
            cls_name = getattr(det, "class_name", "unknown")
            color = CLASS_COLORS.get(cls_name, (0, 255, 0))  # default green if unknown

            # Box
            cv2.rectangle(full, (x1, y1), (x2, y2), color, 2)

            # Label text
            score = getattr(det, "score", None)
            score_txt = f"{score:.2f}" if isinstance(score, (float, int)) else "?"
            tracker_id = getattr(det, "tracker_id", None)
            id_txt = f" ID:{tracker_id}" if tracker_id is not None else ""
            label = f"{cls_name}:{score_txt}{id_txt}"

            # Text (with background) slightly above the box
            _put_label(full, label, (x1, y1 - 8), color)

        # Build a resized panel for the hstack (display only)
        h, w = full.shape[:2]
        scale = window_height / max(1, h)
        disp = cv2.resize(full, (int(w * scale), window_height))
        cv2.putText(disp, f"Camera {cam_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        vis_panels.append(disp)

        # Save this camera image individually (annotated, full-res)
        if save_images:
            cam_dir = os.path.join(save_dir, f"cam_{cam_id}")
            os.makedirs(cam_dir, exist_ok=True)
            out_name = f"cam_{cam_id}__{ts}.png"
            cv2.imwrite(os.path.join(cam_dir, out_name), full)

    # Show hstack window (optional)
    if show_hstack and vis_panels:
        vis = cv2.hconcat(vis_panels)
        cv2.imshow(window_name, vis)
        cv2.waitKey(1)
        if save_hstack:
            # save once per call, using the first image's ts if available
            ts0 = getattr(images[0], "time_stamp", None) if images else None
            ts0 = ts0 or fallback_ts
            cv2.imwrite(os.path.join(save_dir, f"hstack__{ts0}.png"), vis)




"""
import cv2
import numpy as np
import os


def show(images, detections, window_height=640, save_images=True):
    
    #Visualize side-by-side, resizing all to fixed height.
    
    imgs = []
    for img_obj, det_list in zip(images, detections):
        img = img_obj.data.copy()
        for det in det_list.detections:
            x1, y1, x2, y2 = int(det.x), int(det.y), int(det.x), int(det.h)
            id_str = f"ID:{det.tracker_id}" if hasattr(det, "tracker_id") and det.tracker_id is not None else ""
            label = f"{det.class_name}:{det.score:.2f} {id_str}"
            #label = f"{det.class_name}:{det.score:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        # Resize to window_height keeping aspect ratio
        h, w = img.shape[:2]
        scale = window_height / h
        img_small = cv2.resize(img, (int(w * scale), window_height))
        cam_title = f"Camera {img_obj.camera_id}"
        cv2.putText(img_small, cam_title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        imgs.append(img_small)
    vis = cv2.hconcat(imgs)
    cv2.imshow("All Cameras", vis)
    cv2.waitKey(1)

    if save_images:
        SAVE_DIR = "visualization_frames"
        os.makedirs(SAVE_DIR, exist_ok=True)
        # Use first camera's timestamp, or current time if not available
        ts = None
        if images and hasattr(images[0], 'time_stamp'):
            ts = images[0].time_stamp
        if not ts:
            from datetime import datetime
            ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        filename = os.path.join(SAVE_DIR, f"vis_{ts}.png")
        cv2.imwrite(filename, vis)
"""