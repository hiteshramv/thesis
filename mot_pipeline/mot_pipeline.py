#ros2 bag play "/home/vh17r/thesis/inani_ros2_bag/inani__2025-07-07_07-26-43-168/inani__2025-07-07_07-26-43-168__bag1" --clock
#ros2 bag play "/home/vh17r/thesis/inani_ros2_bag/inani__2025-07-07_07-33-03-367/inani__2025-07-07_07-33-03-367__bag1" --clock
#ros2 bag play "/home/vh17r/thesis/inani_ros2_bag/inani__2025-07-07_07-37-36-291/inani__2025-07-07_07-37-36-291__bag1" --clock
#ros2 bag play "/home/vh17r/thesis/inani_ros2_bag/inani__2025-07-07_07-42-54-306/inani__2025-07-07_07-42-54-306__bag1" --clock

#ros2 bag play "/home/vh17r/ros2_yolo/inani/inani__2025-10-17_07-30-27-642__bag1" --clock

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import logging
import rclpy

from dataloading import SensorDataLoader
from detector2D import Detector2D
from vehicle_pipeline import run_measurement_association_sdiou
from datastructures.calibration_manager import CalibrationManager
from datastructures.final_calibration_dict import calibration_data
from debug_viz import render_overlays_for_all_cams

# ---------------- logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("mot_pipeline.txt"), logging.StreamHandler()]
)
logger = logging.getLogger("mot_pipeline")


def _build_shapes_by_cam(images):
    """Return {cam_id: (H,W)} and {cam_id: ImageData} for all non-None images."""
    image_shapes_by_cam = {}
    images_by_cam = {}
    for img in images:
        if img is None:
            continue
        H, W = img.data.shape[:2]
        cam_id = int(img.camera_id)
        image_shapes_by_cam[cam_id] = (H, W)
        images_by_cam[cam_id] = img
    return image_shapes_by_cam, images_by_cam


def _log_assoc_results(assoc_out, detections):
    # per-camera counts
    for dl in detections:
        cam_id = int(dl.camera_id)
        n2d = len(dl.detections)
        u_lidar = assoc_out.unmatched_tracks_by_cam.get(cam_id, [])
        u_2d = assoc_out.unmatched_detections_by_cam.get(cam_id, [])
        logger.info(f"[cam{cam_id}] 2D={n2d}  unmatched LiDAR={len(u_lidar)}  unmatched 2D={len(u_2d)}")

    # class confirmations
    for c in assoc_out.class_confirmations:
        prev = c.prev_class or ""
        new = c.confirmed_class or ""
        changed = " (UPDATED)" if new and new != prev else ""
        logger.info(f"[CONFIRM] T{c.track_id} via cam{c.camera_id}: SDIoU={c.sdiou:.2f} {prev}->{new}{changed}")

    # merges
    for m in assoc_out.merges:
        sdiou_str = ", ".join(f"{s:.2f}" for s in m.sdiou_values)
        logger.info(f"[MERGE] cam{m.camera_id} det#{m.det_index} "
                    f"LiDAR {m.track_ids_merged} -> kept {m.new_track_object.track_id} "
                    f"(SDIoUs=[{sdiou_str}])")


def main(args=None):
    rclpy.init(args=args)

    # sensor reader + detector
    node = SensorDataLoader()
    detector = Detector2D(model_path='yolo11s_inani.engine', conf=0.35)

    # calibration
    calib = CalibrationManager(calibration_data)

    # optional compatibility (car<->van allowed; bus/truck strict)
    class_compat = {"bus": ["bus"], "truck": ["truck"], "car": ["car"], "van": ["van"]}

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            # inputs
            images = node.get_latest_images()      # List[ImageData], may contain None
            track_list = node.get_latest_tracks()  # TrackObject3dList

            # skip until at least one camera is available
            if not images or all(img is None for img in images):
                continue

            # log timestamps of available cameras
            for img in images:
                if img is not None:
                    logger.info(f"Camera: {img.camera_id}, Timestamp: {img.time_stamp}")

            # shapes and available images
            image_shapes_by_cam, images_by_cam = _build_shapes_by_cam(images)
            if not images_by_cam:
                continue

            # run 2D detector on available images in ascending cam_id order
            cam_ids_sorted = sorted(images_by_cam.keys())
            imgs_for_detector = [images_by_cam[cid] for cid in cam_ids_sorted]
            detections = detector.detect(imgs_for_detector)
            # NOTE: Detector2D must set .camera_id in each ImageDetectionList matching the input images

            # association
            t0 = time.time()
            assoc_out = run_measurement_association_sdiou(
                tracks=track_list,
                detections=detections,             # list[ImageDetectionList] (each has .camera_id)
                image_shapes_by_cam=image_shapes_by_cam,
                calib=calib,
                min_sdiou=0.30,
                large_box_px_area=14000,
                class_map=class_compat,
                update_class_on_confirm=True,
                perform_merging=True
            )
            elapsed_ms = (time.time() - t0) * 1000.0
            logger.info(f"Association took: {elapsed_ms:.1f} ms")

            # logs (counts, confirmations, merges)
            _log_assoc_results(assoc_out, detections)


            canvases = render_overlays_for_all_cams(
                images=[images_by_cam[cid] for cid in sorted(images_by_cam.keys())],  # or your original `images` list
                detections=detections,
                track_list=track_list,
                calib=calib,
                assoc_out=assoc_out,
                show=True,                # set False to avoid GUI popups
                out_dir="/tmp/inani_viz", # or None to skip saving
                frame_id=node.get_frame_index() if hasattr(node, "get_frame_index") else None,
                draw_wireframe_3d=True   # True if you want corner wireframes
                )

            # loop continues; no visualization, no OpenCV calls

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()




"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import rclpy
from dataloading import SensorDataLoader
from detector2D import Detector2D

from vehicle_pipeline import run_measurement_association_sdiou
from datastructures.calibration_manager import CalibrationManager
from datastructures.final_calibration_dict import calibration_data

import visualization
import time
import logging
from typing import Dict, Tuple, List

# Configure logging: writes to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("mot_pipeline.txt"), logging.StreamHandler()]
)
logger = logging.getLogger("mot_pipeline")


# ---------- small helpers to map association output -> viz overlays ----------

def _build_image_shapes_by_cam(images) -> Dict[int, Tuple[int, int]]:
    shapes = {}
    for img in images:
        if img is None:
            continue
        cam_id = int(getattr(img, "camera_id", -1))
        if cam_id >= 0:
            h, w = img.data.shape[:2]
            shapes[cam_id] = (h, w)
    return shapes

def _build_matches_by_cam(assoc_out) -> Dict[int, List[tuple]]:
    
    #Expect assoc_out to maybe have .per_cam_matches (our implementation),
    #or return {} if not available (viz will still show projections).
    
    # Our run_measurement_association_sdiou returned per-cam matches internally;
    # if you exposed them, prefer that. Otherwise leave empty (no green lines).
    if hasattr(assoc_out, "per_cam_matches"):
        return assoc_out.per_cam_matches  # {cam_id: [(track_id, det_idx, sdiou), ...]}
    if isinstance(assoc_out, dict) and "per_cam_matches" in assoc_out:
        return assoc_out["per_cam_matches"]
    return {}

def _build_confirm_map(assoc_out) -> Dict[int, Tuple[str, str]]:
    confirm_map = {}
    items = getattr(assoc_out, "class_confirmations", None)
    if items is None and isinstance(assoc_out, dict):
        items = assoc_out.get("class_confirmations", [])
    items = items or []
    for c in items:
        prev_cls = getattr(c, "prev_class", None) if not isinstance(c, dict) else c.get("prev_class")
        new_cls  = getattr(c, "confirmed_class", None) if not isinstance(c, dict) else c.get("confirmed_class")
        tid      = getattr(c, "track_id", None) if not isinstance(c, dict) else c.get("track_id")
        if tid is not None and prev_cls != new_cls:
            confirm_map[int(tid)] = (prev_cls or "", new_cls or "")
    return confirm_map

def _collect_merged_ids(assoc_out) -> List[int]:
    merged_ids = []
    items = getattr(assoc_out, "merges", None)
    if items is None and isinstance(assoc_out, dict):
        items = assoc_out.get("merges", [])
    items = items or []
    for m in items:
        ids = getattr(m, "track_ids_merged", None) if not isinstance(m, dict) else m.get("track_ids_merged")
        if ids:
            merged_ids.extend(int(x) for x in ids)
    return merged_ids


def main(args=None):
    rclpy.init(args=args)
    node = SensorDataLoader()
    detector = Detector2D(model_path='yolo11s_inani.engine', conf=0.35)

    # Your CalibrationManager seems to accept the dict directly
    calib = CalibrationManager(calibration_data)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            start_time = time.time()
            images = node.get_latest_images()     # [ImageData, ImageData, ImageData]
            track_list = node.get_latest_tracks() # TrackObject3dList

            # Log image info for all cameras in the set
            for img in images:
                if img is not None:
                    logger.info(f"Camera: {img.camera_id}, Timestamp: {img.time_stamp}")

            # Ensure we have all three images
            if not images or any(img is None for img in images):
                continue

            # Shapes per camera
            image_shapes_by_cam = _build_image_shapes_by_cam(images)

            # 2D detections
            detections = detector.detect(images)

            # ---- Measurement-level association (projection + SDIoU + merge) ----
            assoc_out = run_measurement_association_sdiou(
                tracks=track_list,
                detections=detections,            # list[ImageDetectionList]
                image_shapes_by_cam=image_shapes_by_cam,
                calib=calib,
                min_sdiou=0.10,
                large_box_px_area=140000,
                class_map={"bus": ["bus"], "truck": ["truck"]},  # tighten if you want
                update_class_on_confirm=True,
                perform_merging=True
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Processing time for this set of 3 images: {elapsed*1000:.1f} ms")            

            # Build overlay maps for the visualizer
            matches_by_cam = _build_matches_by_cam(assoc_out)
            confirm_map    = _build_confirm_map(assoc_out)
            merged_ids     = _collect_merged_ids(assoc_out)

            #if matches_by_cam is 

            # ---- Visualization (now includes overlays from association) ----
            visualization.show(
                images=images,
                detections=detections,
                show_hstack=True,            # show live strip
                save_hstack=False,           # enable if you want to dump overview images
                save_images=False,           # enable to save each cam image
                save_dir="visualization_frames",
                window_name="INANI — Measurement Association (LiDAR↔Camera)",
                # overlays:
                track_list=track_list,
                calib=calib,
                image_shapes_by_cam=image_shapes_by_cam,
                matches_by_cam=matches_by_cam,   # green lines + SDIoU
                confirm_map=confirm_map,         # prev->new label tags
                merged_ids=merged_ids,           # grey (or magenta) LiDAR boxes
                draw_projected=True,
                draw_matches=True,
            )

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

"""
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import rclpy
from dataloading import SensorDataLoader
from detector2D import Detector2D
from vehicle_pipeline import run_measurement_association_sdiou
from datastructures.calibration_manager import CalibrationManager
from datastructures.final_calibration_dict import calibration_data

import visualization
import time
import logging

# Configure logging: writes to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("mot_pipeline.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mot_pipeline")

def main(args=None):
    rclpy.init(args=args)
    node = SensorDataLoader()
    detector = Detector2D(model_path='yolo11s_inani.engine', conf=0.35)
    calib = CalibrationManager(calibration_data)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            start_time = time.time()
            images = node.get_latest_images()
            track_list = node.get_latest_tracks()

            # Log image info for all cameras in the set
            for i, img in enumerate(images):
                if img is not None:
                    logger.info(f"Camera: {img.camera_id}, Timestamp: {img.time_stamp}")  
                #else:
                    #logger.warning(f"Image {i+1} is None.")

            if all(img is not None for img in images):
                image_shapes_by_cam = {
                    images[0].camera_id: images[0].data.shape[:2],   # (H, W)
                    images[1].camera_id: images[1].data.shape[:2],
                    images[2].camera_id: images[2].data.shape[:2],
                    }
                detections = detector.detect(images)

                
                assoc_out = run_measurement_association_sdiou(
                    tracks=track_list,
                    detections=detections,            # <— list, not dict
                    image_shapes_by_cam=image_shapes_by_cam,
                    calib=calib,
                    min_sdiou=0.30,
                    large_box_px_area=14000,
                    class_map={"bus": ["bus"], "truck": ["truck"]},
                    update_class_on_confirm=True,
                    perform_merging=True
                )

                elapsed = time.time() - start_time
                logger.info(f"Processing time for this set of 3 images: {elapsed*1000:.1f} ms")

                visualization.show(images, detections)


    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
"""