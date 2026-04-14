#ros2 bag play "/home/vh17r/ros2_yolo/inani/inani__2025-12-10_08-13-42-168/inani__2025-12-10_08-13-42-168__bag1"
#ros2 bag play "/home/vh17r/ros2_yolo/inani/inani__2025-12-10_08-13-42-168/inani__2025-12-10_08-13-42-168__bag2"

#ros2 bag play "/home/vh17r/ros2_yolo/inani/inani__2025-12-05_08-08-56-478/inani__2025-12-05_08-08-56-478__bag1" --clock
#ros2 bag play "/home/vh17r/ros2_yolo/inani/inani__2025-12-05_08-08-56-478/inani__2025-12-05_08-08-56-478__bag2" --clock

#ros2 bag play "/home/vh17r/ros2_yolo/inani/inani__2026-02-11_07-38-45-114/inani__2026-02-11_07-38-45-114__bag1" --clock
#ros2 bag play "/home/vh17r/ros2_yolo/inani/inani__2026-02-11_07-38-45-114/inani__2026-02-11_07-38-45-114__bag2" --clock

#ros2 bag play "/home/vh17r/ros2_yolo/inani/inani__2026-02-10_12-14-54-919/inani__2026-02-10_12-14-54-919__bag1" --clock
#ros2 bag play "/home/vh17r/ros2_yolo/inani/inani__2026-02-10_12-14-54-919/inani__2026-02-10_12-14-54-919__bag2" --clock

#ros2 bag play "/home/vh17r/ros2_yolo/inani/inani__2026-02-27_07-43-40-839/inani__2026-02-27_07-43-40-839__bag1" --clock
#ros2 bag play "/home/vh17r/ros2_yolo/inani/inani__2026-02-27_07-43-40-839/inani__2026-02-27_07-43-40-839__bag2" --clock

import sys
import os
import time
from collections import defaultdict, deque
import numpy as np
import threading
from typing import Dict, List, Optional, Tuple

from rclpy.executors import SingleThreadedExecutor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datastructures.lidar_detections import (Object3d, Position3d, Size3d, Orientation3d, Speed3d, Object3dList)
from datastructures.image_detection import ImageDetection, ImageDetectionList
from datastructures.PcdHelper import PcdHelper
from datastructures.final_calibration_dict import calibration_data

#from kalman_tracker import KalmanMultiObjectTracker
from ekf_tracker import KalmanMultiObjectTracker

import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter

# ROS Message Imports
from nse_ros_interfaces.msg import NSEObjectList
from camera_2d_objects_msgs.msg import MultiCamera2DObjects
from sensor_msgs.msg import PointCloud2, Image
#from track_objects_3d_msgs.msg import TrackObjectList

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

# VIS
import cv2
from cv_bridge import CvBridge

# FUSION (projection + Hungarian)
from sensor_fusion import SensorFusion, MultiCamProjector, xywh_to_xyxy
from helper_functions_lidar import choose_display_tracks_with_aliasing, cleanup_group_parent, class_color


# CONFIGURATION
BAG_1_PATH = "/home/vh17r/ros2_yolo/inani/inani__2026-02-27_07-28-36-251/inani__2026-02-27_07-28-36-251__bag1/inani__2026-02-27_07-28-36-251__bag1_0.db3"
BAG_2_PATH = "/home/vh17r/ros2_yolo/inani/inani__2026-02-27_07-28-36-251/inani__2026-02-27_07-28-36-251__bag2/inani__2026-02-27_07-28-36-251__bag2_0.db3"
#BAG_1_PATH = "/home/vh17r/ros2_yolo/inani/inani__2026-02-27_07-33-37-818/inani__2026-02-27_07-33-37-818__bag1/inani__2026-02-27_07-33-37-818__bag1_0.db3"
#BAG_2_PATH = "/home/vh17r/ros2_yolo/inani/inani__2026-02-27_07-33-37-818/inani__2026-02-27_07-33-37-818__bag2/inani__2026-02-27_07-33-37-818__bag2_0.db3"
#BAG_1_PATH = "/home/vh17r/ros2_yolo/inani/inani__2026-02-27_07-38-39-280/inani__2026-02-27_07-38-39-280__bag1/inani__2026-02-27_07-38-39-280__bag1_0.db3"
#BAG_2_PATH = "/home/vh17r/ros2_yolo/inani/inani__2026-02-27_07-38-39-280/inani__2026-02-27_07-38-39-280__bag2/inani__2026-02-27_07-38-39-280__bag2_0.db3"
#BAG_1_PATH = "/home/vh17r/ros2_yolo/inani/inani__2026-02-27_07-43-40-839/inani__2026-02-27_07-43-40-839__bag1/inani__2026-02-27_07-43-40-839__bag1_0.db3"
#BAG_2_PATH = "/home/vh17r/ros2_yolo/inani/inani__2026-02-27_07-43-40-839/inani__2026-02-27_07-43-40-839__bag2/inani__2026-02-27_07-43-40-839__bag2_0.db3"

#TOPIC_TRACKS = "/object_list/tracks"
TOPIC_TRACKS = "/object_list/fused/tracked/processed"
#TOPIC_TRACKS = "/inani_final_tracks"
TOPIC_YOLO   = "/three_cam_yolov11s_inani_2d_detections"
TOPIC_OUSTER = "/ouster/points"
TOPIC_DOME   = "/dome/points"

TOPIC_CAM1 = "/camera1/pylon_ros2_camera_node_camera1/image_raw"
TOPIC_CAM2 = "/camera2/pylon_ros2_camera_node_camera2/image_raw"
TOPIC_CAM3 = "/camera3/pylon_ros2_camera_node_camera3/image_raw"

TOPIC_TYPES = {
    TOPIC_TRACKS: NSEObjectList,
    #TOPIC_TRACKS: TrackObjectList,
    TOPIC_OUSTER: PointCloud2,
    TOPIC_DOME:   PointCloud2,
    TOPIC_YOLO:   MultiCamera2DObjects,
    TOPIC_CAM1:   Image,
    TOPIC_CAM2:   Image,
    TOPIC_CAM3:   Image,
}

CLASS_LIST = ["Background","Adult","Child","Cyclist","Motorcycle","Car","Van","Truck","Bus"]
CLASS_VEHICLES = ["Car", "Van", "Bus", "Truck", "Motorcycle"]

dome_extrinsic = np.array(calibration_data["lidars"]["dome"]["extrinsic"])


def _get(obj, path, default=None):
    cur = obj
    for p in path.split("."):
        if cur is None:
            return default
        cur = getattr(cur, p, None)
    return default if cur is None else cur

def center_xyxy(box):
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

def draw_xyxy(img, box, color, label="", thickness=2):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(img, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_mot_tracks_overlay(cv_img, cam_id, projector, tracks_out):
    """
    Draw MOT tracks (projected 3D bbox) + track_id on the image.
    tracks_out: List[TrackOutput] from KalmanMultiObjectTracker
    """
    out = cv_img.copy()

    for t in tracks_out:
        st = t.state
        if st is None or len(st) < 7:
            continue

        cx, cy, cz = float(st[0]), float(st[1]), float(st[2])
        yaw = float(st[3])
        l, w, h = float(st[4]), float(st[5]), float(st[6])

        class _Vec:
            def __init__(self, x, y, z):
                self.x, self.y, self.z = x, y, z

        class _Obj:
            pass

        o = _Obj()
        o.position = _Vec(cx, cy, cz)
        o.size = _Vec(l, w, h)
        o.yaw_angle = yaw

        bb = projector.project_object_to_xyxy(o, cam_id)
        if bb is None:
            continue

        # OPTIONAL: draw projected bbox + ID
        #draw_xyxy(out, bb, (0, 255, 255), label=f"ID:{t.track_id}", thickness=2)

    return out

def draw_assoc_overlay(cv_img, cam_id, projector, lidar_objs, det_list, assoc_res):
    """
    - YOLO boxes: green
    - projected LiDAR boxes: blue
    - match lines: red with score text
    """
    out = cv_img.copy()

    yolo_boxes = [xywh_to_xyxy(d.x, d.y, d.w, d.h) for d in det_list.detections]
    for j, (b, d) in enumerate(zip(yolo_boxes, det_list.detections)):
        draw_xyxy(out, b, (0, 255, 0), label=f"Y{j}:{d.class_name}")

    lidar_boxes = []
    for i, obj in enumerate(lidar_objs):
        pb = projector.project_object_to_xyxy(obj, cam_id)
        lidar_boxes.append(pb)
        if pb is not None:
            draw_xyxy(out, pb, (255, 0, 0), label=f"LP{i}:{obj.class_name}:{obj.size.z:.3f}")

    for (li, di, s) in assoc_res.matches:
        lb = lidar_boxes[li]
        if lb is None:
            continue
        db = yolo_boxes[di]
        lc = center_xyxy(lb)
        dc = center_xyxy(db)
        cv2.line(out, (int(lc[0]), int(lc[1])), (int(dc[0]), int(dc[1])), (0, 0, 255), 2)
        cv2.putText(out, f"{s:.2f}", (int(dc[0]), int(dc[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.putText(out,
                f"cam{cam_id}  matches={len(assoc_res.matches)}  "
                f"uL={len(assoc_res.unmatched_lidar)}  uC={len(assoc_res.unmatched_cam)}",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out

def make_box_corners(cx, cy, cz, l, w, h, yaw):
    hx, hy, hz = l/2.0, w/2.0, h/2.0
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

    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0],
                   [s,  c, 0],
                   [0,  0, 1]], dtype=np.float32)

    corners = (Rz @ corners.T).T
    corners[:, 0] += cx
    corners[:, 1] += cy
    corners[:, 2] += cz
    return corners

def box_edges_line_list(corners):
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    pts = []
    for i, j in edges:
        pts.append(corners[i])
        pts.append(corners[j])
    return pts


class DualBagProcessor(Node):

    def __init__(self):
        super().__init__("dual_bag_processor")
        self.processed_frame_count = 0
        self.ENABLE_MERGE = False
        self.pcd_helper = PcdHelper()
        self.msg_buffer = defaultdict(dict)

        self.x_min, self.x_max = 0.2, 24.0
        self.y_min, self.y_max = -20.0, 20.0

        # visualization
        self.bridge = CvBridge()
        self.enable_viz = True
        self.viz_every_n = 1
        self.img_slop = 0.08

        # SAVE IMAGES
        self.save_viz = False
        self.save_every_n = 1
        self.save_dir = os.path.join(os.path.dirname(__file__), "saved_viz_tracks")
        os.makedirs(self.save_dir, exist_ok=True)

        # --- STEP MODE (NEW) ---
        # Press any key to go to next frame, 'q' quits.
        self.step_mode = True
        self.step_window_name = "INANI- 3 Cam Assoc Overlay"

        self.pub_merged_cloud = self.create_publisher(PointCloud2, "/inani_combined_points", 10)

        # projector + fusion
        self.projector = MultiCamProjector(calibration_data)
        self.fusion = SensorFusion(self.projector, score_fn="iou", threshold=0.2, resolve_global_unique_lidar=True)

        self.mot = KalmanMultiObjectTracker(
            #motion_model='ca',
            motion_model='ctra',
            max_age=7,
            min_hits=3,
            #dist3d_thresh=1.5, #gating(linear)
            dist3d_thresh = 2,  #gating(nonlinear)
            iou2d_thresh=0.4
        )
        self._group_parent = {}
        self._group_last_seen: Dict[int, int] = {}
        self.pub_markers = self.create_publisher(MarkerArray, "/inani_track_objects_visualization", 10)
        self.viz_only_confirmed = True
        self.viz_only_recent = True

        self.get_logger().info("Dual Bag Processor Initialized.")

    def create_reader(self, path, topics):
        storage_options = StorageOptions(uri=path, storage_id='sqlite3')
        converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        reader = SequentialReader()
        reader.open(storage_options, converter_options)
        if topics:
            reader.set_filter(StorageFilter(topics=topics))
        return reader

    def get_next_msg(self, reader):
        while reader.has_next():
            topic, data, t_rec = reader.read_next()
            if topic not in TOPIC_TYPES:
                continue
            msg = deserialize_message(data, TOPIC_TYPES[topic])
            ts = (msg.header.stamp.sec, msg.header.stamp.nanosec)
            return ts, topic, msg
        return None

    def merge_pointclouds(self, ouster_msg, dome_msg):
        ouster_np = self.pcd_helper.pc2msg_to_recarray(ouster_msg)
        dome_np = self.pcd_helper.pc2msg_to_recarray(dome_msg)
        dome_trans = self.pcd_helper.transform_coordinates_return_with_orginal_intensity(
            cloud=dome_np, transformation_mat=dome_extrinsic, location_fields=["x", "y", "z"]
        )
        combined = np.concatenate([ouster_np, dome_trans])
        if "index" in combined.dtype.names:
            combined["index"] = np.arange(len(combined), dtype=combined["index"].dtype)
        return self.pcd_helper.recarray_to_pc2msg(combined, header=ouster_msg.header)

    # method for rosbags without topic - "/inani_final_tracks"
    def convert_3d_objects_ros_to_custom(self, ros_msg, ts_float):
        custom_3d_obj_list = []

        if not hasattr(ros_msg, "objects_count"):
            raise AttributeError("ROS msg missing objects_count")
        N = int(getattr(ros_msg, "objects_count"))
        if N <= 0:
            return Object3dList(time_stamp=ts_float, objects_3d=[])
        print(N)
        names = getattr(ros_msg, "objects_class_name", [])
        xs = getattr(ros_msg, "objects_position_x", [])
        ys = getattr(ros_msg, "objects_position_y", [])
        zs = getattr(ros_msg, "objects_position_z", [])

        sx = getattr(ros_msg, "objects_scale_x", [])
        sy = getattr(ros_msg, "objects_scale_y", [])
        sz = getattr(ros_msg, "objects_scale_z", [])

        yaws = getattr(ros_msg, "objects_yaw", [])
        scores = getattr(ros_msg, "objects_score", [])

        # N = min(
        #     n,
        #     len(xs), len(ys), len(zs),
        #     len(sx), len(sy), len(sz),
        #     len(yaws),
        #     len(scores) if scores is not None else n,
        #     len(names) if names is not None else n,
        # )

        def _to_str(v) -> str:
            if hasattr(v, "data"):
                return str(v.data)
            return str(v)

        for i in range(N):
            x = float(xs[i])
            y = float(ys[i])
            z = float(zs[i]) - 2.9

            cname = _to_str(names[i]) if names is not None and i < len(names) else "Unknown"
            if not (self.x_min <= x <= self.x_max):
                continue
            if not (self.y_min <= y <= self.y_max):
                continue
            if sz[i] >=2.5 and cname in ("Pedestrian","Adult","Child"):
                continue
            #cname = _to_str(names[i]) if names is not None and i < len(names) else "Unknown"
            if cname=="Pedestrian":
                #print(f"{sz[i]}")
                if sz[i]<=1.4:
                    cname = "Child"
                else:
                    cname = "Adult"

            #if cname in CLASS_VEHICLES:
            #    continue


            size = Size3d(
                x=float(sx[i]),
                y=float(sy[i]),
                z=float(sz[i]),
            )

            position = Position3d(x=x, y=y, z=z)
            yaw = float(yaws[i])
            score = float(scores[i]) if scores is not None and i < len(scores) else 0.0


            obj3d = Object3d(
                position=position,
                size=size,
                yaw_angle=yaw,
                class_name=cname,
                category_confidence=score,
            )
            custom_3d_obj_list.append(obj3d)

        return Object3dList(time_stamp=ts_float, objects_3d=custom_3d_obj_list)

    def convert_yolo_ros_to_custom(self, ros_msg, ts_float):
        cam_data = {}
        for cam_obj in ros_msg.camera_2d_objects:
            camera_id = int(cam_obj.camera_id)
            custom_detections = []

            for box in cam_obj.boxes:
                cid = box.class_id
                cname = CLASS_LIST[cid] if 0 <= cid < len(CLASS_LIST) else "Unknown"
                #if cname in {"Adult", "Child"}:
                #    cname = "Pedestrian"

                custom_det = ImageDetection(
                    x=box.center_x,
                    y=box.center_y,
                    w=box.width,
                    h=box.height,
                    score=float(box.score),
                    class_id=int(cid),
                    class_name=cname,
                    tracker_id=None
                )
                custom_detections.append(custom_det)

            img_list = ImageDetectionList(
                camera_id=int(camera_id),
                time_stamp=ts_float,
                detections=custom_detections
            )
            cam_data[camera_id] = img_list
        return cam_data

    def publish_track_wireframes(self, tracks_out, frame_id="sys_world"):
        ma = MarkerArray()

        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        now = self.get_clock().now().to_msg()

        for idx, t in enumerate(tracks_out):
            st = getattr(t, "state", None)
            if st is None or len(st) < 7:
                continue
            hits = int(getattr(t, "hits", 0))
            tsu = int(getattr(t, "time_since_update", 999))

            #if self.viz_only_confirmed and hits < self.mot.min_hits:
                #continue
            if self.viz_only_recent and tsu >= 3:
                continue

            cx, cy, cz = float(st[0]), float(st[1]), float(st[2])
            yaw = float(st[3])
            l, w, h = float(st[4]), float(st[5]), float(st[6])
            if not (self.x_min <= cx <= self.x_max):
                continue
            if not (self.y_min <= cy <= self.y_max):
                continue

            # get class name from track if available
            cls_name = getattr(t, "class_name", "unknown")
            color = class_color(cls_name)

            corners = make_box_corners(cx, cy, cz, l, w, h, yaw)
            line_pts = box_edges_line_list(corners)

            m = Marker()
            m.header.frame_id = frame_id
            m.header.stamp = now
            m.ns = "track_wireframes"
            m.id = int(getattr(t, "track_id", idx))
            m.type = Marker.LINE_LIST
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = 0.1
            m.color = color
            #m.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)

            m.points = []
            for p in line_pts:
                pt = Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                m.points.append(pt)

            ma.markers.append(m)

            txt = Marker()
            txt.header.frame_id = frame_id
            txt.header.stamp = now
            txt.ns = "track_ids"
            txt.id = 100000 + m.id
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = cx
            txt.pose.position.y = cy
            txt.pose.position.z = cz + h/2.0 + 0.3
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.6
            txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            txt.text = f"ID:{getattr(t,'track_id', idx)}"
            ma.markers.append(txt)

        self.pub_markers.publish(ma)

    def _peek_nearest_image(self, img_queue: deque, t_ref: float, slop: float):
        while img_queue and img_queue[0][0] < t_ref:
            img_queue.popleft()
        if not img_queue:
            return None
        img_time, img_msg = img_queue[0]
        if img_time - t_ref <= slop:
            return img_msg
        return None

    def process_frame(self, ts_key):
        data = self.msg_buffer[ts_key]

        ouster_msg = data[TOPIC_OUSTER]
        dome_msg = data[TOPIC_DOME]
        tracks_msg = data[TOPIC_TRACKS]
        yolo_msg = data[TOPIC_YOLO]

        cam1_msg = data.get(TOPIC_CAM1, None)
        cam2_msg = data.get(TOPIC_CAM2, None)
        cam3_msg = data.get(TOPIC_CAM3, None)

        ts_float = ouster_msg.header.stamp.sec + ouster_msg.header.stamp.nanosec * 1e-9

        try:
            viz_time = ouster_msg.header.stamp
            if self.ENABLE_MERGE:
                final_cloud = self.merge_pointclouds(ouster_msg, dome_msg)
                final_cloud.header.frame_id = "sys_world"
                final_cloud.header.stamp = viz_time
                self.pub_merged_cloud.publish(final_cloud)

            custom_tracks = self.convert_3d_objects_ros_to_custom(tracks_msg, ts_float)
            custom_yolo_dict = self.convert_yolo_ros_to_custom(yolo_msg, ts_float)

            self.processed_frame_count += 1
            if self.processed_frame_count % 10 == 0:
                print(f"Processed Frame #{self.processed_frame_count} | TS: {ts_float:.3f}")

            fused_lidar, assoc_dbg, fusion_meta = self.fusion.fuse_object3d_list(custom_tracks, custom_yolo_dict)
            
            tracks_out = self.mot.update_two_stage(
                dets3d=list(fused_lidar.objects_3d),
                yolo_by_cam=custom_yolo_dict,
                assoc_debug=assoc_dbg,
                projector=self.projector,
                t=ts_float,
                fusion_meta=fusion_meta
            )
            
            #tracks_out = suppress_duplicate_tracks_3d(tracks_out, iou_thr=0.1)
            alive_ids = [t.track_id for t in tracks_out]
            tracks_display = choose_display_tracks_with_aliasing(
               tracks_out=tracks_out,
               group_parent=self._group_parent,
               iou_thr=0.8,           # tune dense scene 0.8, or 0.4
            )

            # publish only the display tracks
            self.publish_track_wireframes(tracks_display, frame_id="sys_world")

            #cleanup_group_parent(self._group_parent, [t.track_id for t in tracks_out])

            #self.publish_track_wireframes(tracks_out, frame_id="sys_world")

            if self.processed_frame_count % 20 == 0:
                cleanup_group_parent(self._group_parent, alive_ids)
                ids = [t.track_id for t in tracks_out[:10]]
                print(f"[MOT] num_tracks={len(tracks_out)} sample_ids={ids}")

            if self.enable_viz and (self.processed_frame_count % self.viz_every_n == 0):
                lidar_objs = custom_tracks.objects_3d

                def make_blank_panel(text, w=960, h=540):
                    panel = np.zeros((h, w, 3), dtype=np.uint8)
                    cv2.putText(panel, text, (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
                    return panel

                def prep_panel(cam_msg, cam_id):
                    if cam_msg is None:
                        return make_blank_panel(f"CAM {cam_id}: NO IMAGE")

                    img = self.bridge.imgmsg_to_cv2(cam_msg, desired_encoding="bgr8")

                    if (cam_id in custom_yolo_dict) and (cam_id in assoc_dbg):
                        img = draw_assoc_overlay(
                            img, cam_id, self.projector, lidar_objs,
                            custom_yolo_dict[cam_id], assoc_dbg[cam_id]
                        )
                    else:
                        cv2.putText(img, f"cam{cam_id} (no assoc)", (20, 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

                    img = draw_mot_tracks_overlay(img, cam_id, self.projector, tracks_out)
                    return img

                out1 = prep_panel(cam1_msg, 1)
                out2 = prep_panel(cam2_msg, 2)
                out3 = prep_panel(cam3_msg, 3)

                H = 540
                def resize_to_h(img):
                    return cv2.resize(img, (int(img.shape[1] * H / img.shape[0]), H))

                out1 = resize_to_h(out1)
                out2 = resize_to_h(out2)
                out3 = resize_to_h(out3)

                combined = np.hstack([out3, out2, out1])

                cv2.imshow(self.step_window_name, combined)

                # --- STEP MODE BEHAVIOR (NEW) ---
                if self.step_mode:
                    key = cv2.waitKey(0) & 0xFF  # wait for key press
                    if key == ord('q'):
                        raise KeyboardInterrupt
                else:
                    cv2.waitKey(1)

                if self.save_viz and (self.processed_frame_count % self.save_every_n == 0):
                    fname = f"frame_{self.processed_frame_count:06d}_t_{ts_float:.3f}.png"
                    fpath = os.path.join(self.save_dir, fname)
                    cv2.imwrite(fpath, combined)

        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.get_logger().error(f"Error processing frame: {e}")

    def run(self):
        reader1 = self.create_reader(
            BAG_1_PATH,
            [TOPIC_OUSTER, TOPIC_TRACKS, TOPIC_YOLO, TOPIC_CAM1, TOPIC_CAM2, TOPIC_CAM3]
        )
        reader2 = self.create_reader(BAG_2_PATH, [TOPIC_DOME])

        next_1 = self.get_next_msg(reader1)
        next_2 = self.get_next_msg(reader2)

        buffers = {
            TOPIC_OUSTER: deque(),
            TOPIC_DOME: deque(),
            TOPIC_TRACKS: deque(),
            TOPIC_YOLO: deque(),
            TOPIC_CAM1: deque(),
            TOPIC_CAM2: deque(),
            TOPIC_CAM3: deque(),
        }

        SYNC_SLOP = 0.09
        if self.ENABLE_MERGE:
            self.get_logger().info("Starting CUSTOM DATA merge...")

        while next_1 is not None or next_2 is not None:
            process_from_1 = False
            if next_1 is not None and next_2 is not None:
                if next_1[0] <= next_2[0]:
                    process_from_1 = True
            elif next_1 is not None:
                process_from_1 = True

            if process_from_1:
                ts, topic, msg = next_1
                next_1 = self.get_next_msg(reader1)
            else:
                ts, topic, msg = next_2
                next_2 = self.get_next_msg(reader2)

            ts_float = ts[0] + ts[1] * 1e-9
            buffers[topic].append((ts_float, msg))

            MAX_Q = {
                TOPIC_OUSTER: 30,
                TOPIC_DOME: 30,
                TOPIC_TRACKS: 200,
                TOPIC_YOLO: 200,
                TOPIC_CAM1: 30,
                TOPIC_CAM2: 30,
                TOPIC_CAM3: 30,
            }
            if len(buffers[topic]) > MAX_Q.get(topic, 50):
                buffers[topic].popleft()

            if (len(buffers[TOPIC_OUSTER]) > 0 and
                len(buffers[TOPIC_DOME]) > 0 and
                len(buffers[TOPIC_TRACKS]) > 0 and
                len(buffers[TOPIC_YOLO]) > 0):

                t_ouster = buffers[TOPIC_OUSTER][0][0]
                t_dome   = buffers[TOPIC_DOME][0][0]
                t_tracks = buffers[TOPIC_TRACKS][0][0]
                t_yolo   = buffers[TOPIC_YOLO][0][0]

                if t_dome < t_ouster - SYNC_SLOP:
                    buffers[TOPIC_DOME].popleft()
                    continue
                if t_tracks < t_ouster - SYNC_SLOP:
                    buffers[TOPIC_TRACKS].popleft()
                    continue
                if t_yolo < t_ouster - SYNC_SLOP:
                    buffers[TOPIC_YOLO].popleft()
                    continue

                if (t_dome > t_ouster + SYNC_SLOP) or \
                   (t_tracks > t_ouster + SYNC_SLOP) or \
                   (t_yolo > t_ouster + SYNC_SLOP):
                    buffers[TOPIC_OUSTER].popleft()
                    continue

                ouster_msg = buffers[TOPIC_OUSTER][0][1]
                ouster_stamp = ouster_msg.header.stamp
                ts_key = (ouster_stamp.sec, ouster_stamp.nanosec)

                cam1_msg = self._peek_nearest_image(buffers[TOPIC_CAM1], t_ouster, self.img_slop)
                cam2_msg = self._peek_nearest_image(buffers[TOPIC_CAM2], t_ouster, self.img_slop)
                cam3_msg = self._peek_nearest_image(buffers[TOPIC_CAM3], t_ouster, self.img_slop)

                self.msg_buffer[ts_key] = {
                    TOPIC_OUSTER: ouster_msg,
                    TOPIC_DOME:   buffers[TOPIC_DOME][0][1],
                    TOPIC_TRACKS: buffers[TOPIC_TRACKS][0][1],
                    TOPIC_YOLO:   buffers[TOPIC_YOLO][0][1],
                }

                if cam1_msg is not None:
                    self.msg_buffer[ts_key][TOPIC_CAM1] = cam1_msg
                if cam2_msg is not None:
                    self.msg_buffer[ts_key][TOPIC_CAM2] = cam2_msg
                if cam3_msg is not None:
                    self.msg_buffer[ts_key][TOPIC_CAM3] = cam3_msg

                self.process_frame(ts_key)
                del self.msg_buffer[ts_key]

                buffers[TOPIC_OUSTER].popleft()
                buffers[TOPIC_DOME].popleft()
                buffers[TOPIC_TRACKS].popleft()
                buffers[TOPIC_YOLO].popleft()

        self.get_logger().info(f"Finished. Total Frames Processed: {self.processed_frame_count}")


def main(args=None):
    rclpy.init(args=args)
    node = DualBagProcessor()

    exec_ = SingleThreadedExecutor()
    exec_.add_node(node)

    worker = threading.Thread(target=node.run, daemon=True)
    worker.start()

    try:
        exec_.spin()   # <-- REQUIRED for publishers to actually publish
    except KeyboardInterrupt:
        pass
    finally:
        exec_.shutdown()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()