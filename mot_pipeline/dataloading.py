from collections import deque
from typing import Any, Dict, List, Optional
import rclpy
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nse_ros_interfaces.msg import NSEObjectList
#from message_filters import ApproximateTimeSynchronizer
from message_filters import Subscriber
from message_filters_ATS import ApproximateTimeSynchronizer
import cv2
from cv_bridge import CvBridge
from rclpy.time import Time as RclpyTime
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datastructures.image import ImageData
from datastructures.lidar_detections import (Object3d, Object3dList,Position3d, Size3d, Orientation3d, Speed3d)
from datastructures.track_object_3d import TrackObject3d, TrackObject3dList

LiDAR_DETECTIONS_TOPIC = "/object_list/tracks"
CAM_TOPICS = {
    1: "/camera1/pylon_ros2_camera_node_camera1/image_raw",
    2: "/camera2/pylon_ros2_camera_node_camera2/image_raw",
    3: "/camera3/pylon_ros2_camera_node_camera3/image_raw",
}

LOCAL_TZ = ZoneInfo("Europe/Berlin")
DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"



def _get(obj, path: str, default=None):
    """Safely pull nested attributes from ROS messages (e.g., 'bounding_box.position.x')."""
    cur = obj
    for p in path.split("."):
        if cur is None:
            return default
        cur = getattr(cur, p, None)
    return default if cur is None else cur

def to_datetime_utc(stamp) -> datetime:
    t = RclpyTime.from_msg(stamp)
    ns = t.nanoseconds
    sec, nsec = divmod(ns, 1_000_000_000)
    return datetime.fromtimestamp(sec + nsec / 1e9, tz=timezone.utc).astimezone(LOCAL_TZ).strftime(DATETIME_FORMAT)[:-3]

class SensorDataLoader(Node):
    def __init__(self):
        super().__init__('sensor_data_loader')
        assert len(CAM_TOPICS) == 3, "Expected 3 camera topics"
        self.lidar_detections_sub = Subscriber(self, NSEObjectList, LiDAR_DETECTIONS_TOPIC)
        self.cam1_sub = Subscriber(self, Image, CAM_TOPICS[1])
        self.cam2_sub = Subscriber(self, Image, CAM_TOPICS[2])
        self.cam3_sub = Subscriber(self, Image, CAM_TOPICS[3])
        self.ts = ApproximateTimeSynchronizer(
            [self.lidar_detections_sub,self.cam1_sub, self.cam2_sub, self.cam3_sub],
            #[self.cam1_sub, self.cam2_sub, self.cam3_sub],
            queue_size=20,
            slop=0.08,
            queue_offset=[0, 31000000, 51000000, 71000000],
            allow_headerless=False,
            primary_index=0
            )
        self.ts.registerCallback(self.synced_callback)
        self.bridge = CvBridge()
        self.image_queue = deque()   # <-- use queue for sets

        self.last_lidar_objlist: Optional[Object3dList] = None
        self.last_track_list = None          # TrackObject3dList or List[TrackObject3d]
        self.last_tracks_by_id: Dict[int, Any] = {}

    def synced_callback(self, lidar_detections: NSEObjectList, img1: Image, img2: Image, img3: Image):

        obj_list, track_list, frame_id = self._nse_to_object3d_and_tracks(lidar_detections, to_datetime_utc(lidar_detections.header.stamp))
        self.last_lidar_objlist = obj_list
        self.last_track_list = track_list

        img_bgr1 = self.bridge.imgmsg_to_cv2(img1, desired_encoding="bgr8")
        img_bgr2 = self.bridge.imgmsg_to_cv2(img2, desired_encoding="bgr8")
        img_bgr3 = self.bridge.imgmsg_to_cv2(img3, desired_encoding="bgr8")

        image1 = ImageData(1, to_datetime_utc(img1.header.stamp), img_bgr1)
        image2 = ImageData(2, to_datetime_utc(img2.header.stamp), img_bgr2)
        image3 = ImageData(3, to_datetime_utc(img3.header.stamp), img_bgr3)
        
        self.image_queue.append([image1, image2, image3]) 
    """
        if self.last_track_list:
            try:
                debug_path = "lidar_track_log.txt"  # change path if you want
                with open(debug_path, "a") as f:
                    f.write("\n=====================================\n")
                    f.write(f"Frame ID: {frame_id}, Time: {to_datetime_utc(lidar_detections.header.stamp)}\n")
                    f.write(str(self.last_track_list))  # uses your __str__ formatting
                    f.write("\n=====================================\n")
                self.get_logger().info(
                    f"Track list (frame {frame_id}) written to {debug_path} "
                    f"({len(getattr(self.last_track_list, 'track_objects_3d', []))} tracks)"
                )
            except Exception as e:
                self.get_logger().warn(f"Failed to log track list: {e}")
    """



    def get_latest_images(self):
        if self.image_queue:
            return self.image_queue.popleft()
        else:
            return [None, None, None]

    def _make_track_object3d(self, track_id: int, obj3d: Object3d):
        if TrackObject3d is None:
            return None
        try:
            return TrackObject3d(track_id, obj3d)  # matches your signature exactly
        except Exception:
            return None

    def _make_track_list(self, epoch_ts_obj: Any, frame_id: int, tracks: List[Any]):
        """
        Build TrackObject3dList(epoch_time, frame_id, track_objects_3d)
        Falls back to returning the raw 'tracks' list if TimeStamp or class is missing.
        """
        if TrackObject3dList is None or epoch_ts_obj is None:
            return tracks if tracks else None
        try:
            return TrackObject3dList(epoch_time=epoch_ts_obj, frame_id=int(frame_id), track_objects_3d=tracks)
        except Exception:
            # try positional order as defined in your class: (epoch_time, frame_id, track_objects_3d)
            try:
                return TrackObject3dList(epoch_ts_obj, int(frame_id), tracks)
            except Exception:
                return tracks if tracks else None
    # ---------------------------------------------------

    def _nse_to_object3d_and_tracks(self, nse_list_msg, time_stamp_str):
        """
        IMPORTANT mapping (as requested):
          - bbox.yaw → Object3d.yaw_angle
          - twist.angular.z → Orientation3d.yaw  (yaw-rate)
        """
        objects: List[Object3d] = []
        tracks_raw: List[Any] = []

        # epoch TimeStamp object for TrackObject3dList
        epoch_ts_obj = time_stamp_str

        # top-level frame_id (your sample shows an INT frame_id: 5587)
        frame_id = int(getattr(nse_list_msg, "frame_id", -1))

        for o in getattr(nse_list_msg, "objects", []):
            # Geometry
            px = _get(o, "bounding_box.position.x", 0.0)
            py = _get(o, "bounding_box.position.y", 0.0)
            pz = _get(o, "bounding_box.position.z", 0.0)
            sx = _get(o, "bounding_box.scale.x", 0.0)
            sy = _get(o, "bounding_box.scale.y", 0.0)
            sz = _get(o, "bounding_box.scale.z", 0.0)
            yaw_angle = float(_get(o, "bounding_box.yaw", 0.0))  # bbox yaw -> Object3d.yaw_angle

            position = Position3d(px, py, pz)
            size = Size3d(sx, sy, sz)

            # Kinematics
            vx = _get(o, "twist.linear.x", 0.0)
            vy = _get(o, "twist.linear.y", 0.0)
            vz = _get(o, "twist.linear.z", 0.0)
            yaw_rate = float(_get(o, "twist.angular.z", 0.0))     # yaw-rate -> Orientation3d.yaw
            orientation = Orientation3d(0.0, 0.0, yaw_rate)
            speed = Speed3d(vx, vy, vz)

            # Class & score
            class_name = _get(o, "class_name.data", "unknown")
            #class_id = -1
            category_conf = float(_get(o, "score", 0.0))
            existence_conf = 0.0

            track_id = _get(o, "track_id", None)

            obj = Object3d(
                position=position,
                size=size,
                orientation=orientation,
                speed=speed,
                yaw_angle=yaw_angle,
                class_name=class_name,
                #class_id=class_id,
                category_confidence=category_conf,
                existence_confidence=existence_conf,
                #track_id=track_id
            )
            objects.append(obj)

            if TrackObject3d is not None and track_id is not None:
                trk = self._make_track_object3d(int(track_id), obj)
                if trk is not None:
                    tracks_raw.append(trk)

        obj_list = Object3dList(time_stamp=time_stamp_str, objects_3d=objects)
        track_list = self._make_track_list(epoch_ts_obj, frame_id, tracks_raw)

        return obj_list, track_list, frame_id

    def get_latest_lidar(self) -> Optional[Object3dList]:
        return self.last_lidar_objlist

    def get_latest_tracks(self):
        """TrackObject3dList if available, else List[TrackObject3d], else None."""
        return self.last_track_list

    def get_track_by_id(self, track_id: int):
        return self.last_tracks_by_id.get(int(track_id))


