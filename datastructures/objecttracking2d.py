import os
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from message_filters import Subscriber
from message_filters import ApproximateTimeSynchronizer

from final_calibration_dict import calibration_data  # your calibration file

CAM_TOPICS = {
    1: "/camera1/pylon_ros2_camera_node_camera1/image_raw",
    2: "/camera2/pylon_ros2_camera_node_camera2/image_raw",
    3: "/camera3/pylon_ros2_camera_node_camera3/image_raw",
}

CLASS_NAMES = ["background","adult","child","bicycle","motorcycle","car","van","truck","bus"]
CONF = 0.35

# 🟩 class color mapping
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


def K_and_Rt(camera_key):
    K = np.array(calibration_data["cameras"][camera_key]["intrinsic"], dtype=np.float64)
    Rt = np.array(calibration_data["cameras"][camera_key]["extrinsic"], dtype=np.float64)
    R = Rt[:, :3]
    t = Rt[:, 3:4]
    return K, R, t


def homography_ground_from_KRt(K, R, t):
    H = K @ np.hstack([R[:, :2], t])  # 3x3
    return H


def img_to_ground_uv(u, v, H_inv):
    pix = np.array([u, v, 1.0], dtype=np.float64)
    XYw = H_inv @ pix
    XYw /= XYw[2] + 1e-9
    return float(XYw[0]), float(XYw[1])


class GlobalTracker:
    #Tiny ID manager over ground plane points.
    def __init__(self, max_dist=1.5):
        self.next_id = 1
        self.tracks = {}  # id -> (x,y, age, ttl)
        self.max_dist = max_dist

    def update(self, pts):
        assigned = {}
        ids_out = []
        for x, y, cls, cam, bbox in pts:
            best_id, best_d = None, 1e9
            for tid, (tx, ty, age, ttl) in self.tracks.items():
                d = (tx - x)**2 + (ty - y)**2
                if d < best_d:
                    best_d, best_id = d, tid
            if best_d**0.5 < self.max_dist:
                tx, ty, age, ttl = self.tracks[best_id]
                self.tracks[best_id] = (x, y, age + 1, 8)
                assigned[best_id] = True
                ids_out.append((best_id, x, y, cls, cam, bbox))
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = (x, y, 1, 8)
                assigned[tid] = True
                ids_out.append((tid, x, y, cls, cam, bbox))
        # decay unassigned
        for tid in list(self.tracks.keys()):
            if tid not in assigned:
                tx, ty, age, ttl = self.tracks[tid]
                ttl -= 1
                if ttl <= 0:
                    del self.tracks[tid]
                else:
                    self.tracks[tid] = (tx, ty, age, ttl)
        return ids_out


class MultiCamNode(Node):
    def __init__(self):
        super().__init__("multicam_tracker")
        self.bridge = CvBridge()
        self.model = YOLO("yolo11s_inani.engine")

        # Output directory for stacked frames
        self.output_dir = os.path.expanduser("~/multicam_outputs")
        os.makedirs(self.output_dir, exist_ok=True)
        self.frame_idx = 0

        # Precompute homographies and inverses
        self.H_inv = {}
        for cam_key in ["camera_01", "camera_02", "camera_03"]:
            K, R, t = K_and_Rt(cam_key)
            H = homography_ground_from_KRt(K, R, t)
            self.H_inv[cam_key] = np.linalg.inv(H)

        s1 = Subscriber(self, Image, CAM_TOPICS[1])
        s2 = Subscriber(self, Image, CAM_TOPICS[2])
        s3 = Subscriber(self, Image, CAM_TOPICS[3])

        self.ats = ApproximateTimeSynchronizer([s1, s2, s3], queue_size=5, slop=0.09)
        self.ats.registerCallback(self.callback)

        self.tracker = GlobalTracker(max_dist=1.5)
        self.get_logger().info("MultiCam Tracker started.")

    def callback(self, msg1, msg2, msg3):
        cam_keys = ["camera_01", "camera_02", "camera_03"]
        frames_dict = {
            "camera_01": self.bridge.imgmsg_to_cv2(msg1, desired_encoding="bgr8"),
            "camera_02": self.bridge.imgmsg_to_cv2(msg2, desired_encoding="bgr8"),
            "camera_03": self.bridge.imgmsg_to_cv2(msg3, desired_encoding="bgr8"),
        }
        frames_list = [frames_dict[k] for k in cam_keys]

        # Batch inference
        results = self.model.predict(frames_list, conf=CONF, verbose=False)
        all_measurements = []
        annotated = {}

        for cam_key, result, img in zip(cam_keys, results, frames_list):
            Hin = self.H_inv[cam_key]
            anno = img.copy()

            for b in result.boxes:
                cls_id = int(b.cls[0].item())
                cls_name = result.names.get(cls_id, str(cls_id))
                if CLASS_NAMES and cls_name not in CLASS_NAMES:
                    continue
                color = CLASS_COLORS.get(cls_name, (255, 255, 255))

                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                u = 0.5 * (x1 + x2)
                v = y2
                X, Y = img_to_ground_uv(u, v, Hin)
                all_measurements.append((X, Y, cls_name, cam_key, (x1, y1, x2, y2)))

                # Draw class-colored box + label
                cv2.rectangle(anno, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(anno, cls_name, (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.circle(anno, (int(u), int(v)), 3, (0, 255, 255), -1)
            annotated[cam_key] = anno

        fused = self.tracker.update(all_measurements)

        # Draw global IDs
        for tid, X, Y, cls_name, cam_key, (x1, y1, x2, y2) in fused:
            color = CLASS_COLORS.get(cls_name, (255, 255, 255))
            img = annotated[cam_key]
            cv2.putText(img, f"ID {tid} {cls_name}",
                        (int(x1), int(y1) - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Stack all 3 annotated frames horizontally
        target_h = 480
        resized = []
        for cam_key in cam_keys:
            img = annotated[cam_key]
            scale = target_h / img.shape[0]
            new_w = int(img.shape[1] * scale)
            resized.append(cv2.resize(img, (new_w, target_h)))
        combined = np.hstack(resized)

        # Save stacked frame (with ROS timestamp)
        stamp = msg1.header.stamp.sec * 1e9 + msg1.header.stamp.nanosec
        filename = os.path.join(self.output_dir, f"stack_{int(stamp)}.jpg")
        cv2.imwrite(filename, combined)

        # Display live
        cv2.imshow("Multi-Camera View", combined)
        cv2.waitKey(1)
        self.frame_idx += 1


def main():
    rclpy.init()
    node = MultiCamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()





"""
# pip install ultralytics filterpy scipy
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from message_filters import Subscriber
from message_filters import ApproximateTimeSynchronizer

from final_calibration_dict import calibration_data  # your file
# ^ contains K (intrinsic) and [R|t] (extrinsic) per camera, and ouster_to_ground=I. :contentReference[oaicite:1]{index=1}

CAM_TOPICS = {
    1: "/camera1/pylon_ros2_camera_node_camera1/image_raw",
    2: "/camera2/pylon_ros2_camera_node_camera2/image_raw",
    3: "/camera3/pylon_ros2_camera_node_camera3/image_raw",
}

CLASS_NAMES = ["background","adult","child","bicycle","motorcycle","car","van","truck","bus"]  # trim to what you need
CONF = 0.35

def K_and_Rt(camera_key):
    K = np.array(calibration_data["cameras"][camera_key]["intrinsic"], dtype=np.float64)
    Rt = np.array(calibration_data["cameras"][camera_key]["extrinsic"], dtype=np.float64) # 3x4, world=ouster
    R = Rt[:, :3]
    t = Rt[:, 3:4]
    return K, R, t

def homography_ground_from_KRt(K, R, t):
    # World (ground) plane is Z=0 in ouster frame (ouster_to_ground is identity) :contentReference[oaicite:2]{index=2}
    # For a point (X,Y,0,1)^T, image u ~ K [r1 r2 t] [X Y 1]^T
    H = K @ np.hstack([R[:, :2], t])  # 3x3
    return H

def img_to_ground_uv(u, v, H_inv):
    pix = np.array([u, v, 1.0], dtype=np.float64)
    XYw = H_inv @ pix
    XYw /= XYw[2] + 1e-9
    return float(XYw[0]), float(XYw[1])

class GlobalTracker:
    #Tiny ID manager over ground plane points.
    def __init__(self, max_dist=1.5):
        self.next_id = 1
        self.tracks = {}  # id -> (x,y, age, ttl)
        self.max_dist = max_dist

    def update(self, pts):  # pts: [(x,y, cls, cam, bbox)]
        # Greedy NN match to existing tracks
        assigned = {}
        ids_out = []
        for x,y,cls,cam,bbox in pts:
            best_id, best_d = None, 1e9
            for tid,(tx,ty,age,ttl) in self.tracks.items():
                d = (tx - x)**2 + (ty - y)**2
                if d < best_d:
                    best_d, best_id = d, tid
            if best_d**0.5 < self.max_dist:
                # update existing
                tx,ty,age,ttl = self.tracks[best_id]
                self.tracks[best_id] = (x,y, age+1, 8)
                assigned[best_id] = True
                ids_out.append((best_id, x,y,cls,cam,bbox))
            else:
                # new track
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = (x,y, 1, 8)
                assigned[tid] = True
                ids_out.append((tid, x,y,cls,cam,bbox))
        # decay unassigned
        for tid in list(self.tracks.keys()):
            if tid not in assigned:
                tx,ty,age,ttl = self.tracks[tid]
                ttl -= 1
                if ttl <= 0: del self.tracks[tid]
                else: self.tracks[tid] = (tx,ty,age,ttl)
        return ids_out

class MultiCamNode(Node):
    def __init__(self):
        super().__init__("multicam_tracker")
        self.bridge = CvBridge()
        self.model = YOLO("yolo11s_inani.engine")  # or n.pt for speed

        # Precompute homographies and inverses
        self.H_inv = {}
        for cam_key in ["camera_01","camera_02","camera_03"]:
            K,R,t = K_and_Rt(cam_key)
            H = homography_ground_from_KRt(K,R,t)
            self.H_inv[cam_key] = np.linalg.inv(H)

        s1 = Subscriber(self, Image, CAM_TOPICS[1])
        s2 = Subscriber(self, Image, CAM_TOPICS[2])
        s3 = Subscriber(self, Image, CAM_TOPICS[3])

        self.ats = ApproximateTimeSynchronizer([s1,s2,s3], queue_size=5, slop=0.09)  # ~50ms
        self.ats.registerCallback(self.callback)

        self.tracker = GlobalTracker(max_dist=1.5)  # meters on ground

    def detect(self, img):
        r = self.model.predict(img, conf=CONF, verbose=False)[0]
        dets = []
        for b in r.boxes:
            cls = int(b.cls[0].item())
            if CLASS_NAMES and r.names[cls] not in CLASS_NAMES: continue
            x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
            dets.append((x1,y1,x2,y2, r.names[cls]))
        return dets

    def callback(self, msg1, msg2, msg3):
        frames = {
            "camera_01": self.bridge.imgmsg_to_cv2(msg1, desired_encoding="bgr8"),
            "camera_02": self.bridge.imgmsg_to_cv2(msg2, desired_encoding="bgr8"),
            "camera_03": self.bridge.imgmsg_to_cv2(msg3, desired_encoding="bgr8"),
        }
        all_measurements = []
        annotated = {}

        for cam_key, img in frames.items():
            dets = self.detect(img)
            Hin = self.H_inv[cam_key]
            anno = img.copy()

            for (x1,y1,x2,y2,cls_name) in dets:
                u = 0.5*(x1+x2); v = y2  # footpoint
                X,Y = img_to_ground_uv(u, v, Hin)  # meters in ground frame
                all_measurements.append((X,Y, cls_name, cam_key, (x1,y1,x2,y2)))
                # temp draw (ID filled after association)
                cv2.rectangle(anno, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
                cv2.circle(anno, (int(u),int(v)), 3, (0,255,255), -1)
            annotated[cam_key] = anno

        fused = self.tracker.update(all_measurements)

        # draw IDs
        for tid, X,Y, cls_name, cam_key, (x1,y1,x2,y2) in fused:
            img = annotated[cam_key]
            cv2.putText(img, f"ID {tid} {cls_name}", (int(x1), int(y1)-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,50,0), 2)

        # show (or publish)
        target_h = 480
        resized = []
        for cam_key in ["camera_01", "camera_02", "camera_03"]:
            img = annotated[cam_key]
            scale = target_h / img.shape[0]
            new_w = int(img.shape[1] * scale)
            resized.append(cv2.resize(img, (new_w, target_h)))

        # Combine horizontally
        combined = np.hstack(resized)

        # Show in single window
        cv2.imshow("Multi-Camera View", combined)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = MultiCamNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
"""