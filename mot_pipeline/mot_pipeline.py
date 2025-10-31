#ros2 bag play "/home/vh17r/thesis/inani_ros2_bag/inani__2025-07-07_07-26-43-168/inani__2025-07-07_07-26-43-168__bag1" --clock
#ros2 bag play "/home/vh17r/thesis/inani_ros2_bag/inani__2025-07-07_07-33-03-367/inani__2025-07-07_07-33-03-367__bag1" --clock
#ros2 bag play "/home/vh17r/thesis/inani_ros2_bag/inani__2025-07-07_07-37-36-291/inani__2025-07-07_07-37-36-291__bag1" --clock
#ros2 bag play "/home/vh17r/thesis/inani_ros2_bag/inani__2025-07-07_07-42-54-306/inani__2025-07-07_07-42-54-306__bag1" --clock

#ros2 bag play "/home/vh17r/ros2_yolo/inani/inani__2025-10-17_07-30-27-642__bag1" --clock


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import rclpy
from dataloading import SensorDataLoader
from detector2D import Detector2D
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
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            start_time = time.time()
            images = node.get_latest_images()

            # Log image info for all cameras in the set
            for i, img in enumerate(images):
                if img is not None:
                    logger.info(f"Camera: {img.camera_id}, Timestamp: {img.time_stamp}")
                #else:
                    #logger.warning(f"Image {i+1} is None.")

            if all(img is not None for img in images):
                detections = detector.detect(images)
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
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import rclpy
from dataloading import SensorDataLoader
from detector2D import Detector2D
import visualization
import time

def main(args=None):
    rclpy.init(args=args)
    node = SensorDataLoader()
    detector = Detector2D(model_path='yolo11s.pt', conf=0.3)
    try:
        while rclpy.ok():
            # Spin node to handle ROS callbacks and update latest images
            rclpy.spin_once(node, timeout_sec=0.1)
            start_time = time.time()
            images = node.get_latest_images()
            print(images)
            # Ensure all 3 images are present
            if all(img is not None for img in images):
                detections = detector.detect(images)
                elapsed = time.time() - start_time
                print(f"Processing time for this set of 3 images: {elapsed*1000:.1f} ms")
                visualization.show(images, detections)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

########################################################################################################
# just an example not final 


import rclpy
from sensordataloader import SensorDataLoader
from detector2d import Detector2D  # Or import YOLO directly
# from detector3d import Detector3D
# from multiobject_tracker import MultiObjectTracker

class MOTPipeline:
    def __init__(self):
        self.loader = SensorDataLoader()
        self.detector = Detector2D()  # Or just use YOLO directly
        # self.tracker = MultiObjectTracker()
        # ... more initializations ...

    def run(self):
        rclpy.init()
        try:
            while rclpy.ok():
                rclpy.spin_once(self.loader, timeout_sec=0.1)
                images, timestamps = self.loader.get_latest_images(wait=True, timeout=2.0)
                if images and all(images):
                    detections = []
                    for idx, img in enumerate(images):
                        dets = self.detector(img)  # or self.detector.infer(img)
                        detections.append(dets)
                    # self.tracker.update(detections, timestamps)
                    # ... do association, write results, etc ...
        finally:
            self.loader.destroy_node()
            rclpy.shutdown()

if __name__ == "__main__":
    pipeline = MOTPipeline()
    pipeline.run()
"""