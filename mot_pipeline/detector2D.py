import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ultralytics import YOLO
from datastructures.image_detection import ImageDetection, ImageDetectionList
from datastructures.image import ImageData
import cv2

class Detector2D:
    def __init__(self, model_path='yolo11s.engine', conf=0.55):
        self.model = YOLO(model_path, task = 'detect')
        #self.model.to('cuda')
        self.conf = conf

    def detect(self, images):
        """
        Args:
            images: list of ImageData objects (each with .data as np.ndarray)
        Returns:
            list of ImageDetectionList objects
        """
        all_detections = []
        imgs_np = [img_obj.data for img_obj in images]
        #results = [self.model.track([img_obj.data], conf=self.conf, half=True) for img_obj in images]
        results = self.model.predict(imgs_np, conf=self.conf, half=True)
        for i, (img_obj, res) in enumerate(zip(images, results)):
            det_list = []
            if hasattr(res, 'boxes') and res.boxes is not None and len(res.boxes) > 0:
                for box in res.boxes:
                    x = float(box.xywh[0][0])
                    y = float(box.xywh[0][1])
                    w = float(box.xywh[0][2])
                    h = float(box.xywh[0][3])
                    score = float(box.conf[0]) if hasattr(box, 'conf') else float(box.confidence)
                    class_id = int(box.cls[0]) if hasattr(box, 'cls') else int(box.class_id)
                    class_name = self.model.names[class_id] if hasattr(self.model, 'names') else None
                    #tracker_id = box.id[0]
                    det = ImageDetection(
                        x=x,
                        y=y,
                        w=w,
                        h=h,
                        score=score,
                        class_id=class_id,
                        class_name=class_name,
                        #tracker_id = tracker_id
                    )
                    det_list.append(det)
            detection_list = ImageDetectionList(
                camera_id=img_obj.camera_id,
                time_stamp=img_obj.time_stamp,
                detections=det_list
            )
            all_detections.append(detection_list)
        return all_detections
