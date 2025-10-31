from typing import List

class ImageDetection:
    def __init__(self, x,y,w,h,score,class_id,class_name,tracker_id=None):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.score = score
        self.class_id = class_id
        self.class_name = class_name
        self.tracker_id = tracker_id

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value    
    
    @property
    def w(self):
        return self._w
    
    @w.setter
    def w(self, value):
        self._w = value

    @property
    def h(self):
        return self._h
    
    @h.setter
    def h(self, value):
        self._h = value

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        self._score = value

    @property
    def class_id(self):
        return self._class_id

    @class_id.setter
    def class_id(self, value):
        self._class_id = value

    @property
    def class_name(self):
        return self._class_name

    @class_name.setter
    def class_name(self, value):
        self._class_name = value

    @property
    def tracker_id(self):
        return self._tracker_id

    @tracker_id.setter
    def tracker_id(self, value):
        self._tracker_id = value

    def __str__(self) -> str:
        out = f'{__class__.__name__} - [x = {self.x}, y = {self.y}, w = {self.w}, h = {self.h}, score = {self.score}, Class ID = {self.class_id}, Class Name = {self.class_name} ]'
        return out



class ImageDetectionList:
    #Can rosbag time stamp be used here??

    def __init__(self,camera_id, time_stamp, detections: List[ImageDetection] = None):
        self.camera_id = camera_id
        self.time_stamp = time_stamp
        self.detections = detections if detections is not None else []

    @property
    def camera_id(self):
        return self._camera_id
    
    @camera_id.setter
    def camera_id(self, value):
        # validate frame id
        if not (isinstance(value, int) and 1 <= value <= 3):
            raise ValueError("Camera id must be integer 1, 2, or 3.")
        self._camera_id = value
    
    @property
    def time_stamp(self):
        return self._time_stamp
    
    @time_stamp.setter
    def time_stamp(self, value):
        self._time_stamp = value
    
    @property
    def detections(self):
        return self._detections

    def __iter__(self):
        for obj in self._detections:
            yield obj

    def __getitem__(self, idx):
        return self.detections[idx]

    @detections.setter
    def detections(self, value):
        if isinstance(value, list):
            if len(value) == 0: # if list is empty
                self._detections = value
            else: # if list is not empty
                for obj in value:
                    if not isinstance(obj, ImageDetection): # check if all instances in list are of type ImageDetection
                        raise TypeError(f'List must contain only objects of ImageDetection in {__class__.__name__}. But object of type {type(obj)} is given.')
                self._detections = value
        else:
            raise TypeError(f'Value of image objects 2d must be given as list in {__class__.__name__}. But of type {type(value)} is given.')

    def __str__(self):
        str_object_list = ""
        for index, obj in enumerate(self.detections):
            str_object_list += str(index+1) + ". " + str(obj) +'\n'
        
        return(
            f'-------------------------------------\n'

            f'{__class__.__name__}: camera id = {self.camera_id,} time stamp = {self.time_stamp}\n'
            f'{str_object_list}'
            f'-------------------------------------\n' 
        )