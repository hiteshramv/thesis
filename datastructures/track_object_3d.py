import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datastructures.lidar_detections import Object3d
from typing import List


class TrackObject3d:
    def __init__(self, track_id: int, object_3d: Object3d) -> None:
        self.track_id = track_id
        self.object_3d = object_3d
        
    @property
    def track_id(self):
        return self._track_id
    
    @track_id.setter
    def track_id(self, value):
        if isinstance(value, int) and value > 0:
            self._track_id = value
        else:
            raise TypeError("Track ID must be positive integer value")  
        
    @property
    def object_3d(self):
        return self._object_3d
    
    @object_3d.setter
    def object_3d(self, value):
        if not isinstance(value, Object3d):
            raise TypeError("Invalid type of 3D object assigned for track object generation.")
        self._object_3d = value
        
    def __str__(self):
        out = f'Track ID = {self.track_id} \n'
        out += str(self.object_3d)
        return out 
        

class TrackObject3dList:
    def __init__(self, epoch_time, 
                 frame_id: int = -1,
                 track_objects_3d: List[TrackObject3d] = []):
        
        self.epoch_time = epoch_time
        self.frame_id = frame_id
        self.track_objects_3d = track_objects_3d
        self._total_track_objects = 0 # this is computed property
        
    @property # getter (no setter required for this computed property)
    def total_objects(self):
       return len(self.track_objects_3d)
   
    def __len__(self):
        return self.total_objects
    
    @property # getter
    def frame_id(self):
        return self._frame_id
     
    @frame_id.setter # setter
    def frame_id(self, value):
        if isinstance(value, int):
            self._frame_id = value

    @property # getter
    def track_objects_3d(self):
        return self._track_objects_3d
    
    def __iter__(self):
        for obj in self.track_objects_3d:
            yield obj
    
    @track_objects_3d.setter
    def track_objects_3d(self, value):
        if isinstance(value, List):
            if len(value) == 0: # if list is empty
                self._track_objects_3d = value
            else: # if list is not empty
                for obj in value:
                    if not isinstance(obj, (TrackObject3d, type(None))): # check if all instances in list are of type Object3d
                        raise TypeError(f'List must contain only objects of TrackObject3d in {__class__.__name__}. But object of type {type(obj)} is given.')
                self._track_objects_3d = value
        else:
            raise TypeError(f'Value of 3D track objects must be given as list in {__class__.__name__}. But of type {type(value)} is given.')
        
    @property
    def epoch_time(self):
        return self._epoch_time
    
    @epoch_time.setter
    def epoch_time(self, value):
        self._epoch_time = value

    def __str__(self):
        str_track_object_list = ""
        for index, obj in enumerate(self.track_objects_3d):
            str_track_object_list += str(index+1) + ". " + str(obj) +'\n'
        return(
            f'-------------------------------------\n'
            f'{__class__.__name__}: epoch time = {self.epoch_time}, frame id = {self.frame_id}, total track objects = {self.total_objects}\n'
            f'{str_track_object_list}'
            f'-------------------------------------\n' 
        )
