from enum import Enum
import warnings
import time
from typing import List

class Position3d:
    def __init__(self, 
                 x : float = 0.0,
                 y : float = 0.0,
                 z : float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z
    
    @property # getter
    def x(self):
        return self._x
    
    @x.setter # setter
    def x(self, value):
        if not isinstance(value, float):
            if isinstance(value, int):
                self._x = float(value)
            else:
                raise TypeError(f'Value of x of {__class__.__name__} can be int or float. But x = {value} of type {type(value)} is given.')
        else:
            self._x = value
    
    @property # getter
    def y(self):
        return self._y
    
    @y.setter # setter
    def y(self, value):
        if not isinstance(value, float):
            if isinstance(value, int):
                self._y = float(value)
            else:
                raise TypeError(f'Value of y of {__class__.__name__} can be int or float. But y = {value} of type {type(value)} is given.')
        else:
            self._y = value
    
    @property # getter
    def z(self):
        return self._z
    
    @z.setter # setter
    def z(self, value):
        if not isinstance(value, float):
            if isinstance(value, int):
                self._z = float(value)
            else:
                raise TypeError(f'Value of z of {__class__.__name__} can be int or float. But z = {value} of type {type(value)} is given.')
        else:
            self._z = value
    
    def __str__(self):
        out = f'3D position - [x = {self.x}, y = {self.y}, z = {self.z}] meters'
        return out


class Size3d:
    def __init__(self, 
                 x: float = 0.0,
                 y: float = 0.0,
                 z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z
        
    @property # getter
    def x(self):
        return self._x
    
    @x.setter # setter
    def x(self, value):
        if isinstance(value, float) or isinstance(value, int):
            if value >= 0:
                self._x = float(value)
            else:
                raise ValueError(f'Value of x of object in {__class__.__name__} cannot be negative. But x = {value} is given.')
        else:
            raise TypeError(f'Value of x of object in {__class__.__name__} can be int or float. But x = {value} of type {type(value)} is given.')


    @property # getter
    def y(self):
        return self._y
    
    @y.setter # setter
    def y(self, value):
        if isinstance(value, float) or isinstance(value, int):
            if value >= 0:
                self._y = float(value)
            else:
                raise ValueError(f'Value of y of object in {__class__.__name__} cannot be negative. But y = {value} is given.')
        else:
            raise TypeError(f'Value of y of object in {__class__.__name__} can be int or float. But y = {value} of type {type(value)} is given.')
    
    
    @property # getter
    def z(self):
        return self._z
    
    @z.setter # setter
    def z(self, value):
        if isinstance(value, float) or isinstance(value, int):
            if value >= 0:
                self._z = float(value)
            else:
                raise ValueError(f'Value of z of object in {__class__.__name__} cannot be negative. But z = {value} is given.')
        else:
            raise TypeError(f'Value of z of object in {__class__.__name__} can be int or float. But z = {value} of type {type(value)} is given.')
        
    def __str__(self):
        out = f'3D size - [x = {self.x}, y = {self.y}, z = {self.z}] meters'
        return out

class Orientation3d:
    def __init__(self,
                 roll: float = 0.0,
                 pitch: float = 0.0,
                 yaw: float = 0.0) -> None:
        
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        
    @property # getter
    def roll(self):
        return self._roll
    
    @roll.setter # setter
    def roll(self, value):
        if isinstance(value, float) or isinstance(value, int):
            if value >= 0 and value <=180:
                self._roll = float(value)
            else:
                raise ValueError(f'Value of roll angle of object in {__class__.__name__} can be between [0,180] degrees. But roll angle = {value} is given.')
        else:
            raise TypeError(f'Value of roll angle (in degrees) of object in {__class__.__name__} can be int or float. But roll angle = {value} of type {type(value)} is given.')
    
    @property # getter
    def pitch(self):
        return self._pitch
    
    @pitch.setter # setter
    def pitch(self, value):
        if isinstance(value, float) or isinstance(value, int):
            if value >= 0 and value <=180:
                self._pitch = float(value)
            else:
                raise ValueError(f'Value of pitch angle of object in {__class__.__name__} can be between [0,180] degrees. But pitch angle = {value} is given.')
        else:
            raise TypeError(f'Value of pitch angle (in degrees) of object in {__class__.__name__} can be int or float. But pitch angle = {value} of type {type(value)} is given.')
    
    
    @property # getter
    def yaw(self):
        return self._yaw
    
    @yaw.setter # setter
    def yaw(self, value):
        if isinstance(value, float) or isinstance(value, int):
            self._yaw = float(value)
            # if value >= 0 and value <=180:
            #     self._yaw = float(value)
            # else:
            #     raise ValueError(f'Value of yaw angle of object in {__class__.__name__} can be between [0,180] degrees. But yaw angle = {value} is given.')
        else:
            raise TypeError(f'Value of yaw angle (in degrees) of object in {__class__.__name__} can be int or float. But yaw angle = {value} of type {type(value)} is given.')
    
    def __str__(self):
        out = f'3D orientation - [roll = {self.roll}, pitch = {self.pitch}, yaw = {self.yaw}] radians.'
        return out

class Speed3d:
    def __init__(self,
                 vx: float = 0.0,
                 vy: float = 0.0,
                 vz: float = 0.0) -> None:
        
        self.vx = vx
        self.vy = vy
        self.vz = vz
        
    @property # getter
    def vx(self):
        return self._vx
    
    @vx.setter # setter
    def vx(self, value):
        if isinstance(value, float) or isinstance(value, int):
            if abs(value) > 50: 
                warnings.warn(f'Given Speed Vx = {value} m/sec ({value*3.6} kmph) is very high. Please check once.')
            self._vx = float(value)
        else:
            raise TypeError(f'Value of Vx (in m/sec) of object in {__class__.__name__} can be int or float. But Vx = {value} of type {type(value)} is given.')
        
    @property # getter
    def vy(self):
        return self._vy
    
    @vy.setter # setter
    def vy(self, value):
        if isinstance(value, float) or isinstance(value, int):
            if abs(value) > 50: 
                warnings.warn(f'Given Speed Vy = {value} m/sec ({value*3.6} kmph) is very high. Please check once.')
            self._vy = float(value)
        else:
            raise TypeError(f'Value of Vy (in m/sec) of object in {__class__.__name__} can be int or float. But Vy = {value} of type {type(value)} is given.')
    
    @property # getter
    def vz(self):
        return self._vz
    
    @vz.setter # setter
    def vz(self, value):
        if isinstance(value, float) or isinstance(value, int):
            if abs(value) > 50: 
                warnings.warn(f'Given Speed Vz = {value} m/sec ({value*3.6} kmph) is very high. Please check once.')
            self._vz = float(value)
        else:
            raise TypeError(f'Value of Vz (in m/sec) of object in {__class__.__name__} can be int or float. But Vz = {value} of type {type(value)} is given.')
    
    
    def __str__(self):
        out = f'3D speed - [vx = {self.vx}, vy = {self.vy}, vz = {self.vz}] meters'
        return out


#Check if category confidence and existance confidence available from 3dDetections
class Object3d:
    def __init__(self, 
                 #shape : Object3dShapeType = Object3dShapeType.UNKNOWN, 
                 position: Position3d = Position3d(0.0, 0.0, 0.0), 
                 size: Size3d = Size3d(0.0, 0.0, 0.0), 
                 orientation: Orientation3d = Orientation3d(0.0, 0.0, 0.0), 
                 speed: Speed3d = Speed3d(0.0, 0.0, 0.0),
                 yaw_angle : float = 0.0, 
                 class_name = " ", 
                 category_confidence : float = 0.0, 
                 existence_confidence: float = 0.0,
                 sensor_name = "lidar") -> None:
        
        
        #self.shape = shape
        self.position = position
        self.size = size
        self.orientation = orientation
        self.speed = speed
        self.yaw_angle = yaw_angle
        self.class_name = class_name
        self.category_confidence = category_confidence
        self.existence_confidence = existence_confidence
        self.sensor_name = sensor_name

    @property # getter
    def yaw_angle(self):
        return self._yaw_angle

    @yaw_angle.setter # setter
    def yaw_angle(self, value):
        self._yaw_angle = float(value)

    @property # getter
    def class_name(self):
        return self._class_name

    @class_name.setter # setter
    def class_name(self, value):
        self._class_name = value

    @property # getter
    def existence_confidence(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self._existence_confidence
    
    @existence_confidence.setter # setter
    def existence_confidence(self, value):
        if isinstance(value, float) or isinstance(value, int):
            if not (value >=0 and value < 100):
                raise ValueError(f'Value of 3d object existence confidence must be between [0,100] in {__class__.__name__}. But existence confidence = {value} is given.')
            else:
                self._existence_confidence = float(value)
        else:
            raise TypeError(f'Value of 3d object existence confidence must be of type float or int in {__class__.__name__}. But existence confidence = {value} of type {type(value)} is given.')

    
    @property # getter
    def category_confidence(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self._category_confidence
    
    @category_confidence.setter # setter
    def category_confidence(self, value):
        if isinstance(value, float) or isinstance(value, int):
            if not (value >=0 and value < 100):
                raise ValueError(f'Value of 3d object category confidence must be between [0,100] in {__class__.__name__}. But category confidence = {value} is given.')
            else:
                self._category_confidence = float(value)
        else:
            raise TypeError(f'Value of 3d object category confidence must be of type float or int in {__class__.__name__}. But category confidence = {value} of type {type(value)} is given.')

    def __str__(self) -> str:
        return(
            f'sensor name: {self.sensor_name}\n'
            f'{__class__.__name__}: class_name = {self.class_name}\n'
            f'{str(self.position)},\n' 
            f'{str(self.size)},\n'
            f'{str(self.orientation)},\n'
            f'speed = {str(self.speed)},\n'
            f'yaw_angle = {str(self.yaw_angle)},\n'
            f'category confidence = {self.category_confidence}, existence confidence = {self.existence_confidence}\n'
        )

class Object3dList:
    def __init__(self,
                 time_stamp,
                 #frame_id: int = 0,
                 objects_3d: List[Object3d] = None) -> None:
        
        self.time_stamp = time_stamp
        #self.frame_id = frame_id
        self.objects_3d = objects_3d if objects_3d is not None else []
        self._total_objects = 0 # this is computed property
        
    @property # getter (no setter required for this computed property)
    def total_objects(self):
       return len(self.objects_3d)
    
    """
    @property # getter
    def frame_id(self):
        return self._frame_id
     
    @frame_id.setter # setter
    def frame_id(self, value):
        if isinstance(value, int):
            if value < 0:
                raise ValueError(f'Value of frame id in {__class__.__name__} must be a positive integer. But frame id = {value} of type {type(value)} is given.') 
            else:
                self._frame_id = value
        else: 
            raise TypeError(f'Value of frame id must be of type int in {__class__.__name__}. But frame id = {value} of type {type(value)} is given.')
    """   
    @property # getter
    def objects_3d(self):
        return self._objects_3d
    
    def __iter__(self):
        for obj in self.objects_3d:
            yield obj
    
    @objects_3d.setter
    def objects_3d(self, value):
        if isinstance(value, list):
            if len(value) == 0: # if list is empty
                self._objects_3d = value
            else: # if list is not empty
                for obj in value:
                    if not isinstance(obj, (Object3d, type(None))): # check if all instances in list are of type Object3d
                        raise TypeError(f'List must contain only objects of Object3d in {__class__.__name__}. But object of type {type(obj)} is given.')
                self._objects_3d = value
        else:
            raise TypeError(f'Value of 3D objects must be given as list in {__class__.__name__}. But of type {type(value)} is given.')
        
    @property
    def time_stamp(self):
        return self._time_stamp
    
    @time_stamp.setter
    def time_stamp(self, value):
        #if isinstance(value, TimeStamp):
        self._time_stamp = value
        #else:
            #raise TypeError(f'Value of epoch time must be of type TimeStamp in {__class__.__name__}. But time_stamp = {value} of type {type(value)} is given.')
    
    def __str__(self):
        str_object_list = ""
        for index, obj in enumerate(self.objects_3d):
            str_object_list += str(index+1) + ". " + str(obj) +'\n'
        return(
            f'-------------------------------------\n'
            f'{__class__.__name__}: time_stamp = {self.time_stamp}, total objects = {self.total_objects}\n'
            f'{str_object_list}'
            f'-------------------------------------\n' 
        )