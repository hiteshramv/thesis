# Can time stamp be imported from rosbag

import numpy as np

class ImageData:
    '''
    The class Image defines the data structure for one Image, including many other parameters associated with image
    '''
    def __init__(self,
                 camera_id: int, 
                 time_stamp , 
                 image_data : np.ndarray, 
                 ):
        
        # member variables
        self.camera_id = camera_id
        self.time_stamp = time_stamp
        self.data = image_data
        self._width = None
        self._height = None
        self._channels = None
        
    # getter and setters
    @property
    def camera_id(self):
        return self._camera_id
    
    @camera_id.setter
    def camera_id(self, value):
        # validate frame id
        if not value >= 1 and value < 4:
            raise ValueError("Frame id = " + str(value) + " for the image is invalid. It must be between 1-3")
        if not isinstance(value, int):
            raise TypeError("Frame id = " + str(value) + " for the image is invalid. It must be an integer value.")
        
        self._camera_id = value
    
    @property
    def time_stamp(self):
        return self._time_stamp
    
    @time_stamp.setter
    def time_stamp(self, value):
        self._time_stamp = value
    
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, value):
        if not isinstance(value, np.ndarray):
            raise TypeError("Image data is invalid. It must be numpy 2D array with one or more channels")
        self._data = value
        
        # set other properties to None when new image is set
        self._width = None
        self._height = None
        self._channels = None


    # only getter
    @property
    def width(self):
        if self._width is None:
            _, self._width, _ = self._data.shape
        return self._width
    
    # only getter
    @property
    def height(self):
        if self._height is None:
            self._height, _ , _ = self._data.shape
        return self._height
    
    # only getter
    @property
    def channels(self):
        if self._channels is None:
            _, _, self._channels = self._data.shape
        return self._channels

    def __repr__(self) -> str:
        out = "----Image-----\n" \
            + "camera id: " + str(self.camera_id) + "\n" \
            + "size: [" + str(self.height) + "," + str(self.width) + "," + str(self.channels) + "]\n" \
            + "timestamp: " + str(self.time_stamp) #\
            #+ "values: " + str(self.data) + "\n"
            
        return out
