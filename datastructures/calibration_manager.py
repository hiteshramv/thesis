import numpy as np
from typing import Dict, Any, Tuple, List

class CalibrationManager:

    """
    Handles calibration: provides intrinsics, extrinsics, and distortion.
    Loads from a calibration dictionary (e.g., from final_calibration_dict.py).
    """

    def __init__(self, calib_data: Dict[str, Any]):
        self.calib_data = calib_data

    def get_intrinsic(self, cam_id: int) -> np.ndarray:
        """
        Returns 3x3 camera intrinsic matrix.
        """
        cam_key = f"camera_{cam_id:02d}"
        return np.array(self.calib_data["cameras"][cam_key]["intrinsic"],  dtype=np.float32)

    def get_extrinsic(self, cam_id: int) -> np.ndarray:
        """
        Returns 3x4 camera extrinsic matrix.
        """
        cam_key = f"camera_{cam_id:02d}"
        return np.array(self.calib_data["cameras"][cam_key]["extrinsic"],  dtype=np.float32)

    def get_distortion(self, cam_id: int) -> np.ndarray:
        """
        Returns camera distortion matrix.
        """
        cam_key = f"camera_{cam_id:02d}"
        return np.array(self.calib_data["cameras"][cam_key]["distortion"],  dtype=np.float32)

    def get_ouster_extrinsic(self):
        """
        Returns ouster lidar's extrinsic matrix
        """
        return np.array(self.calib_data["lidars"]["ouster"]["extrinsic"], dtype=np.float32)

    def get_dome_extrinsic(self):
        """
        Returns dome lidar's extrinsic matrix
        """
        return np.array(self.calib_data["lidars"]["dome"]["extrinsic"], dtype=np.float32)

    def get_ouster_to_ground_extrinsic(self):
        """
        Returns ouster_to_ground extrinsic matrix
        """
        return np.array(self.calib_data["ouster_to_ground"]["extrinsic"], dtype=np.float32)

    