from typing import Tuple
from enum import Enum
import numpy as np
from filterpy.kalman import KalmanFilter

class ModelName(str, Enum):
    CV = "cv"
    CA = "ca"


def CV3D() -> KalmanFilter:
    """
    Measurement z (dim_z=7):
      [x, y, z, yaw, l, w, h]

    State x:
      [x, y, z, yaw, l, w, h, vx, vy, vz]
    """
    kf = KalmanFilter(dim_x=10, dim_z=7)

    kf.F = np.eye(10, dtype=float)
    kf.F[0, 7] = 1.0
    kf.F[1, 8] = 1.0
    kf.F[2, 9] = 1.0

    kf.H = np.zeros((7, 10), dtype=float)
    kf.H[0, 0] = 1.0
    kf.H[1, 1] = 1.0
    kf.H[2, 2] = 1.0
    kf.H[3, 3] = 1.0
    kf.H[4, 4] = 1.0
    kf.H[5, 5] = 1.0
    kf.H[6, 6] = 1.0

    kf.P = np.eye(10, dtype=float)
    kf.P[0:3, 0:3] *= 4.0     
    kf.P[3,   3]   *= 0.25    
    kf.P[4:7, 4:7] *= 0.25
    kf.P[7:,  7:]  *= 25.0

    # Q
    kf.Q = np.eye(10, dtype=float)
    kf.Q[0:7, 0:7] *= 0.01
    kf.Q[7:, 7:] *= 0.01

    #r_current
    kf.R = np.diag([
        0.10, 0.10, 0.20,   # x, y, z
        0.30,               # yaw
        0.20, 0.20, 0.30    # l, w, h
    ])
    #r_very_loose
    # kf.R = np.diag([
    #     1.50, 1.50, 2.0,   # x, y, z
    #     2.00,               # yaw
    #     1.50, 1.50, 1.50    # l, w, h
    # ])
    #r_tight
    # kf.R = np.diag([
    #     0.05, 0.05, 0.10,   # x, y, z
    #     0.15,               # yaw
    #     0.10, 0.10, 0.15    # l, w, h
    # ])
    #r_medium
    # kf.R = np.diag([
    #     0.4, 0.4, 0.50,   # x, y, z
    #     0.6,               # yaw
    #     0.40, 0.40, 0.5    # l, w, h
    # ])
    # #r_loose
    # kf.R = np.diag([
    #     0.6, 0.6, 0.70,   # x, y, z
    #     1.0,               # yaw
    #     0.70, 0.70, 0.8    # l, w, h
    # ])

    kf.model_name = "CV3D"
    kf.state_dim = 10
    kf.meas_dim = 7

    return kf

def set_kf_dt(kf: KalmanFilter, dt: float) -> None:
    dt = float(max(1e-3, dt))
    dim_x = int(kf.dim_x)

    if dim_x == 10:
        kf.F[:] = np.eye(10, dtype=float)
        kf.F[0, 7] = dt
        kf.F[1, 8] = dt
        kf.F[2, 9] = dt
        return

    if dim_x == 13:
        kf.F[:] = np.eye(13, dtype=float)

        kf.F[0, 7] = dt
        kf.F[1, 8] = dt
        kf.F[2, 9] = dt

        dt2 = dt * dt
        kf.F[0, 10] = 0.5 * dt2
        kf.F[1, 11] = 0.5 * dt2
        kf.F[2, 12] = 0.5 * dt2

        kf.F[7, 10] = dt
        kf.F[8, 11] = dt
        kf.F[9, 12] = dt
        return

    raise ValueError(f"Unsupported KF dim_x={dim_x} in set_kf_dt()")

def set_kf_q(kf: KalmanFilter, dt: float,
             sigma_ax: float = 4.0,
             sigma_ay: float = 4.0,
             sigma_az: float = 0.8,
             sigma_jx: float = 2.0,
             sigma_jy: float = 2.0,
             sigma_jz: float = 0.5,
             sigma_yaw: float = 0.2,   # rad/sqrt(s)
             sigma_size: float = 0.3  # m/sqrt(s)
             ) -> None:

    dt = float(max(1e-3, dt))
    dim_x = int(kf.dim_x)

    if dim_x == 10:
        Q = np.zeros((10, 10), dtype=float)

        def q_cv_1d(sigma_a: float) -> np.ndarray:
            dt2 = dt*dt
            dt3 = dt2*dt
            dt4 = dt2*dt2
            return (sigma_a**2) * np.array([
                [dt4/4.0, dt3/2.0],
                [dt3/2.0, dt2]
            ], dtype=float)

        Q[np.ix_([0,7],[0,7])] = q_cv_1d(sigma_ax)
        Q[np.ix_([1,8],[1,8])] = q_cv_1d(sigma_ay)
        Q[np.ix_([2,9],[2,9])] = q_cv_1d(sigma_az)
        Q[3,3] = (sigma_yaw**2) * dt
        Q[4,4] = (sigma_size**2) * dt
        Q[5,5] = (sigma_size**2) * dt
        Q[6,6] = (sigma_size**2) * dt

        kf.Q[:] = Q
        return

    if dim_x == 13:
        Q = np.zeros((13, 13), dtype=float)

        def q_ca_1d(sigma_j: float) -> np.ndarray:
            dt2 = dt*dt
            dt3 = dt2*dt
            dt4 = dt2*dt2
            dt5 = dt4*dt
            return (sigma_j**2) * np.array([
                [dt5/20.0, dt4/8.0, dt3/6.0],
                [dt4/8.0,  dt3/3.0, dt2/2.0],
                [dt3/6.0,  dt2/2.0, dt]
            ], dtype=float)

        Q[np.ix_([0,7,10],[0,7,10])] = q_ca_1d(sigma_jx)
        Q[np.ix_([1,8,11],[1,8,11])] = q_ca_1d(sigma_jy)
        Q[np.ix_([2,9,12],[2,9,12])] = q_ca_1d(sigma_jz)
        Q[3,3] = (sigma_yaw**2) * dt
        Q[4,4] = (sigma_size**2) * dt
        Q[5,5] = (sigma_size**2) * dt
        Q[6,6] = (sigma_size**2) * dt

        kf.Q[:] = Q
        return

def CA3D() -> KalmanFilter:
    """
    Constant-Acceleration 3D KF (NO velocity measurement)

    Measurement z (dim_z=7):
      [x, y, z, yaw, l, w, h]

    State x (dim_x=13):
      [x, y, z, yaw, l, w, h, vx, vy, vz, ax, ay, az]
    """

    kf = KalmanFilter(dim_x=13, dim_z=7)
    kf.F = np.eye(13, dtype=float)

    kf.H = np.zeros((7, 13), dtype=float)
    for i in range(7):
        kf.H[i, i] = 1.0

    kf.P = np.eye(13, dtype=float)
    kf.P[0:3, 0:3] *= 4.0     
    kf.P[3,   3]   *= 0.25    
    kf.P[4:7, 4:7] *= 0.25    
    kf.P[7:10, 7:10] *= 25.0  
    kf.P[10:13,10:13]*= 9.0

    kf.Q = np.eye(13, dtype=float)
    kf.Q[0:7, 0:7] *= 0.01       
    kf.Q[7:10, 7:10] *= 0.1      
    kf.Q[10:13, 10:13] *= 1.0    

    kf.R = np.diag([
        0.10, 0.10, 0.20,   # x, y, z
        0.30,               # yaw
        0.20, 0.20, 0.30    # l, w, h
    ])

    kf.model_name = "CA3D"
    kf.state_dim = 13
    kf.meas_dim = 7
    
    return kf

def make_kf(model: str) -> KalmanFilter:
    """
    Factory:
      - 'cv' -> CV3D()
      - 'ca' -> CA3D()
    """
    model = str(model).lower().strip()
    if model == ModelName.CV.value:
        return CV3D()
    if model == ModelName.CA.value:
        return CA3D()
    raise ValueError(f"Unknown motion model: {model}")

def normalize_angle(angle: float) -> float:
    """Keep the angle in [0, 2*pi]."""
    while angle < 0.0:
        angle += 2 * np.pi
    while angle > 2 * np.pi:
        angle -= 2 * np.pi
    return angle


def correct_new_angle_and_diff(current_angle: float, new_angle_to_correct: float) -> Tuple[float, float]:
    """
    Return an angle equivalent to new_angle_to_correct so that its difference to current_angle is minimal.
    """
    abs_diff = normalize_angle(new_angle_to_correct) - normalize_angle(current_angle)

    if abs(abs_diff) <= np.pi / 2.0:
        return new_angle_to_correct, abs_diff

    if abs(abs_diff) >= 3.0 * np.pi / 2.0:
        abs_diff = 2 * np.pi - abs(abs_diff)
        if current_angle < new_angle_to_correct:
            return current_angle - abs_diff, abs_diff
        else:
            return current_angle + abs_diff, abs_diff

    return correct_new_angle_and_diff(current_angle, np.pi + new_angle_to_correct)
