from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional
 
import numpy as np
 
 
class ModelName(str, Enum):
    CTRV = "ctrv"
    CTRA = "ctra"
 
 
# -----------------------------------------------------------------------------
# Angle helpers
# -----------------------------------------------------------------------------
 
def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)
 
 
def normalize_angle(angle: float) -> float:
    """Keep angle in [0, 2*pi). Uses modulo — safe for any magnitude."""
    return float(float(angle) % (2.0 * np.pi))
 
 
def correct_new_angle_and_diff(current_angle: float, new_angle_to_correct: float):
    # """
    # Return (corrected_angle, signed_diff) where corrected_angle is the equivalent
    # of new_angle_to_correct that minimises the absolute difference to current_angle.
 
    # signed_diff is always in [-pi, pi] — consistent across all cases.
 
    # Iterative (was recursive) — terminates in at most 2 iterations; the safety
    # cap at 3 guards against floating-point edge cases at the pi/2 and 3pi/2
    # boundaries.
    # """
    candidate = float(new_angle_to_correct)
 
    for _ in range(3):
        diff = normalize_angle(candidate) - normalize_angle(current_angle)
 
        if abs(diff) <= np.pi / 2.0:
            return float(candidate), float(diff)
 
        if abs(diff) >= 3.0 * np.pi / 2.0:
            # Shorter arc wraps past 0/2π — pick the equivalent angle closer to current.
            reduced = 2.0 * np.pi - abs(diff)
            sign = -1.0 if current_angle < candidate else 1.0
            corrected = float(current_angle + sign * reduced)
            # Compute signed diff consistently with case 1
            signed = normalize_angle(corrected) - normalize_angle(current_angle)
            if abs(signed) > np.pi:
                signed -= np.sign(signed) * 2.0 * np.pi
            return corrected, float(signed)
 
        # diff in (pi/2, 3pi/2): flip by pi and retry
        candidate = np.pi + candidate
 
    # Fallback (should never be reached)
    signed = wrap_to_pi(normalize_angle(candidate) - normalize_angle(current_angle))
    return float(candidate), float(signed)
 
 
# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
 
def _obj3d_to_z7(obj3d: Any) -> np.ndarray:
    """
    Measurement layout used by the tracker: [x, y, z, yaw, l, w, h].
    Handles None yaw defensively (does not use 'or 0.0' to avoid masking yaw=0).
    """
    raw_yaw = getattr(obj3d, "yaw_angle", None)
    yaw = float(raw_yaw) if raw_yaw is not None else 0.0
    return np.array(
        [
            float(obj3d.position.x),
            float(obj3d.position.y),
            float(obj3d.position.z),
            yaw,
            float(obj3d.size.x),
            float(obj3d.size.y),
            float(obj3d.size.z),
        ],
        dtype=float,
    )
 
 
def _speed_components(obj3d: Any) -> tuple[float, float, float]:
    speed = getattr(obj3d, "speed", None)
    if speed is None:
        return 0.0, 0.0, 0.0
    return (
        float(getattr(speed, "vx", 0.0) or 0.0),
        float(getattr(speed, "vy", 0.0) or 0.0),
        float(getattr(speed, "vz", 0.0) or 0.0),
    )
 
 
def numerical_jacobian(fn, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    y0 = np.asarray(fn(x), dtype=float).reshape(-1)
    m, n = y0.size, x.size
    J = np.zeros((m, n), dtype=float)
 
    for i in range(n):
        dx = np.zeros(n, dtype=float)
        step = eps * max(1.0, abs(x[i]))
        dx[i] = step
        yp = np.asarray(fn(x + dx), dtype=float).reshape(-1)
        ym = np.asarray(fn(x - dx), dtype=float).reshape(-1)
        J[:, i] = (yp - ym) / (2.0 * step)
 
    return J
 
 
# -----------------------------------------------------------------------------
# Default measurement noise sigmas: [x, y, z, yaw, l, w, h]
# -----------------------------------------------------------------------------
 
_DEFAULT_R_SIGMAS = np.array([0.8, 0.8, 0.6, 0.8, 0.5, 0.5, 0.5], dtype=float)
 
 
# -----------------------------------------------------------------------------
# Motion-model base class
# -----------------------------------------------------------------------------
 
class BaseNonLinearModel(ABC):
    # """
    # Base API for nonlinear motion models used by the EKF pipeline.
 
    # Measurement layout compatible with the tracker:
    #   z = [x, y, z, yaw, l, w, h]
 
    # H and R are constant for all concrete subclasses (measurement is simply
    # the first meas_dim state elements; noise is diagonal and fixed).
    # They are precomputed once and cached so the EKF does not recompute them
    # on every update call.
    # """
 
    meas_dim = 7
 
    def __init__(self, has_velo: bool = False, dt: float = 0.1,
                 r_sigmas: Optional[np.ndarray] = None) -> None:
        self.has_velo = bool(has_velo)
        self.dt = float(max(1e-3, dt))
 
        # Precompute constant R
        sigmas = np.asarray(r_sigmas, dtype=float) if r_sigmas is not None else _DEFAULT_R_SIGMAS
        if sigmas.shape != (self.meas_dim,):
            raise ValueError(f"r_sigmas must have length {self.meas_dim}, got {sigmas.shape}")
        self._R: np.ndarray = np.diag(sigmas ** 2)
 
        # H is built lazily after state_dim is known (property on subclass)
        self._H: Optional[np.ndarray] = None
 
    @property
    @abstractmethod
    def state_dim(self) -> int:
        raise NotImplementedError
 
    def _build_h(self) -> np.ndarray:
        """First-meas_dim rows of identity: selects [x,y,z,yaw,l,w,h] from state."""
        H = np.zeros((self.meas_dim, self.state_dim), dtype=float)
        for i in range(self.meas_dim):
            H[i, i] = 1.0
        return H
 
    @abstractmethod
    def get_init_state(self, obj3d: Any) -> np.ndarray:
        raise NotImplementedError
 
    @abstractmethod
    def get_init_cov_p(self) -> np.ndarray:
        raise NotImplementedError
 
    @abstractmethod
    def get_process_noise_q(self) -> np.ndarray:
        raise NotImplementedError
 
    def get_mea_noise_r(self) -> np.ndarray:
        """Return the precomputed constant R matrix (copy to prevent mutation)."""
        return self._R.copy()
 
    @abstractmethod
    def state_transition(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError
 
    def state_to_measure(self, state: np.ndarray) -> np.ndarray:
        z = np.asarray(state, dtype=float).reshape(-1)[: self.meas_dim].copy()
        z[3] = wrap_to_pi(z[3])
        return z.reshape(self.meas_dim, 1)
 
    @abstractmethod
    def get_transition_f(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError
 
    def get_mea_state_h(self, _state: np.ndarray) -> np.ndarray:
        """
        Return the precomputed constant H matrix (copy to prevent mutation).
        The _state argument is accepted for API compatibility but is ignored —
        the measurement model is linear and state-independent.
        """
        if self._H is None:
            self._H = self._build_h()
        return self._H.copy()
 
    @staticmethod
    def warp_res_yaw_to_pi(res: np.ndarray) -> np.ndarray:
        res = np.asarray(res, dtype=float).reshape(-1, 1)
        res[3, 0] = wrap_to_pi(float(res[3, 0]))
        return res
 
    @staticmethod
    def warp_state_yaw_to_pi(state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float).reshape(-1, 1)
        state[3, 0] = wrap_to_pi(float(state[3, 0]))
        return state
 
 
# -----------------------------------------------------------------------------
# CTRV
# -----------------------------------------------------------------------------
 
class CTRV(BaseNonLinearModel):
    """
    State vector : [x, y, z, yaw, l, w, h, v, yaw_rate, vz]
    Measurement  : [x, y, z, yaw, l, w, h]
    Jacobian     : analytical
    """
 
    @property
    def state_dim(self) -> int:
        return 10
 
    def get_init_state(self, obj3d: Any) -> np.ndarray:
        z = _obj3d_to_z7(obj3d)
        vx, vy, vz = _speed_components(obj3d)
        state = np.zeros((self.state_dim, 1), dtype=float)
        state[:7, 0] = z
        state[7, 0] = float(np.hypot(vx, vy)) if self.has_velo else 0.0
        state[8, 0] = 0.0   # yaw rate
        state[9, 0] = float(vz) if self.has_velo else 0.0
        state[3, 0] = wrap_to_pi(state[3, 0])
        return state
 
    def get_init_cov_p(self) -> np.ndarray:
        diag = np.array([4.0, 4.0, 2.0, 0.5, 1.0, 1.0, 1.0, 25.0, 4.0, 9.0], dtype=float)
        return np.diag(diag)
 
    def get_process_noise_q(self) -> np.ndarray:
        dt             = self.dt
        sigma_v        = 1.5
        sigma_yaw_rate = 0.2
        sigma_vz       = 0.7
        sigma_size     = 0.2
 
        Q = np.zeros((self.state_dim, self.state_dim), dtype=float)
        Q[0, 0] = (0.5 * sigma_v        * dt * dt) ** 2
        Q[1, 1] = (0.5 * sigma_v        * dt * dt) ** 2
        Q[2, 2] = (sigma_vz             * dt)      ** 2
        Q[3, 3] = (sigma_yaw_rate       * dt)      ** 2
        Q[4, 4] = (sigma_size ** 2)     * dt
        Q[5, 5] = (sigma_size ** 2)     * dt
        Q[6, 6] = (sigma_size ** 2)     * dt
        Q[7, 7] = (sigma_v        ** 2) * dt
        Q[8, 8] = (sigma_yaw_rate ** 2) * dt
        Q[9, 9] = (sigma_vz       ** 2) * dt
        return Q
 
    def state_transition(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float).reshape(-1)
        dt = self.dt
        x, y, z, yaw, l, w, h, v, omega, vz = state
        out = state.copy()
 
        if abs(omega) < 1e-3:
            ds   = v * dt
            out[0] = x + ds * np.cos(yaw)
            out[1] = y + ds * np.sin(yaw)
        else:
            yaw_next = yaw + omega * dt
            out[0] = x + (v / omega) * ( np.sin(yaw_next) - np.sin(yaw))
            out[1] = y + (v / omega) * (-np.cos(yaw_next) + np.cos(yaw))
 
        out[2] = z + vz * dt
        out[3] = wrap_to_pi(yaw + omega * dt)
        out[4] = l
        out[5] = w
        out[6] = h
        out[7] = v
        out[8] = omega
        out[9] = vz
        return out.reshape(self.state_dim, 1)
 
    def get_transition_f(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float).reshape(-1)
        dt = self.dt
        _, _, _, yaw, _, _, _, v, omega, _ = state
        F = np.eye(self.state_dim, dtype=float)
 
        if abs(omega) < 1e-3:
            c = np.cos(yaw)
            s = np.sin(yaw)
            F[0, 3] = -v * dt * s      # ∂x/∂yaw
            F[0, 7] =  dt * c          # ∂x/∂v
            F[1, 3] =  v * dt * c      # ∂y/∂yaw
            F[1, 7] =  dt * s          # ∂y/∂v
        else:
            yaw_next = yaw + omega * dt
            s0, c0   = np.sin(yaw),      np.cos(yaw)
            s1, c1   = np.sin(yaw_next), np.cos(yaw_next)
            inv_w    = 1.0 / omega
            inv_w2   = inv_w * inv_w
 
            F[0, 3] =  v * inv_w  * (c1 - c0)
            F[0, 7] =      inv_w  * (s1 - s0)
            F[0, 8] = -v * inv_w2 * (s1 - s0) + v * inv_w * c1 * dt
 
            F[1, 3] =  v * inv_w  * (s1 - s0)
            F[1, 7] =      inv_w  * (-c1 + c0)
            F[1, 8] = -v * inv_w2 * (-c1 + c0) + v * inv_w * s1 * dt
 
        F[2, 9] = dt   # ∂z/∂vz
        F[3, 8] = dt   # ∂yaw/∂omega
        return F
 
 
# -----------------------------------------------------------------------------
# CTRA
# -----------------------------------------------------------------------------
 
class CTRA(BaseNonLinearModel):
    """
    State vector : [x, y, z, yaw, l, w, h, v, a, yaw_rate, vz]
    Measurement  : [x, y, z, yaw, l, w, h]
    Jacobian     : numerical (CTRA analytical form is large; numerical is
                   accurate given the step-size used and the smoothness of the
                   state transition outside the wrap_to_pi cut).
    """
 
    @property
    def state_dim(self) -> int:
        return 11
 
    def get_init_state(self, obj3d: Any) -> np.ndarray:
        z = _obj3d_to_z7(obj3d)
        vx, vy, vz = _speed_components(obj3d)
        state = np.zeros((self.state_dim, 1), dtype=float)
        state[:7, 0]  = z
        state[7,  0]  = float(np.hypot(vx, vy)) if self.has_velo else 0.0
        state[8,  0]  = 0.0   # longitudinal acceleration
        state[9,  0]  = 0.0   # yaw rate
        state[10, 0]  = float(vz) if self.has_velo else 0.0
        state[3,  0]  = wrap_to_pi(state[3, 0])
        return state
 
    def get_init_cov_p(self) -> np.ndarray:
        diag = np.array([4.0, 4.0, 2.0, 0.5, 1.0, 1.0, 1.0, 25.0, 16.0, 4.0, 9.0], dtype=float)
        return np.diag(diag)
 
    def get_process_noise_q(self) -> np.ndarray:
        dt             = self.dt
        sigma_v        = 1.2
        sigma_a        = 1.5
        sigma_yaw_rate = 0.3
        sigma_vz       = 0.7
        sigma_size     = 0.2
 
        Q = np.zeros((self.state_dim, self.state_dim), dtype=float)
        Q[0,  0]  = (0.5 * sigma_a      * dt * dt) ** 2
        Q[1,  1]  = (0.5 * sigma_a      * dt * dt) ** 2
        Q[2,  2]  = (sigma_vz           * dt)      ** 2
        Q[3,  3]  = (sigma_yaw_rate     * dt)      ** 2
        Q[4,  4]  = (sigma_size ** 2)   * dt
        Q[5,  5]  = (sigma_size ** 2)   * dt
        Q[6,  6]  = (sigma_size ** 2)   * dt
        Q[7,  7]  = (sigma_v        ** 2) * dt
        Q[8,  8]  = (sigma_a        ** 2) * dt
        Q[9,  9]  = (sigma_yaw_rate ** 2) * dt
        Q[10, 10] = (sigma_vz       ** 2) * dt
        return Q
 
    def state_transition(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float).reshape(-1)
        dt = self.dt
        x, y, z, yaw, l, w, h, v, a, omega, vz = state
        out = state.copy()
 
        if abs(omega) < 1e-3:
            ds     = v * dt + 0.5 * a * dt * dt
            out[0] = x + ds * np.cos(yaw)
            out[1] = y + ds * np.sin(yaw)
        else:
            yaw_next = yaw + omega * dt
            s0, c0   = np.sin(yaw),      np.cos(yaw)
            s1, c1   = np.sin(yaw_next), np.cos(yaw_next)
            inv_w    = 1.0 / omega
            inv_w2   = inv_w * inv_w
 
            out[0] = x + (v * inv_w) * (s1 - s0)   + a * ( dt * s1 * inv_w + (c1 - c0) * inv_w2)
            out[1] = y + (v * inv_w) * (-c1 + c0)   + a * (-dt * c1 * inv_w + (s1 - s0) * inv_w2)
 
        out[2]  = z + vz * dt
        out[3]  = wrap_to_pi(yaw + omega * dt)
        out[4]  = l
        out[5]  = w
        out[6]  = h
        out[7]  = v + a * dt
        out[8]  = a
        out[9]  = omega
        out[10] = vz
        return out.reshape(self.state_dim, 1)
 
    def get_transition_f(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float).reshape(-1)
        return numerical_jacobian(lambda xx: self.state_transition(xx).reshape(-1), state)
 
 
# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------
 
def make_model(model: str, has_velo: bool = False, dt: float = 0.1,
               r_sigmas: Optional[np.ndarray] = None) -> BaseNonLinearModel:
    model = str(model).lower().strip()
    if model == ModelName.CTRV.value:
        return CTRV(has_velo=has_velo, dt=dt, r_sigmas=r_sigmas)
    if model == ModelName.CTRA.value:
        return CTRA(has_velo=has_velo, dt=dt, r_sigmas=r_sigmas)
    raise ValueError(f"Unknown nonlinear motion model: {model!r}")









# from abc import ABC, abstractmethod
# from enum import Enum
# from typing import Any

# import numpy as np


# class ModelName(str, Enum):
#     CTRV = "ctrv"
#     CTRA = "ctra"


# # -----------------------------------------------------------------------------
# # Angle helpers
# # -----------------------------------------------------------------------------

# def wrap_to_pi(angle: float) -> float:
#     """Wrap angle to [-pi, pi)."""
#     return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


# def normalize_angle(angle: float) -> float:
#     """Keep angle in [0, 2*pi]."""
#     angle = float(angle)
#     while angle < 0.0:
#         angle += 2.0 * np.pi
#     while angle > 2.0 * np.pi:
#         angle -= 2.0 * np.pi
#     return angle


# def correct_new_angle_and_diff(current_angle: float, new_angle_to_correct: float):
#     """Return an equivalent angle whose difference to current_angle is minimal."""
#     abs_diff = normalize_angle(new_angle_to_correct) - normalize_angle(current_angle)

#     if abs(abs_diff) <= np.pi / 2.0:
#         return float(new_angle_to_correct), float(abs_diff)

#     if abs(abs_diff) >= 3.0 * np.pi / 2.0:
#         abs_diff = 2.0 * np.pi - abs(abs_diff)
#         if current_angle < new_angle_to_correct:
#             return float(current_angle - abs_diff), float(abs_diff)
#         return float(current_angle + abs_diff), float(abs_diff)

#     return correct_new_angle_and_diff(current_angle, np.pi + new_angle_to_correct)


# # -----------------------------------------------------------------------------
# # Generic helpers
# # -----------------------------------------------------------------------------

# def _obj3d_to_z7(obj3d: Any) -> np.ndarray:
#     """Measurement layout used by your current pipeline: [x, y, z, yaw, l, w, h]."""
#     yaw = float(getattr(obj3d, "yaw_angle", 0.0) or 0.0)
#     return np.array(
#         [
#             float(obj3d.position.x),
#             float(obj3d.position.y),
#             float(obj3d.position.z),
#             yaw,
#             float(obj3d.size.x),
#             float(obj3d.size.y),
#             float(obj3d.size.z),
#         ],
#         dtype=float,
#     )


# def _speed_components(obj3d: Any) -> tuple[float, float, float]:
#     speed = getattr(obj3d, "speed", None)
#     if speed is None:
#         return 0.0, 0.0, 0.0
#     return (
#         float(getattr(speed, "vx", 0.0) or 0.0),
#         float(getattr(speed, "vy", 0.0) or 0.0),
#         float(getattr(speed, "vz", 0.0) or 0.0),
#     )


# def numerical_jacobian(fn, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
#     x = np.asarray(x, dtype=float).reshape(-1)
#     y0 = np.asarray(fn(x), dtype=float).reshape(-1)
#     m, n = y0.size, x.size
#     J = np.zeros((m, n), dtype=float)

#     for i in range(n):
#         dx = np.zeros(n, dtype=float)
#         step = eps * max(1.0, abs(x[i]))
#         dx[i] = step
#         yp = np.asarray(fn(x + dx), dtype=float).reshape(-1)
#         ym = np.asarray(fn(x - dx), dtype=float).reshape(-1)
#         J[:, i] = (yp - ym) / (2.0 * step)

#     return J


# # -----------------------------------------------------------------------------
# # Motion-model base class
# # -----------------------------------------------------------------------------

# class BaseNonLinearModel(ABC):
#     """
#     Base API for nonlinear motion models used by the new EKF pipeline.

#     Measurement layout is kept compatible with your current tracker:
#       z = [x, y, z, yaw, l, w, h]
#     """

#     meas_dim = 7

#     def __init__(self, has_velo: bool = False, dt: float = 0.1) -> None:
#         self.has_velo = bool(has_velo)
#         self.dt = float(max(1e-3, dt))

#     @property
#     @abstractmethod
#     def state_dim(self) -> int:
#         raise NotImplementedError

#     @abstractmethod
#     def get_init_state(self, obj3d: Any) -> np.ndarray:
#         raise NotImplementedError

#     @abstractmethod
#     def get_init_cov_p(self) -> np.ndarray:
#         raise NotImplementedError

#     @abstractmethod
#     def get_process_noise_q(self) -> np.ndarray:
#         raise NotImplementedError

#     def get_mea_noise_r(self) -> np.ndarray:
#         # [x, y, z, yaw, l, w, h]
#         sigmas = np.array([0.8, 0.8, 0.6, 0.25, 0.5, 0.5, 0.5], dtype=float)
#         return np.diag(sigmas ** 2)

#     @abstractmethod
#     def state_transition(self, state: np.ndarray) -> np.ndarray:
#         raise NotImplementedError

#     def state_to_measure(self, state: np.ndarray) -> np.ndarray:
#         z = np.asarray(state, dtype=float).reshape(-1)[: self.meas_dim].copy()
#         z[3] = wrap_to_pi(z[3])
#         return z.reshape(self.meas_dim, 1)

#     @abstractmethod
#     def get_transition_f(self, state: np.ndarray) -> np.ndarray:
#         raise NotImplementedError

#     def get_mea_state_h(self, _state: np.ndarray) -> np.ndarray:
#         H = np.zeros((self.meas_dim, self.state_dim), dtype=float)
#         for i in range(self.meas_dim):
#             H[i, i] = 1.0
#         return H

#     @staticmethod
#     def warp_res_yaw_to_pi(res: np.ndarray) -> np.ndarray:
#         res = np.asarray(res, dtype=float).reshape(-1, 1)
#         res[3, 0] = wrap_to_pi(float(res[3, 0]))
#         return res

#     @staticmethod
#     def warp_state_yaw_to_pi(state: np.ndarray) -> np.ndarray:
#         state = np.asarray(state, dtype=float).reshape(-1, 1)
#         state[3, 0] = wrap_to_pi(float(state[3, 0]))
#         return state


# # -----------------------------------------------------------------------------
# # CTRV
# # -----------------------------------------------------------------------------

# class CTRV(BaseNonLinearModel):
#     """
#     State vector:
#       [x, y, z, yaw, l, w, h, v, yaw_rate, vz]

#     Measurement vector:
#       [x, y, z, yaw, l, w, h]
#     """

#     @property
#     def state_dim(self) -> int:
#         return 10

#     def get_init_state(self, obj3d: Any) -> np.ndarray:
#         z = _obj3d_to_z7(obj3d)
#         vx, vy, vz = _speed_components(obj3d)
#         state = np.zeros((self.state_dim, 1), dtype=float)
#         state[:7, 0] = z
#         state[7, 0] = float(np.hypot(vx, vy)) if self.has_velo else 0.0
#         state[8, 0] = 0.0  # yaw rate
#         state[9, 0] = float(vz) if self.has_velo else 0.0
#         state[3, 0] = wrap_to_pi(state[3, 0])
#         return state

#     def get_init_cov_p(self) -> np.ndarray:
#         diag = np.array([4.0, 4.0, 2.0, 0.5, 1.0, 1.0, 1.0, 25.0, 4.0, 9.0], dtype=float)
#         return np.diag(diag)

#     def get_process_noise_q(self) -> np.ndarray:
#         dt = self.dt
#         sigma_v = 2.5
#         sigma_yaw_rate = 0.8
#         sigma_vz = 1.0
#         sigma_size = 0.25

#         Q = np.zeros((self.state_dim, self.state_dim), dtype=float)
#         Q[0, 0] = (0.5 * sigma_v * dt * dt) ** 2
#         Q[1, 1] = (0.5 * sigma_v * dt * dt) ** 2
#         Q[2, 2] = (sigma_vz * dt) ** 2
#         Q[3, 3] = (sigma_yaw_rate * dt) ** 2
#         Q[4, 4] = (sigma_size ** 2) * dt
#         Q[5, 5] = (sigma_size ** 2) * dt
#         Q[6, 6] = (sigma_size ** 2) * dt
#         Q[7, 7] = (sigma_v ** 2) * dt
#         Q[8, 8] = (sigma_yaw_rate ** 2) * dt
#         Q[9, 9] = (sigma_vz ** 2) * dt
#         return Q

#     def state_transition(self, state: np.ndarray) -> np.ndarray:
#         state = np.asarray(state, dtype=float).reshape(-1)
#         dt = self.dt
#         x, y, z, yaw, l, w, h, v, omega, vz = state
#         out = state.copy()

#         if abs(omega) < 1e-3:
#             ds = v * dt
#             out[0] = x + ds * np.cos(yaw)
#             out[1] = y + ds * np.sin(yaw)
#         else:
#             yaw_next = yaw + omega * dt
#             out[0] = x + (v / omega) * (np.sin(yaw_next) - np.sin(yaw))
#             out[1] = y + (v / omega) * (-np.cos(yaw_next) + np.cos(yaw))

#         out[2] = z + vz * dt
#         out[3] = wrap_to_pi(yaw + omega * dt)
#         out[4] = l
#         out[5] = w
#         out[6] = h
#         out[7] = v
#         out[8] = omega
#         out[9] = vz
#         return out.reshape(self.state_dim, 1)

#     def get_transition_f(self, state: np.ndarray) -> np.ndarray:
#         state = np.asarray(state, dtype=float).reshape(-1)
#         dt = self.dt
#         _, _, _, yaw, _, _, _, v, omega, _ = state
#         F = np.eye(self.state_dim, dtype=float)

#         if abs(omega) < 1e-3:
#             c = np.cos(yaw)
#             s = np.sin(yaw)
#             F[0, 3] = -v * dt * s
#             F[0, 7] = dt * c
#             F[1, 3] = v * dt * c
#             F[1, 7] = dt * s
#         else:
#             yaw_next = yaw + omega * dt
#             s0, c0 = np.sin(yaw), np.cos(yaw)
#             s1, c1 = np.sin(yaw_next), np.cos(yaw_next)
#             inv_w = 1.0 / omega
#             inv_w2 = inv_w * inv_w

#             F[0, 3] = v * inv_w * (c1 - c0)
#             F[0, 7] = inv_w * (s1 - s0)
#             F[0, 8] = -v * inv_w2 * (s1 - s0) + v * inv_w * c1 * dt

#             F[1, 3] = v * inv_w * (s1 - s0)
#             F[1, 7] = inv_w * (-c1 + c0)
#             F[1, 8] = -v * inv_w2 * (-c1 + c0) + v * inv_w * s1 * dt

#         F[2, 9] = dt
#         F[3, 8] = dt
#         return F


# # -----------------------------------------------------------------------------
# # CTRA
# # -----------------------------------------------------------------------------

# class CTRA(BaseNonLinearModel):
#     """
#     State vector:
#       [x, y, z, yaw, l, w, h, v, a, yaw_rate, vz]

#     Measurement vector:
#       [x, y, z, yaw, l, w, h]
#     """

#     @property
#     def state_dim(self) -> int:
#         return 11

#     def get_init_state(self, obj3d: Any) -> np.ndarray:
#         z = _obj3d_to_z7(obj3d)
#         vx, vy, vz = _speed_components(obj3d)
#         state = np.zeros((self.state_dim, 1), dtype=float)
#         state[:7, 0] = z
#         state[7, 0] = float(np.hypot(vx, vy)) if self.has_velo else 0.0
#         state[8, 0] = 0.0  # longitudinal acceleration
#         state[9, 0] = 0.0  # yaw rate
#         state[10, 0] = float(vz) if self.has_velo else 0.0
#         state[3, 0] = wrap_to_pi(state[3, 0])
#         return state

#     def get_init_cov_p(self) -> np.ndarray:
#         diag = np.array([4.0, 4.0, 2.0, 0.5, 1.0, 1.0, 1.0, 25.0, 16.0, 4.0, 9.0], dtype=float)
#         return np.diag(diag)

#     def get_process_noise_q(self) -> np.ndarray:
#         dt = self.dt
#         sigma_v = 2.0
#         sigma_a = 3.0
#         sigma_yaw_rate = 0.8
#         sigma_vz = 1.0
#         sigma_size = 0.25

#         Q = np.zeros((self.state_dim, self.state_dim), dtype=float)
#         Q[0, 0] = (0.5 * sigma_a * dt * dt) ** 2
#         Q[1, 1] = (0.5 * sigma_a * dt * dt) ** 2
#         Q[2, 2] = (sigma_vz * dt) ** 2
#         Q[3, 3] = (sigma_yaw_rate * dt) ** 2
#         Q[4, 4] = (sigma_size ** 2) * dt
#         Q[5, 5] = (sigma_size ** 2) * dt
#         Q[6, 6] = (sigma_size ** 2) * dt
#         Q[7, 7] = (sigma_v ** 2) * dt
#         Q[8, 8] = (sigma_a ** 2) * dt
#         Q[9, 9] = (sigma_yaw_rate ** 2) * dt
#         Q[10, 10] = (sigma_vz ** 2) * dt
#         return Q

#     def state_transition(self, state: np.ndarray) -> np.ndarray:
#         state = np.asarray(state, dtype=float).reshape(-1)
#         dt = self.dt
#         x, y, z, yaw, l, w, h, v, a, omega, vz = state
#         out = state.copy()

#         if abs(omega) < 1e-3:
#             ds = v * dt + 0.5 * a * dt * dt
#             out[0] = x + ds * np.cos(yaw)
#             out[1] = y + ds * np.sin(yaw)
#         else:
#             yaw_next = yaw + omega * dt
#             s0, c0 = np.sin(yaw), np.cos(yaw)
#             s1, c1 = np.sin(yaw_next), np.cos(yaw_next)
#             inv_w = 1.0 / omega
#             inv_w2 = inv_w * inv_w

#             out[0] = x + (v * inv_w) * (s1 - s0) + a * ((dt * s1) * inv_w + (c1 - c0) * inv_w2)
#             out[1] = y + (v * inv_w) * (-c1 + c0) + a * ((-dt * c1) * inv_w + (s1 - s0) * inv_w2)

#         out[2] = z + vz * dt
#         out[3] = wrap_to_pi(yaw + omega * dt)
#         out[4] = l
#         out[5] = w
#         out[6] = h
#         out[7] = v + a * dt
#         out[8] = a
#         out[9] = omega
#         out[10] = vz
#         return out.reshape(self.state_dim, 1)

#     def get_transition_f(self, state: np.ndarray) -> np.ndarray:
#         state = np.asarray(state, dtype=float).reshape(-1)
#         return numerical_jacobian(lambda xx: self.state_transition(xx).reshape(-1), state)


# # -----------------------------------------------------------------------------
# # Factory
# # -----------------------------------------------------------------------------

# def make_model(model: str, has_velo: bool = False, dt: float = 0.1) -> BaseNonLinearModel:
#     model = str(model).lower().strip()
#     if model == ModelName.CTRV.value:
#         return CTRV(has_velo=has_velo, dt=dt)
#     if model == ModelName.CTRA.value:
#         return CTRA(has_velo=has_velo, dt=dt)
#     raise ValueError(f"Unknown nonlinear motion model: {model}")
