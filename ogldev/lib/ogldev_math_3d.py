import numpy as np
from dataclasses import dataclass
# from typing import Self
from typing_extensions import Self
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class PersProjInfo:
    fov: float = 0.0
    width: float = 0.0
    height: float = 0.0
    z_near: float = 0.0
    z_far: float = 0.0

    def to_matrix(self):
        m = np.identity(4)
        ar = self.height / self.width
        z_range = self.z_near - self.z_far
        tan_half_fov = np.tan(np.deg2rad(self.fov / 2))

        m[0][0] = 1/tan_half_fov 
        m[0][1] = 0.0                 
        m[0][2] = 0.0                        
        m[0][3] = 0.0
        
        m[1][0] = 0.0         
        m[1][1] = 1.0/(tan_half_fov*ar) 
        m[1][2] = 0.0                        
        m[1][3] = 0.0
        
        m[2][0] = 0.0         
        m[2][1] = 0.0                 
        m[2][2] = (-self.z_near - self.z_far)/z_range  
        m[2][3] = 2.0*self.z_far*self.z_near/z_range
        
        m[3][0] = 0.0         
        m[3][1] = 0.0                 
        m[3][2] = 1.0                        
        m[3][3] = 0.0
        return m


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm


def init_translation_transform(pos):
    m = np.identity(4)
    m[:3, 3] = pos
    return m


def init_camera_transform(target, up):
    n = normalize(target)
    up_norm = normalize(up)
    u = np.cross(up_norm, n)
    u = normalize(u)
    v = np.cross(n, u)

    m = np.identity(4)

    m[0, :3] = u
    m[1, :3] = v
    m[2, :3] = n
    return m


def init_rotation_transform(x, y, z, degree=True):
    if degree:
        x = np.deg2rad(x)
        y = np.deg2rad(y)
        z = np.deg2rad(z)
    m = np.identity(4)
    m[0][0] = 1.0; m[0][1] = 0.0   ;  m[0][2] = 0.0    ; m[0][3] = 0.0;
    m[1][0] = 0.0; m[1][1] = np.cos(x);  m[1][2] = np.sin(x);  m[1][3] = 0.0;
    m[2][0] = 0.0; m[2][1] = -np.sin(x); m[2][2] = np.cos(x) ; m[2][3] = 0.0;
    m[3][0] = 0.0; m[3][1] = 0.0   ;  m[3][2] = 0.0    ; m[3][3] = 1.0;
    rx = m

    m = np.identity(4)
    m[0][0] = np.cos(y); m[0][1] = 0.0; m[0][2] = -np.sin(y); m[0][3] = 0.0;
    m[1][0] = 0.0   ; m[1][1] = 1.0; m[1][2] = 0.0    ; m[1][3] = 0.0;
    m[2][0] = np.sin(y); m[2][1] = 0.0; m[2][2] = np.cos(y) ; m[2][3] = 0.0;
    m[3][0] = 0.0   ; m[3][1] = 0.0; m[3][2] = 0.0    ; m[3][3] = 1.0;
    ry = m

    m = np.identity(4)
    m[0][0] = np.cos(z);  m[0][1] = np.sin(z);  m[0][2] = 0.0; m[0][3] = 0.0;
    m[1][0] = -np.sin(z); m[1][1] = np.cos(z) ; m[1][2] = 0.0; m[1][3] = 0.0;
    m[2][0] = 0.0   ;  m[2][1] = 0.0    ; m[2][2] = 1.0; m[2][3] = 0.0;
    m[3][0] = 0.0   ;  m[3][1] = 0.0    ; m[3][2] = 0.0; m[3][3] = 1.0;
    rz = m

    return rz @ ry @ rx


class Quaternion:
    def __init__(self, quat: np.ndarray):
        self.quat = quat

    @property
    def x(self):
        return self.quat[0]

    @property
    def y(self):
        return self.quat[1]

    @property
    def z(self):
        return self.quat[2]

    @property
    def w(self):
        return self.quat[3]

    @classmethod
    def from_angle_and_axis(cls, angle: float, axis: np.ndarray, degree=True):
        if degree:
            half_angle_rad = np.deg2rad(angle / 2)
        else:
            half_angle_rad = angle / 2
        sin_half_angle = np.sin(half_angle_rad)
        cos_half_angle = np.cos(half_angle_rad)
        return cls(np.array([
            axis[0] * sin_half_angle,
            axis[1] * sin_half_angle,
            axis[2] * sin_half_angle,
            cos_half_angle
        ]))

    @classmethod
    def from_xyzw(cls, x, y, z, w):
        return cls(np.array([x, y, z, w]))

    @classmethod
    def from_pyr(cls, pitch, yaw, roll, degree=True):
        if degree:
            pitch = np.deg2rad(pitch)
            yaw = np.deg2rad(yaw)
            roll = np.deg2rad(roll)
        sin_pitch = np.sin(pitch/2)
        cos_pitch = np.cos(pitch/2)
        sin_yaw = np.sin(yaw/2)
        cos_yaw = np.cos(yaw/2)
        sin_roll = np.sin(roll/2)
        cos_roll = np.cos(roll/2)
        cos_pitch_cos_yaw = cos_pitch * cos_yaw
        sin_pitch_sin_yaw = sin_pitch * sin_yaw
        x = sin_roll * cos_pitch_cos_yaw - cos_roll * sin_pitch_sin_yaw
        y = cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch + sin_yaw
        z = cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw
        w = cos_roll * cos_pitch_cos_yaw + sin_roll * sin_pitch_sin_yaw
        return cls(np.array([x, y, z, w]))

    def get_matrix(self, to_full=False):
        rotation = Rotation.from_quat(self.quat)
        matrix = rotation.as_matrix()
        if to_full:
            full_matrix = np.identity(4)
            full_matrix[:3, :3] = matrix
            matrix = full_matrix
        return matrix

    def normalize(self):
        self.quat = self.quat / np.linalg.norm(self.quat)

    def conjugate(self):
        return Quaternion(np.array([-self.quat[0], -self.quat[1], -self.quat[2], self.quat[3]]))

    def __mul__(l, r) -> 'Quaternion':
        if isinstance(r, Quaternion):
            w = (l.w * r.w) - (l.x * r.x) - (l.y * r.y) - (l.z * r.z)
            x = (l.x * r.w) + (l.w * r.x) + (l.y * r.z) - (l.z * r.y)
            y = (l.y * r.w) + (l.w * r.y) + (l.z * r.x) - (l.x * r.z)
            z = (l.z * r.w) + (l.w * r.z) + (l.x * r.y) - (l.y * r.x)
            return Quaternion.from_xyzw(x, y, z, w)
        elif isinstance(r, np.ndarray):
            q = l
            v = r
            w = - (q.x * v[0]) - (q.y * v[1]) - (q.z * v[2])
            x = (q.w * v[0]) + (q.y * v[2]) - (q.z * v[1])
            y = (q.w * v[1]) + (q.z * v[0]) - (q.x * v[2])
            z = (q.w * v[2]) + (q.x * v[1]) - (q.y * v[0])
            return Quaternion.from_xyzw(x, y, z, w)
        else:
            raise NotImplemented

    def to_degrees(self):
        x, y, z, w = self.x, self.y, self.z, self.w
        rads = np.array(
            [
                np.arctan2(x * z + y * w, x * w - y * z),
                np.arccos(-x * x - y * y - z * z - w * w),
                np.arctan2(x * z - y * w, x * w + y * z),
            ]
        )
        return np.rad2deg(rads)

    @staticmethod
    def interpolate(start: 'Quaternion', end: 'Quaternion', factor: float) -> 'Quaternion':
        slerp = Slerp([0, 1],
                      Rotation.from_quat(np.stack([start.quat, end.quat], axis=0)))
        interp_quat = slerp([factor]).as_quat()
        if len(interp_quat.shape) == 2:
            interp_quat = interp_quat[0]
        return Quaternion(interp_quat)


def rotate_vector(src: np.ndarray, angle: float, axis: np.ndarray):
    rotation_q = Quaternion.from_angle_and_axis(angle, axis)
    conjugate_q = rotation_q.conjugate()
    w = (rotation_q * src) * conjugate_q
    return np.array([w.x, w.y, w.z])


class Vector(np.ndarray):
    def __new__(cls, values=None):
        vec_len = cls.vec_len()
        if values is None:
            values = np.zeros(vec_len)
        assert len(values) == vec_len
        obj = np.array(values, dtype=cls.np_type()).view(cls)
        return obj

    @classmethod
    def vec_len(cls):
        raise NotImplementedError

    @classmethod
    def np_type(cls):
        raise NotImplementedError

    def normalize_(self):
        self /= np.linalg.norm(self)

    def cross(self, other: 'Vector') -> Self:
        return np.cross(self, other).view(type(self))

    def copy(self) -> Self:
        return super().copy().view(type(self))
    

class Vector2(Vector):
    @classmethod
    def vec_len(cls):
        return 2

    @property
    def x(self):
        return self[0]

    @x.setter
    def x(self, value):
        self[0] = value

    @property
    def y(self):
        return self[1]

    @y.setter
    def y(self, value):
        self[1] = value


class Vector3(Vector2):
    @classmethod
    def vec_len(cls):
        return 3

    @property
    def z(self):
        return self[2]

    @z.setter
    def z(self, value):
        self[2] = value


class Vector2f(Vector2):
    @classmethod
    def np_type(cls):
        return np.float32


class Vector3f(Vector3):
    @classmethod
    def np_type(cls):
        return np.float32




