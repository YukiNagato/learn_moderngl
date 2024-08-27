import numpy as np
from dataclasses import dataclass


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
    def from_angle_and_axis(cls, angle: float, axis: np.ndarray):
        half_angle_rad = np.deg2rad(angle / 2)
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


def rotate_vector(src: np.ndarray, angle: float, axis: np.ndarray):
    rotation_q = Quaternion.from_angle_and_axis(angle, axis)
    conjugate_q = rotation_q.conjugate()
    w = (rotation_q * src) * conjugate_q
    return np.array([w.x, w.y, w.z])



