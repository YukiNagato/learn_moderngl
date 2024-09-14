from enum import Enum, auto
import numpy as np
from dataclasses import dataclass, field
from assimp.math import Vector3f, Quaternion
from typing import List


@dataclass
class AiCamera:
    name: str = ""
    position: Vector3f = field(default_factory=Vector3f)
    up: Vector3f = field(default=Vector3f([0, 1, 0]))
    look_at: Vector3f = field(default=Vector3f([0, 0, 1]))
    horizontal_fov: float = 0.25 * np.pi
    clip_plane_near: float = 0.1
    clip_plane_far: float = 1000.0
    aspect: float = 0
    orthographic_width: float = 0

    def get_camera_matrix(self) -> np.ndarray:
        out = np.identity(4, dtype=np.float32)
        z_axis = self.look_at.copy()
        z_axis.normalize_()
        y_axis = self.up.copy()
        y_axis.normalize_()
        x_axis = self.up.cross(self.look_at)
        x_axis.normalize_()

        out[0, 3] = -np.prod(x_axis, self.position)
        out[0, 3] = -np.prod(y_axis, self.position)
        out[0, 3] = -np.prod(z_axis, self.position)

        out[0, 0] = x_axis.x
        out[0, 1] = x_axis.y
        out[0, 2] = x_axis.z

        out[1, 0] = y_axis.x
        out[1, 1] = y_axis.y
        out[1, 2] = y_axis.z

        out[2, 0] = z_axis.x
        out[2, 1] = z_axis.y
        out[2, 2] = z_axis.z

        return out




