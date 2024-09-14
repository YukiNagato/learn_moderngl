import numpy as np
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
from ogldev.lib.ogldev_math_3d import PersProjInfo, \
    init_camera_transform, \
    init_translation_transform, init_rotation_transform
from typing import Optional


@dataclass
class SimpleCamera:
    pos: np.ndarray = None
    target: np.ndarray = None
    up: np.ndarray = None


class Pipeline(object):
    def __init__(self):
        self._scale = np.ones(3)
        self._rotation = np.zeros(3)
        self._pos = np.zeros(3)
        self._pers_project_info = PersProjInfo()
        self._w_transformation: Optional[np.ndarray] = None
        self._wp_transformation: Optional[np.ndarray] = None
        self._v_transformation: Optional[np.ndarray] = None
        self._vp_transformation: Optional[np.ndarray] = None
        self._wvp_transformation: Optional[np.ndarray] = None
        self._proj_transformation: Optional[np.ndarray] = None
        self._camera = SimpleCamera()

    def set_scale(self, s0, s1=None, s2=None):
        if s1 is None:
            s1 = s0
        if s2 is None:
            s2 = s1

        self._scale = np.array((s0, s1, s2))

    def set_world_pos(self, pos):
        self._pos = np.array(pos)

    def set_rotate(self, angles, degrees=False):
        if degrees:
            angles = np.deg2rad(angles)
        self._rotation = np.array(angles)

    def set_camera(self, pos, target, up):
        self._camera.pos = np.array(pos)
        self._camera.target = np.array(target)
        self._camera.up = np.array(up)

    def set_perspective_projection(self, p: PersProjInfo):
        self._pers_project_info = p

    def get_world_trans(self):
        scale_matrix = np.diag(self._scale)
        rot_matrix = init_rotation_transform(*self._rotation.tolist(), degree=False)[:3,:3]
        full_matrix = np.identity(4)
        full_matrix[:3, :3] = rot_matrix @ scale_matrix
        full_matrix[:3, 3] = self._pos
        self._w_transformation = full_matrix
        return self._w_transformation

    def get_view_trans(self):
        camera_rotation_trans = init_camera_transform(self._camera.target, self._camera.up)
        camera_translation = init_translation_transform(-self._camera.pos)
        self._v_transformation = camera_rotation_trans @ camera_translation
        return self._v_transformation

    def get_proj_trans(self):
        self._proj_transformation = self._pers_project_info.to_matrix()
        return self._proj_transformation

    def get_vp_trans(self):
        self.get_view_trans()
        self.get_proj_trans()
        self._vp_transformation = self._proj_transformation @ self._v_transformation
        return self._vp_transformation

    def get_wp_trans(self):
        pers_proj_trans = self._pers_project_info.to_matrix()
        self.get_world_trans()
        self._wp_transformation = pers_proj_trans @ self._w_transformation
        return self._wp_transformation

    def get_wv_trans(self):
        self.get_world_trans()
        self.get_view_trans()
        return self._v_transformation @ self._w_transformation

    def get_wvp_trans(self):
        self.get_world_trans()
        self.get_vp_trans()
        self._wvp_transformation = self._vp_transformation @ self._w_transformation
        return self._wvp_transformation


