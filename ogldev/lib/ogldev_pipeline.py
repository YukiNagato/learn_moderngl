import numpy as np
from scipy.spatial.transform import Rotation


class Pipeline(object):
    def __init__(self):
        self._scale = np.ones(3)
        self._rotation = np.zeros(3)
        self._pos = np.zeros(3)

    def scale(self, s0, s1=None, s2=None):
        if s1 is None:
            s1 = s0
        if s2 is None:
            s2 = s1

        self._scale = np.array((s0, s1, s2))

    def world_pos(self, pos):
        self._pos = np.array(pos)

    def rotate(self, angles, degrees=False):
        if degrees:
            angles = np.deg2rad(angles)
        self._rotation = np.array(angles)

    def get_world_trans(self):
        scale_matrix = np.diag(self._scale)
        rot_matrix = Rotation.from_euler('xyz', self._rotation).as_matrix()
        full_matrix = np.identity(4)
        full_matrix[:3, :3] = scale_matrix @ rot_matrix
        full_matrix[:3, 3] = self._pos
        return full_matrix




