import numpy as np
from assimp.math import Vector3f, Vector4f


class AiColor3D(Vector3f):
    @property
    def r(self):
        return self.x

    @r.setter
    def r(self, value):
        self.x = value

    @property
    def g(self):
        return self.y

    @g.setter
    def g(self, value):
        self.y = value

    @property
    def b(self):
        return self.z

    @b.setter
    def b(self, value):
        self.z = value

    def __eq__(self, other):
        return self.r == other.r and self.b == other.b and self.g == other.g

    def __ne__(self, other):
        return not self == other

    def is_black(self):
        epsilon = 10e-3
        return np.all(np.abs(self) < epsilon)


class AiColor4D(AiColor3D):
    @classmethod
    def vec_len(cls):
        return 4

    @property
    def a(self):
        return self[3]

    @a.setter
    def a(self, value):
        self[3] = value

    def __eq__(self, other):
        return self.r == other.r and self.b == other.b and self.g == other.g and self.a == other.a

