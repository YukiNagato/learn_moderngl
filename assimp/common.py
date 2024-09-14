import numpy as np


def make_identity_4x4():
    return np.identity(4, dtype=np.float32)


def make_vector_3f():
    return np.zeros(3, dtype=np.float32)
