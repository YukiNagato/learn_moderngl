import numpy as np


class AiAABB:
    def __init__(self, min=None, max=None):
        if min is None:
            min = np.zeros(3, dtype=np.float32)
        if max is None:
            max = np.zeros(3, dtype=np.float32)
        self.min: np.ndarray = min
        self.max: np.ndarray = max

