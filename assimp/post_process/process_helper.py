from assimp.mesh import AiMesh
from typing import Iterable, List, Any, Tuple, TypeVar
from assimp.math import Vector3f
import numpy as np

T = TypeVar('T')


def array_bounds(data: List[T])->Tuple[T, T]:
    min_dat = min(data)
    max_dat = max(data)
    return min_dat, max_dat


def compute_position_epsilon(mesh: AiMesh):
    epsilon = 1e-4
    min_vec, max_vec = array_bounds(mesh.vertices)
    return np.linalg.norm(max_vec-min_vec) * epsilon



