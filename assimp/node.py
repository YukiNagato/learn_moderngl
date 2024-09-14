from dataclasses import dataclass, field
import numpy as np
from typing import List
from assimp.common import make_identity_4x4


@dataclass
class AiNode:
    name: str = ""
    transformation: np.ndarray = field(default_factory=make_identity_4x4)
    parent: 'AiNode' = None
    children: List['AiNode'] = field(default_factory=list)
    meshes: List[int] = field(default_factory=list)

