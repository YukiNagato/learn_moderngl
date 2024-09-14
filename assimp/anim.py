from enum import Enum, auto
import numpy as np
from dataclasses import dataclass, field
from assimp.math import Vector3f, Quaternion
from typing import List


class AiAnimInterpolation(Enum):
    aiAnimInterpolation_Step = auto()
    aiAnimInterpolation_Linear = auto()
    aiAnimInterpolation_Spherical_Linear = auto()
    aiAnimInterpolation_Cubic_Spline = auto()


@dataclass
class AiVectorKey:
    time: float = 0
    value: Vector3f = field(default_factory=Vector3f)
    interpolation: AiAnimInterpolation = AiAnimInterpolation.aiAnimInterpolation_Step


@dataclass
class AiMeshKey:
    time: float = 0
    value: Vector3f = field(default_factory=Vector3f)


@dataclass
class AiMeshMorphKey:
    time: float = 0
    values: List[int] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)


class AiAnimBehaviour(Enum):
    aiAnimBehaviour_DEFAULT = 0
    aiAnimBehaviour_CONSTANT = 1
    aiAnimBehaviour_LINEAR = 2
    aiAnimBehaviour_REPEAT = 3


@dataclass
class AiNodeAnim:
    node_name: str = ""
    position_keys: List[AiVectorKey] = field(default_factory=list)
    rotation_keys: List[Quaternion] = field(default_factory=list)
    scaling_keys: List[AiVectorKey] = field(default_factory=list)
    pre_state: AiAnimBehaviour = AiAnimBehaviour.aiAnimBehaviour_DEFAULT
    post_state: AiAnimBehaviour = AiAnimBehaviour.aiAnimBehaviour_DEFAULT


@dataclass
class AiMeshAnim:
    name: str = ""
    keys: List[AiMeshKey] = field(default_factory=list)


@dataclass
class AiMeshMorphAnim:
    name: str = ""
    keys: List[AiMeshMorphKey] = field(default_factory=list)


@dataclass
class AiAnimation:
    name: str = ""
    duration: float = -1
    ticks_per_second: float = 0
    channels: List[AiNodeAnim] = field(default_factory=list)
    mesh_channels: List[AiMeshAnim] = field(default_factory=list)
    morph_mesh_channels: List[AiMeshMorphAnim] = field(default_factory=list)
