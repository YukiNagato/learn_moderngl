from enum import Enum, auto
import numpy as np
from dataclasses import dataclass, field
from assimp.math import Vector2f, Vector3f, Quaternion
from assimp.color import AiColor3D
from typing import List


class AiLightSourceType(Enum):
    aiLightSource_UNDEFINED     = 0x0,
    aiLightSource_DIRECTIONAL   = 0x1,
    aiLightSource_POINT         = 0x2,
    aiLightSource_SPOT          = 0x3,
    aiLightSource_AMBIENT       = 0x4,
    aiLightSource_AREA          = 0x5,




@dataclass
class AiLight:
    name: str = ""
    type: AiLightSourceType = AiLightSourceType.aiLightSource_UNDEFINED
    position: Vector3f = field(default_factory=Vector3f)
    direction: Vector3f = field(default_factory=Vector3f)
    up: Vector3f = field(default_factory=Vector3f)
    attenuation_constant: float = 0
    attenuation_linear: float = 1
    attenuation_quadratic: float = 0
    color_diffuse: AiColor3D = field(default_factory=AiColor3D)
    color_specular: AiColor3D = field(default_factory=AiColor3D)
    color_ambient: AiColor3D = field(default_factory=AiColor3D)
    angle_inner_cone: float = np.pi * 2
    angle_outer_cone: float = np.pi * 2
    size: Vector2f = field(default_factory=Vector2f)


