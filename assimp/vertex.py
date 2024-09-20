import numpy as np
from dataclasses import dataclass, field
from assimp.math import Vector3f, Vector4f
from assimp.color import AiColor4D
from typing import List


@dataclass
class Vertex:
    position: Vector3f = field(default_factory=Vector3f)
    normal: Vector3f = field(default_factory=Vector3f)
    tangent: Vector3f = field(default_factory=Vector3f)
    bitangent: Vector3f = field(default_factory=Vector3f)
    texcoords: List[Vector3f] = field(default_factory=list)
    colors: List[AiColor4D] = field(default_factory=list)

