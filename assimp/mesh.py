from typing import List, Optional

from assimp.node import AiNode
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from assimp.common import make_identity_4x4
from copy import deepcopy
from assimp.aabb import AiAABB
from assimp.math import Vector2f, Vector3f, Quaternion
from assimp.color import AiColor4D, AiColor3D


AI_MAX_NUMBER_OF_COLOR_SETS = 0x8
AI_MAX_NUMBER_OF_TEXTURECOORDS = 0x8


class AiMorphingMethod(Enum):
    aiMorphingMethod_UNKNOWN = 0x0
    aiMorphingMethod_VERTEX_BLEND  = 0x1
    aiMorphingMethod_MORPH_NORMALIZED = 0x2
    aiMorphingMethod_MORPH_RELATIVE = 0x3


class AiPrimitiveType(Enum):
    aiPrimitiveType_POINT = 0x1
    aiPrimitiveType_LINE = 0x2
    aiPrimitiveType_TRIANGLE = 0x4
    aiPrimitiveType_POLYGON = 0x8
    aiPrimitiveType_NGONEncodingFlag = 0x10


@dataclass
class AiFace:
    indices: List[int] = field(default_factory=list)

    @property
    def num_indices(self) -> int:
        return len(self.indices)


@dataclass
class AiPrimitiveState:
    point: bool = False
    line: bool = False
    triangle: bool = False
    polygon: bool = False
    ngon_encoding: bool = False


@dataclass
class AiVertexWeight:
    vertex_id: int = 0
    weight: float = 0

    def __eq__(self, other):
        return self.vertex_id == other.vertex_id and self.weight == other.weight


@dataclass
class AiBone:
    name: str = ''
    weights: List[AiVertexWeight] = field(default_factory=list)
    armature: AiNode = None
    node: AiNode = None
    offset_matrix: np.ndarray = field(default_factory=make_identity_4x4)

    def copy_vertex_weight(self, other: 'AiBone'):
        self.weights = deepcopy(other.weights)

    def __eq__(self, other: 'AiBone'):
        if self.name != other.name or len(self.weights) != len(other.weights):
            return False

        for i in range(len(self.weights)):
            if self.weights[i] != other.weights[i]:
                return False
        return True


@dataclass
class AiAnimMesh:
    name: str = ''
    vertices: List[np.ndarray] = field(default_factory=list)
    normals: List[np.ndarray] = field(default_factory=list)
    tangents: List[np.ndarray] = field(default_factory=list)
    colors: List[np.ndarray] = field(default_factory=list)
    texture_coords: List[np.ndarray] = field(default_factory=list)
    weight: float = 0


@dataclass
class AiMesh:
    name: str = ''
    primitive_types: AiPrimitiveState = field(default_factory=AiPrimitiveState)
    vertices: List[Vector3f] = field(default_factory=list)
    normals: List[Vector3f] = field(default_factory=list)
    tangents: List[Vector3f] = field(default_factory=list)
    bi_tangents: List[Vector3f] = field(default_factory=list)
    colors: List[List[Vector3f]] = field(default_factory=list)
    texture_coords: List[List[Vector3f]] = field(default_factory=list)
    num_uv_components: List[int] = field(default_factory=list)
    faces: List[AiFace] = field(default_factory=list)
    bones: List[AiBone] = field(default_factory=list)
    material_index: int = 0
    anim_meshes: List[AiAnimMesh] = field(default_factory=list)
    method: AiMorphingMethod = AiMorphingMethod.aiMorphingMethod_UNKNOWN
    aabb: AiAABB = field(default_factory=AiAABB)
    texture_coords_names: List[Optional[str]] = field(default_factory=list)

    @property
    def num_vertices(self):
        return len(self.vertices)

    @property
    def num_faces(self):
        return len(self.faces)

    def has_positions(self):
        return len(self.vertices) > 0

    def has_faces(self):
        return len(self.faces) > 0

    def has_normals(self):
        return len(self.normals) > 0

    def has_bones(self):
        return len(self.bones) > 0

    def has_tangents_and_bi_tangents(self):
        return len(self.tangents) > 0 and len(self.bi_tangents) > 0

    def has_vertex_colors(self, index):
        return len(self.colors) > index and self.colors[index] is not None and self.has_positions()

    def has_texture_coords(self, index):
        return len(self.texture_coords) > index and self.texture_coords[index] is not None and self.has_positions()

    def get_num_uv_channels(self):
        n = 0
        for coords in self.texture_coords:
            if coords is not None:
                n += 1
        return n

    def get_num_color_channels(self):
        n = 0
        for colors in self.colors:
            if colors is not None:
                n += 1
        return n

    def has_texture_coords_name(self, index):
        return len(self.texture_coords_names) > index and self.texture_coords_names[index]

    def set_texture_coords_name(self, index, value: str):
        if index > AI_MAX_NUMBER_OF_TEXTURECOORDS:
            return

        if len(self.texture_coords_names) == 0:
            self.texture_coords_names = [None for _ in range(AI_MAX_NUMBER_OF_TEXTURECOORDS)]

        if not value:
            self.texture_coords_names[index] = None

        else:
            self.texture_coords_names[index] = value

    def get_texture_coords_name(self, index):
        if index > len(self.texture_coords_names):
            return None

        return self.texture_coords_names[index]


@dataclass
class AiSkeletonBone:
    parent: int = 0
    armature: AiNode = None
    node: AiNode = None
    mesh_id: AiMesh = None
    weights: List[AiVertexWeight] = field(default_factory=list)
    offset_matrix: np.ndarray = field(default_factory=make_identity_4x4)
    local_matrix: np.ndarray = field(default_factory=make_identity_4x4)


@dataclass
class AiSkeleton:
    name: str = ''
    bones: List[AiBone] = field(default_factory=list)











