from dataclasses import dataclass, field
from assimp.node import AiNode
from assimp.texture import AiTexture
from assimp.anim import AiAnimation
from assimp.material import AiMaterial
from assimp.light import AiLight
from assimp.camera import AiCamera
from assimp.mesh import AiMesh, AiSkeleton
from typing import List, Dict, Any


@dataclass
class AiSceneFlag:
    AI_SCENE_FLAGS_INCOMPLETE = False
    AI_SCENE_FLAGS_VALIDATED = False
    AI_SCENE_FLAGS_VALIDATION_WARNING = False
    AI_SCENE_FLAGS_NON_VERBOSE_FORMAT = False
    AI_SCENE_FLAGS_TERRAIN = False
    AI_SCENE_FLAGS_ALLOW_SHARED = False


@dataclass
class AiScene:
    name: str = ""
    flags: AiSceneFlag = field(default_factory=AiSceneFlag)
    root_node: AiNode = None
    meshes: List[AiMesh] = field(default_factory=list)
    materials: List[AiMaterial] = field(default_factory=list)
    textures: List[AiTexture] = field(default_factory=list)
    animations: List[AiAnimation] = field(default_factory=list)
    lights: List[AiLight] = field(default_factory=list)
    cameras: List[AiCamera] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    skeletons: List[AiSkeleton] = field(default_factory=list)

    @property
    def num_meshes(self):
        return len(self.meshes)

    @property
    def num_materials(self):
        return len(self.materials)

    def has_meshes(self):
        return len(self.meshes) > 0

    def has_materials(self):
        return len(self.materials) > 0

    def has_textures(self):
        return len(self.textures) > 0

    def has_animations(self):
        return len(self.animations) > 0

    def has_lights(self):
        return len(self.lights) > 0

    def has_cameras(self):
        return len(self.cameras) > 0

    def has_skeletons(self):
        return len(self.skeletons) > 0







