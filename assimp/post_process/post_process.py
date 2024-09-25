from enum import IntFlag, auto
from assimp.scene import AiScene


class PostProcessSteps(IntFlag):
    aiProcess_JoinIdenticalVertices = 0x2
    aiProcess_MakeLeftHanded = 0x4
    aiProcess_Triangulate = 0x8
    aiProcess_GenSmoothNormals = 0x40

    aiProcess_FlipWindingOrder = 0x1000000

    aiProcess_ForceGenNormals = 0x20000000



class BaseProcess:
    def is_active(self, flag: IntFlag)->bool:
        raise NotImplementedError
    
    def execute(self, scene: AiScene):
        raise NotImplementedError
    

