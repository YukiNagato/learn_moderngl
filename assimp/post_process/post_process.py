from enum import IntFlag, auto
from assimp.scene import AiScene


class PostProcessSteps(IntFlag):
    aiProcess_JoinIdenticalVertices = 0x2
    aiProcess_Triangulate = 0x8
    aiProcess_GenSmoothNormals = 0x40



class BaseProcess:
    def is_active(self, flag: PostProcessSteps)->bool:
        raise NotImplementedError
    
    def execute(self, scene: AiScene):
        raise NotImplementedError
    

