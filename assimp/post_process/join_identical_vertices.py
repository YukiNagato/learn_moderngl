import numpy as np
from typing import List
from assimp.post_process.post_process import BaseProcess, PostProcessSteps
from assimp.scene import AiScene, AiMesh
from assimp.vertex import Vertex as Vertex_


class Vertex(Vertex_):
    def __eq__(self, o: 'Vertex'):
        for key in ['position', 'normal', 'tangent', 'bitangent']:
            if not np.allclose(getattr(self, key), getattr(o, key)):
                return False

        def all_list_close(list_a: List[np.ndarray], list_b: List[np.ndarray]):
            for a, b in zip(list_a, list_b):
                if not np.allclose(a, b):
                    return False
            return True

        for key in ['texcoords', 'colors']:
            if not all_list_close(getattr(self, key), getattr(o, key)):
                return False
        
        return True
            
    def __hash__(self) -> int:
        return hash(self.position.tostring())


class JoinIdenticalVertices(BaseProcess):
    def is_active(self, flag: PostProcessSteps) -> bool:
        return PostProcessSteps.aiProcess_JoinIdenticalVertices in flag
    
    def execute(self, scene: AiScene):
        for a in range(scene.num_meshes):
            self.process_mesh(scene.mesh[a], a)

    def process_mesh(self, mesh: AiMesh, mesh_idx: int):
        pass



