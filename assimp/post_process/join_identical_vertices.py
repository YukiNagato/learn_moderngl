import numpy as np
from typing import List, Optional, Dict, Tuple
from assimp.post_process.post_process import BaseProcess, PostProcessSteps
from assimp.scene import AiScene, AiMesh
from assimp.mesh import AiAnimMesh, AiVertexWeight
from assimp.vertex import Vertex as Vertex_
from scipy.spatial import KDTree


class Vertex(Vertex_):
    def __eq__(self, o: 'Vertex'):
        # if not np.allclose(self.data, o.data):
        #     return False
        if not np.all(np.abs(self.data-o.data) < 1e-5):
            return False

        return True

    def __hash__(self) -> int:
        return hash(self.position.tostring())


def update_xmesh_vertices(mesh: AiMesh | AiAnimMesh, unique_vertices: List[Vertex]):
    if mesh.num_vertices > 0:
        mesh.vertices = [vertex.position for vertex in unique_vertices]

    if mesh.has_normals():
        mesh.normals = [vertex.normal for vertex in unique_vertices]

    if len(mesh.tangents) > 0:
        mesh.tangents = [vertex.tangent for vertex in unique_vertices]

    if len(mesh.bi_tangents) > 0:
        mesh.bi_tangents = [vertex.bi_tangent for vertex in unique_vertices]

    a = 0
    while mesh.has_vertex_colors(a):
        mesh.colors[a] = [vertex.colors[a] for vertex in unique_vertices]
        a += 1

    a = 0
    while mesh.has_texture_coords(a):
        mesh.texture_coords[a] = [vertex.tex_coords[a] for vertex in unique_vertices]
        a += 1


class JoinVerticesProcess(BaseProcess):
    def is_active(self, flag: PostProcessSteps) -> bool:
        return PostProcessSteps.aiProcess_JoinIdenticalVertices in flag
    
    def execute(self, scene: AiScene):
        for a in range(scene.num_meshes):
            self.process_mesh(scene.meshes[a], a)

    def process_mesh(self, mesh: AiMesh, mesh_idx: int):

        if not mesh.has_positions() or not mesh.has_normals():
            return

        used_vertex_indices = set()

        for a in range(mesh.num_faces):
            face = mesh.faces[a]
            for b in range(face.num_indices):
                used_vertex_indices.add(face.indices[b])

        unique_vertices: List[Vertex] = []

        replace_idxs = [-1 for _ in range(mesh.num_vertices)]

        has_anim_meshes = mesh.num_anim_meshes > 0
        complex = mesh.get_num_uv_channels() > 1 or mesh.get_num_color_channels() > 0
        assert not complex

        unique_animated_vertices = [[] for _ in range(mesh.num_anim_meshes)]

        spatial_finder: Dict[Vertex, int] = dict()
        new_idx = 0
        for a in range(mesh.num_vertices):
            if a not in used_vertex_indices:
                continue
            v = Vertex.from_mesh(mesh, a)
            founded_idx = spatial_finder.get(v, None)
            if founded_idx is None:
                spatial_finder[v] = new_idx
                replace_idxs[a] = new_idx
                new_idx += 1
                unique_vertices.append(v)
                if has_anim_meshes:
                    for anim_mesh_index in range(mesh.num_anim_meshes):
                        ani_mesh_vertex = Vertex.from_anim_mesh(mesh.anim_meshes[anim_mesh_index], a)
                        unique_animated_vertices[anim_mesh_index].append(ani_mesh_vertex)
            else:
                replace_idxs[a] = founded_idx

        update_xmesh_vertices(mesh, unique_vertices)
        if has_anim_meshes:
            for anim_mesh_index in range(mesh.num_anim_meshes):
                update_xmesh_vertices(mesh.anim_meshes[anim_mesh_index],
                                      unique_animated_vertices[anim_mesh_index])

        for a in range(mesh.num_faces):
            face = mesh.faces[a]
            for b in range(face.num_indices):
                new_index = replace_idxs[face.indices[b]]
                assert new_index != -1
                face.indices[b] = new_index

        for a in range(mesh.num_bones):
            bone = mesh.bones[a]
            new_weights: List[AiVertexWeight] = []

            if len(bone.weights) > 0:
                for b in range(bone.num_weights):
                    ow = bone.weights[b]
                    if replace_idxs[ow.vertex_id] != -1:
                        nw = AiVertexWeight()
                        nw.vertex_id = replace_idxs[ow.vertex_id]
                        nw.weight = ow.weight
                        new_weights.append(nw)

            if len(new_weights) > 0:
                bone.weights = new_weights


