from enum import IntFlag

import numpy as np
from typing import List, Optional, Dict, Tuple
from assimp.post_process.post_process import BaseProcess, PostProcessSteps
from assimp.scene import AiScene, AiMesh
from assimp.mesh import AiPrimitiveState, AiFace
from assimp.math import Vector2f, Vector3f, Vector4f
from assimp.color import AiColor4D
from assimp.utils.list_pointer import ListPointer
from scipy.spatial import cKDTree
from assimp.post_process.process_helper import compute_position_epsilon


class GenVertexNormalsProcess(BaseProcess):
    def __init__(self):
        self.config_max_angle = np.deg2rad(175.0)
        self.force = False
        self.flipped_winding_order = False
        self.left_handed = False

    def is_active(self, flag: IntFlag) ->bool:
        self.force = PostProcessSteps.aiProcess_ForceGenNormals in flag
        self.flipped_winding_order = PostProcessSteps.aiProcess_FlipWindingOrder in flag
        self.left_handed = PostProcessSteps.aiProcess_MakeLeftHanded in flag
        return PostProcessSteps.aiProcess_GenSmoothNormals in flag

    def execute(self, scene: AiScene):

        if scene.flags.AI_SCENE_FLAGS_NON_VERBOSE_FORMAT:
            raise ImportError("Post-processing order mismatch: expecting pseudo-indexed (\"verbose\") vertices here")

        b_has = False

        for a in range(scene.num_meshes):
            if self.gen_mesh_vertex_normals(scene.meshes[a], a):
                b_has = True

        if b_has:
            print("GenVertexNormalsProcess finished. Vertex normals have been calculated")
        else:
            print("GenVertexNormalsProcess finished. Normals are already there")

    def gen_mesh_vertex_normals(self, mesh: AiMesh, mesh_index: int)->bool:
        if len(mesh.normals) > 0:
            if not self.force:
                return False
            mesh.normals = []

        if not (mesh.primitive_types.triangle or mesh.primitive_types.polygon):
            return False

        qnan = [np.nan, np.nan, np.nan]
        mesh.normals = [Vector3f() for _ in range(mesh.num_vertices)]

        for a in range(mesh.num_faces):
            face = mesh.faces[a]
            if face.num_indices < 3:
                for i in range(face.num_indices):
                    mesh.normals[i] = Vector3f(qnan)
                continue

            v1 = mesh.vertices[face.indices[0]]
            v2 = mesh.vertices[face.indices[1]]
            v3 = mesh.vertices[face.indices[face.num_indices-1]]

            if self.flipped_winding_order != self.left_handed:
                v2, v3 = v3, v2

            v_nor = (v2-v1).cross(v3-v1).normalize_safe_()

            for i in range(face.num_indices):
                mesh.normals[face.indices[i]] = v_nor.copy()


        vertex_finder = cKDTree(mesh.vertices)
        pos_epsilon = compute_position_epsilon(mesh)

        pc_new = [Vector3f() for _ in range(mesh.num_vertices)]

        if self.config_max_angle >= np.deg2rad(175.0):
            ab_had = [False for _ in range(mesh.num_vertices)]
            for i in range(mesh.num_vertices):
                if ab_had[i]:
                    continue
                vertices_found = vertex_finder.query_ball_point(mesh.vertices[i], pos_epsilon, p=2)

                pc_nor = Vector3f()
                for a in range(vertices_found.shape[0]):
                    v = mesh.normals[vertices_found[a]]
                    if not np.isnan(v.x):
                        pc_nor += v
                pc_nor.normalize_safe_()

                for a in range(vertices_found.shape[0]):
                    vidx = vertices_found[a]
                    pc_new[vidx] = pc_nor.copy()
                    ab_had[vidx] = True
        else:
            raise NotImplemented

        mesh.normals = pc_new
        return True
