import numpy as np
from typing import List, Optional, Dict, Tuple
from assimp.post_process.post_process import BaseProcess, PostProcessSteps
from assimp.scene import AiScene, AiMesh
from assimp.mesh import AiPrimitiveState, AiFace
from assimp.math import Vector2f, Vector3f, Vector4f
from assimp.color import AiColor4D
from assimp.utils.list_pointer import ListPointer


class NGONEncoder:
    def __init__(self):
        self.last_ngon_first_index = -1

    def is_considered_same_as_last_ngon(self, tri: AiFace) -> bool:
        assert tri.num_indices == 3
        return tri.indices[0] == self.last_ngon_first_index

    def ngon_encode_triangle(self, tri: AiFace):
        assert tri.num_indices == 3
        if self.is_considered_same_as_last_ngon(tri):
            tri.indices[0], tri.indices[2] = tri.indices[2], tri.indices[0]
            tri.indices[1], tri.indices[2] = tri.indices[2], tri.indices[1]
        self.last_ngon_first_index = tri.indices[0]

    def ngon_encode_quad(self, tri1: AiFace, tri2: AiFace):
        assert tri1.num_indices == 3
        assert tri2.num_indices == 3
        assert tri1.indices[0] == tri2.indices[0]

        if self.is_considered_same_as_last_ngon(tri1):
            tri1.indices[0], tri1.indices[2] = tri1.indices[2], tri1.indices[0]
            tri1.indices[1], tri1.indices[2] = tri1.indices[2], tri1.indices[1]

            tri2.indices[1], tri2.indices[2] = tri2.indices[2], tri2.indices[1]
            tri2.indices[0], tri2.indices[2] = tri2.indices[2], tri2.indices[0]

            assert tri1.indices[0] == tri2.indices[0]
        self.last_ngon_first_index = tri1.indices[0]


class TriangulateProcess(BaseProcess):
    def is_active(self, flag: PostProcessSteps) -> bool:
        return PostProcessSteps.aiProcess_Triangulate in flag

    def execute(self, scene: AiScene):
        for a in range(scene.num_meshes):
            self.triangulate_mesh(scene.meshes[a])

    def triangulate_mesh(self, mesh: AiMesh):
        if mesh.primitive_types == AiPrimitiveState.polygon:
            return False

        num_out = 0
        max_out = 0
        get_normals = True

        for a in range(mesh.num_faces):
            face = mesh.faces[a]
            if face.num_indices <= 4:
                get_normals = False
            if face.num_indices <= 3:
                num_out += 1
            else:
                num_out += (face.num_indices - 2)
                max_out = max(max_out, face.num_indices)

        if num_out == mesh.num_faces:
            return False

        mesh.primitive_types.triangle = True
        mesh.primitive_types.polygon = False
        mesh.primitive_types.ngon_encoding = True

        out = [AiFace() for _ in range(num_out)]
        cur_out = ListPointer(out)
        temp_verts3d = [Vector3f() for _ in range(max_out + 2)]
        temp_verts = [Vector2f() for _ in range(max_out + 2)]

        ngon_encoder = NGONEncoder()
        verts = mesh.vertices

        done = [False for _ in range(max_out)]
        for a in range(mesh.num_faces):
            face = mesh.faces[a]
            idx = face.indices
            num = face.num_indices
            ear = 0
            tmp = 0
            prev = num - 1
            next_ = 0
            max_ = num

            last_face = cur_out.new()

            if face.num_indices <= 3:
                nface: AiFace = cur_out.get()
                cur_out += 1
                nface.indices = face.indices
                face.indices = []

                if nface.num_indices == 3:
                    ngon_encoder.ngon_encode_triangle(nface)

                continue

            elif face.num_indices == 4:
                start_vertex = 0
                for i in range(4):
                    v0 = verts[face.indices[i + 3] % 4]
                    v1 = verts[face.indices[i + 2] % 4]
                    v2 = verts[face.indices[i + 1] % 4]
                    v = verts[face.indices[i]]

                    left = v0 - v
                    diag = v1 - v
                    right = v2 - v

                    left.normalize_()
                    diag.normalize_()
                    right.normalize_()

                    angle = np.arccos(left * diag) + np.arccos(right * diag)
                    if angle > np.pi:
                        start_vertex = i
                        break

                temp = [face.indices[0], face.indices[1], face.indices[2], face.indices[3]]
                nface: AiFace = cur_out.rinc()
                nface.indices = face.indices

                nface.indices[0] = temp[start_vertex]
                nface.indices[1] = temp[(start_vertex + 1) % 4]
                nface.indices[2] = temp[(start_vertex + 2) % 4]

                sface: AiFace = cur_out.rinc()
                sface.indices = [0, 0, 0]
                sface.indices[0] = temp[start_vertex]
                sface.indices[1] = temp[(start_vertex + 2) % 4]
                sface.indices[2] = temp[(start_vertex + 3) % 4]
                face.indices = []
                ngon_encoder.ngon_encode_quad(nface, sface)

                continue

            else:
                raise NotImplementedError

            while last_face.get() is not cur_out.get():
                f: AiFace = last_face.get()
                i = f.indices

                i[0] = idx[i[0]]
                i[1] = idx[i[1]]
                i[2] = idx[i[2]]
                ngon_encoder.ngon_encode_triangle(f)
                last_face += 1

            face.indices = []

        mesh.faces = out
        return True

