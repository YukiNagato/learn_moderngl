import numpy as np
from dataclasses import dataclass, field
from assimp.math import Vector3f, Vector4f
from assimp.color import AiColor4D
from assimp.mesh import AiMesh, AiAnimMesh
from typing import List


# @dataclass
# class Vertex:
#     position: Vector3f = field(default_factory=Vector3f)
#     normal: Vector3f = field(default_factory=Vector3f)
#     tangent: Vector3f = field(default_factory=Vector3f)
#     bi_tangent: Vector3f = field(default_factory=Vector3f)
#     tex_coords: List[Vector3f] = field(default_factory=list)
#     colors: List[AiColor4D] = field(default_factory=list)
#
#     @classmethod
#     def from_mesh(cls, mesh: AiMesh, idx: int) -> 'Vertex':
#         assert idx < mesh.num_vertices
#         vertex = cls()
#         vertex.position = mesh.vertices[idx].copy()
#         if mesh.has_normals():
#             vertex.normal = mesh.normals[idx].copy()
#
#         if mesh.has_tangents_and_bi_tangents():
#             vertex.tangent = mesh.tangents[idx].copy()
#             vertex.bi_tangent = mesh.bi_tangents[idx].copy()
#
#         i = 0
#         while mesh.has_texture_coords(i):
#             vertex.tex_coords.append(mesh.texture_coords[i][idx])
#             i += 1
#
#         i = 0
#         while mesh.has_vertex_colors(i):
#             vertex.colors.append(mesh.colors[i][idx])
#             i += 1
#
#         return vertex
#
#
#     @classmethod
#     def from_anim_mesh(cls, mesh: AiAnimMesh, idx: int) -> 'Vertex':
#         assert idx < mesh.num_vertices
#         vertex = Vertex()
#         vertex.position = mesh.vertices[idx].copy()
#         if mesh.has_normals():
#             vertex.normal = mesh.normals[idx].copy()
#
#         if mesh.has_tangents_and_bi_tangents():
#             vertex.tangent = mesh.tangents[idx].copy()
#             vertex.bi_tangent = mesh.bi_tangents[idx].copy()
#
#         i = 0
#         while mesh.has_texture_coords(i):
#             vertex.tex_coords.append(mesh.texture_coords[i][idx])
#
#         i = 0
#         while mesh.has_vertex_colors(i):
#             vertex.colors.append(mesh.colors[i][idx])
#
#         return vertex


class Vertex:
    def __init__(self, tex_coords_cnt, colors_cnt):
        self.tex_coords_cnt = tex_coords_cnt
        self.colors_cnt = colors_cnt
        self.data = np.zeros(4*3+self.tex_coords_cnt*3+self.colors_cnt*4, dtype=np.float32)

        class ListView:
            def __init__(self, data, start, data_size, data_num, dtype):
                self.data = data
                self.start = start
                self.data_size = data_size
                self.data_num = data_num
                self.dtype = dtype

            def __getitem__(self, idx):
                return self.data[self.start + idx*self.data_size:self.start + self.data_size * (idx+1)].view(self.dtype)

            def __setitem__(self, idx, value):
                self.data[self.start + idx*self.data_size:self.start + self.data_size * (idx+1)] = value

            def __len__(self):
                return self.data_num

        self.tex_coords_view = ListView(self.data, 12, 3, self.tex_coords_cnt, Vector3f)
        self.colors_view = ListView(self.data, 12 + 3*self.tex_coords_cnt, 4, self.colors_cnt, AiColor4D)

    @property
    def position(self):
        return self.data[0:3].view(Vector3f)

    @position.setter
    def position(self, value):
        self.data[0:3] = value

    @property
    def normal(self):
        return self.data[3:6].view(Vector3f)

    @normal.setter
    def normal(self, value):
        self.data[3:6] = value

    @property
    def tangent(self):
        return self.data[6:9].view(Vector3f)

    @tangent.setter
    def tangent(self, value):
        self.data[6:9] = value

    @property
    def bi_tangent(self):
        return self.data[9:12].view(Vector3f)

    @bi_tangent.setter
    def bi_tangent(self, value):
        self.data[9:12] = value

    @property
    def tex_coords(self):
        return self.tex_coords_view

    @property
    def colors(self):
        return self.colors_view

    @classmethod
    def from_mesh(cls, mesh: AiMesh, idx: int) -> 'Vertex':
        assert idx < mesh.num_vertices

        vertex_colors_cnt = mesh.vertex_colors_cnt
        texture_coords_cnt = mesh.texture_coords_cnt
        vertex = cls(texture_coords_cnt, vertex_colors_cnt)
        vertex.position = mesh.vertices[idx].copy()
        if mesh.has_normals():
            vertex.normal = mesh.normals[idx].copy()

        if mesh.has_tangents_and_bi_tangents():
            vertex.tangent = mesh.tangents[idx].copy()
            vertex.bi_tangent = mesh.bi_tangents[idx].copy()

        i = 0
        while mesh.has_texture_coords(i):
            vertex.tex_coords[i] = mesh.texture_coords[i][idx]
            i += 1

        i = 0
        while mesh.has_vertex_colors(i):
            vertex.colors[i] = mesh.colors[i][idx]
            i += 1

        return vertex

    @classmethod
    def from_anim_mesh(cls, mesh: AiAnimMesh, idx: int) -> 'Vertex':
        assert idx < mesh.num_vertices
        vertex_colors_cnt = mesh.vertex_colors_cnt
        texture_coords_cnt = mesh.texture_coords_cnt
        vertex = cls(texture_coords_cnt, vertex_colors_cnt)
        vertex.position = mesh.vertices[idx].copy()
        if mesh.has_normals():
            vertex.normal = mesh.normals[idx].copy()

        if mesh.has_tangents_and_bi_tangents():
            vertex.tangent = mesh.tangents[idx].copy()
            vertex.bi_tangent = mesh.bi_tangents[idx].copy()

        i = 0
        while mesh.has_texture_coords(i):
            vertex.tex_coords[i] = mesh.texture_coords[i][idx]
            i += 1

        i = 0
        while mesh.has_vertex_colors(i):
            vertex.colors[i] = mesh.colors[i][idx]
            i += 1

        return vertex


