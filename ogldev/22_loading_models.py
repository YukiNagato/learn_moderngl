import math
import os

import moderngl
import pygame
import numpy as np
import glm
from ogldev.lib.ogldev_pipeline import Pipeline
from ogldev.lib.ogldev_math_3d import PersProjInfo
from ogldev.lib.ogldev_camera import Camera
from PIL import Image
from ogldev.lib.ogldev_light import DirectionalLight, PointLight, SpotLight
from ogldev.lib.ogldev_math_3d import Vector3f, Vector2f, Vector
from ogldev.lib.ogldev_callbacks import ICallbacks
from ogldev.lib.pygame_backend import PyGameBackend
from typing import List
from assimp.asset_lib.md2 import Md2
from assimp.scene import AiScene, AiMesh, AiMaterial
from assimp.material import AiTextureType
from ogldev.lib.ogldev_basic_lighting import BasicLightingTechnique
from ogldev.lib import DATA_DIR


class ImageTexture:
    def __init__(self, path):
        self.ctx = moderngl.get_context()

        img = Image.open(path).convert('RGBA')
        self.texture = self.ctx.texture(img.size, 4, img.tobytes())
        self.sampler = self.ctx.sampler(texture=self.texture)

    def use(self):
        self.sampler.use()


class Vertex(np.ndarray):
    def __new__(cls, pos=None, tex=None, normal=None):
        values = np.zeros(8)
        if pos is not None:
            values[:3] = np.array(pos)
        if tex is not None:
            values[3:5] = np.array(tex)
        if normal is not None:
            values[5:8] = np.array(normal)

        obj = np.array(values, dtype=np.float32).view(cls)
        return obj

    @property
    def pos(self):
        return self[:3].view(Vector3f)

    @property
    def tex(self):
        return self[3:5].view(Vector3f)

    @property
    def normal(self):
        return self[5:8].view(Vector3f)

    @normal.setter
    def normal(self, value):
        self[5:8] = value


class MeshEntry:
    def __init__(self):
        self.ctx = moderngl.get_context()
        self.vbo = None
        self.ibo = None
        self.material_index = 0

    def init(self, vertices: List[Vertex], indices: List[int]):
        # dst_idx = 800
        # vertices = vertices[dst_idx*3:dst_idx*3+3]
        self.vertices = vertices
        vertices = np.ascontiguousarray(np.stack(vertices).astype('f4').flatten())
        self.vbo = self.ctx.buffer(vertices.tobytes())
        indices = np.array(indices, dtype='i4')
        self.ibo = None
        if len(indices) > 0:
            self.ibo = self.ctx.buffer(indices)

    def vertex_array(self, program):
        self.vao = self.ctx.vertex_array(program, [(self.vbo, '3f 2f 3f', 'Position', 'TexCoord', 'Normal')], self.ibo)

    def render(self):
        self.vao.render()


class Mesh:
    def __init__(self, program):
        self.entries = []
        self.textures = []
        self.program = program
        self.transform = np.identity(4)

    def clear(self):
        self.entries = []
        self.textures = []

    def load_mesh(self, filename: str):
        scene = Md2().read(filename)
        self.init_from_scene(scene, filename)

    def init_from_scene(self, scene: AiScene, filename: str):
        for i in range(scene.num_meshes):
            mesh = scene.meshes[i]
            entry = self.init_mesh(mesh)
            self.entries.append(entry)
        self.init_materials(scene, filename)
        self.transform = scene.root_node.transformation

    def init_materials(self, scene: AiScene, filename: str):
        folder = os.path.dirname(filename)
        for i in range(scene.num_materials):
            material = scene.materials[i]
            texture = None
            if material.get_texture_count(AiTextureType.aiTextureType_DIFFUSE) > 0:
                material_texture_ret = material.get_texture(
                    AiTextureType.aiTextureType_DIFFUSE,
                    0,
                )

                full_path = os.path.join(folder, material_texture_ret.path)
                texture = ImageTexture(full_path)

            if texture is None:
                texture = ImageTexture('ogldev/data/white.png')
            self.textures.append(texture)

    def init_mesh(self, mesh: AiMesh):
        entry = MeshEntry()
        entry.material_index = mesh.material_index

        vertices: List[Vertex] = []
        indices: List[int] = []

        zero_3d = Vector3f()
        for i in range(mesh.num_vertices):
            pos = mesh.vertices[i]
            normal = mesh.normals[i]
            tex_coord = mesh.texture_coords[0][i] if mesh.has_texture_coords(0) else zero_3d

            v = Vertex(pos=pos, tex=tex_coord[:2], normal=normal)
            vertices.append(v)

        for i in range(mesh.num_faces):
            face = mesh.faces[i]
            assert face.num_indices == 3
            indices += face.indices

        # import matplotlib.pyplot as plt
        # vertices = np.array(vertices)
        #
        # # vertices = vertices[2000:3000]
        # # idx = 830
        # # vertices = vertices[2000:3000]
        # # vertices = vertices[idx*3:idx*3+3]
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.set_xlim([-100, 100])
        # ax.set_ylim([-100, 100])
        # ax.set_zlim([-100, 100])
        # ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], marker='o')
        #
        # plt.show()

        # import open3d as o3d
        # mesh_np = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.array(vertices)[:, :3]),
        #                                     o3d.utility.Vector3iVector(np.array(indices).reshape((-1, 3))))
        # o3d.visualization.draw_geometries([mesh_np])

        entry.init(vertices, indices)
        entry.vertex_array(self.program)
        return entry

    def use(self):
        for tex in self.textures:
            tex.use()

    def render(self):
        for entry in self.entries:
            entry.render()


HEIGHT = 1080
WIDTH = 1080
FieldDepth = 10.0


class Scene(ICallbacks):
    def __init__(self):
        self.ctx = moderngl.get_context()

        self.camera = Camera(WIDTH, HEIGHT,
                             pos=(3.0, 7.0, -10.0),
                             target=(0.0, -0.2, 1.0),
                             up=(0.0, 1.0, 0.0))

        self.effect = BasicLightingTechnique()
        self.effect.init()
        self.mesh = Mesh(self.effect.program)
        self.mesh.load_mesh(
            os.path.join(DATA_DIR, 'phoenix_ugv.md2')
        )
        self.scale = 0.0
        self.pipeline = Pipeline()
        self.pers_proj_info = PersProjInfo(
            fov=60.0,
            height=HEIGHT,
            width=WIDTH,
            z_near=1.0,
            z_far=50.0
        )
        self.direction_light = DirectionalLight(
            color=Vector3f([1.0, 1.0, 1.0]),
            ambient_intensity=1.0,
            diffuse_intensity=0.01,
            direction=Vector3f([1.0, -1.0, 0.0])
        )

    def render_scene_callback(self):
        self.ctx.clear()
        self.ctx.enable(self.ctx.DEPTH_TEST)

        self.scale += 0.5
        # self.camera.on_render()
        self.mesh.use()

        pl = [PointLight() for _ in range(2)]
        pl[0].diffuse_intensity = 0.25
        pl[0].color = Vector3f([1.0, 0.5, 0.0])
        pl[0].position = Vector3f([3.0, 1.0, FieldDepth * (np.cos(self.scale) + 1.0) / 2.0])
        pl[0].attenuation.linear = 0.1

        pl[1].diffuse_intensity = 0.25
        pl[1].color = Vector3f([0.0, 0.5, 1.0])
        pl[1].position = Vector3f([7.0, 1.0, FieldDepth * (np.sin(self.scale) + 1.0) / 2.0])
        pl[1].attenuation.linear = 0.1

        self.effect.set_point_lights(pl)

        sl = SpotLight()
        sl.diffuse_intensity = 0.9
        sl.color = Vector3f([0.0, 1.0, 1.0])
        sl.position = Vector3f(self.camera.pos)
        sl.direction = Vector3f(self.camera.target)
        sl.attenuation.linear = 0.1
        sl.cutoff = 10.0

        self.effect.set_spot_lights([sl])
        self.pipeline.set_scale(0.1, 0.1, 0.1)
        self.pipeline.set_rotate([90.0, self.scale, 0.0], degrees=True)
        self.pipeline.set_world_pos([0.0, 0.0, 10.0])

        self.pipeline.set_camera(
            **self.camera.get_camera_dict()
        )
        self.pipeline.set_perspective_projection(self.pers_proj_info)

        self.effect.set_wvp(self.pipeline.get_wvp_trans())
        self.effect.set_world_matrix(self.pipeline.get_world_trans())
        self.effect.set_directional_light(self.direction_light)
        self.effect.set_eye_world_pos(self.camera.pos.astype('float32'))
        self.effect.set_mat_specular_intensity(0.0)
        self.effect.set_mat_specular_power(0.0)

        self.effect.program['useTexture'] = True

        self.mesh.render()

        # vertices = self.mesh.entries[0].vertices
        # # #
        # points = np.array(vertices)[:, :3]
        # # # print(points)
        # points = np.concatenate(
        #     [
        #         points,
        #         np.ones_like(points[:, :1])
        #     ],
        #     1
        # )
        # wvp = self.pipeline.get_wvp_trans()
        # trans_points = points @ wvp.transpose()
        # trans_points = trans_points[:, :3] / trans_points[:, -1:]
        #
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.set_xlim([-1, 1])
        # ax.set_ylim([-1, 1])
        # ax.set_zlim([-1, 1])
        # ax.scatter(trans_points[:, 0], trans_points[:, 1], trans_points[:, 2], marker='o')
        # plt.show()
        # # print(trans_points)
        # # pass
        #
        # def debug_print(array):
        #     print('[')
        #     for row in array:
        #         print('[' + ', '.join(map(str, row)) + '],')
        #     print(']')
        #
        # debug_print(trans_points)
        #
        # import matplotlib.pyplot as plt
        # plt.plot(points[:, 0], points[:, 1], '.')
        # plt.show()



if __name__ == '__main__':
    backend = PyGameBackend(height=HEIGHT, width=WIDTH)
    # Scene()
    backend.run(Scene())
    #
    # Scene()


