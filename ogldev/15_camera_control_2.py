import math
import os
import sys

import moderngl
import pygame
import numpy as np
import glm
from ogldev.lib.ogldev_pipeline import Pipeline
from ogldev.lib.ogldev_math_3d import PersProjInfo
from ogldev.lib.ogldev_camera import Camera


os.environ['SDL_WINDOWS_DPI_AWARENESS'] = 'permonitorv2'

pygame.init()
pygame.display.set_mode((800, 800), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)


class Scene:
    def __init__(self):
        self.ctx = moderngl.get_context()
        self.program = self.ctx.program(
            vertex_shader='''
                #version 330

                layout (location = 0) in vec3 Position;

                uniform mat4 gWorld;

                out vec4 Color;

                void main()
                {
                    gl_Position = gWorld * vec4(Position, 1.0);
                    Color = vec4(clamp(Position, 0.0, 1.0), 1.0);
                }
            ''',
            fragment_shader='''
                #version 330

                in vec4 Color;

                out vec4 FragColor;

                void main()
                {
                    FragColor = Color;
                }
            ''',
        )
        vertices = np.array([
            -1.0, -1.0, 0.5773,
            0.0, -1.0, -1.15475,
            1.0, -1.0, 0.5773,
            0.0, 1.0, 0.0
        ])
        self.vertices = vertices
        indices = np.array([0, 3, 1, 1, 3, 2, 2, 3, 0, 0, 1, 2], dtype='i4')
        self.scale = 0.0
        self.delta = 0.01
        self.pipline = Pipeline()
        self.vbo = self.ctx.buffer(vertices.astype('f4').tobytes())
        self.ibo = self.ctx.buffer(indices)
        self.vao = self.ctx.vertex_array(self.program, [(self.vbo, '3f', 'Position')], self.ibo)
        self.pers_proj_info = PersProjInfo(
            fov=60.0,
            height=800,
            width=800,
            z_near=1.0,
            z_far=100.0
        )
        self.camera = Camera(800, 800)

    def render(self):
        self.ctx.clear()
        self.ctx.enable(self.ctx.DEPTH_TEST)

        self.scale += self.delta
        self.pipline.set_world_pos((0.0, 0.0, 3.0))
        self.pipline.set_rotate((0.0, self.scale, 0.0))
        self.pipline.set_camera(
            **self.camera.get_camera_dict()
        )
        self.pipline.set_perspective_projection(self.pers_proj_info)
        world = self.pipline.get_wvp_trans()
        world = glm.mat4(np.ascontiguousarray(world))
        self.program['gWorld'].write(world)

        self.vao.render()


scene = Scene()
mouse_button_down = False
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            scene.camera.on_keyboard(event.key)
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_button_down = True
        if event.type == pygame.MOUSEBUTTONUP:
            mouse_button_down = False
        if event.type == pygame.MOUSEMOTION:
            if mouse_button_down:
                scene.camera.on_mouse(*event.pos)


    scene.render()

    pygame.display.flip()
