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
from PIL import Image
import os

os.environ['SDL_WINDOWS_DPI_AWARENESS'] = 'permonitorv2'

pygame.init()
pygame.display.set_mode((800, 800), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)


class ImageTexture:
    def __init__(self, path):
        self.ctx = moderngl.get_context()

        img = Image.open(path).convert('RGBA')
        self.texture = self.ctx.texture(img.size, 4, img.tobytes())
        self.sampler = self.ctx.sampler(texture=self.texture)

    def use(self):
        self.sampler.use()


class Scene:
    def __init__(self):
        self.ctx = moderngl.get_context()
        self.program = self.ctx.program(
            vertex_shader='''
                #version 330
                
                layout (location = 0) in vec3 Position;
                layout (location = 1) in vec2 TexCoord;
                
                uniform mat4 gWVP;
                
                out vec2 TexCoord0;
                
                void main()
                {
                    gl_Position = gWVP * vec4(Position, 1.0);
                    TexCoord0 = TexCoord;
                }
            ''',
            fragment_shader='''
                #version 330
                
                in vec2 TexCoord0;
                
                out vec4 FragColor;
                
                uniform sampler2D gSampler;
                
                void main()
                {
                    FragColor = texture2D(gSampler, TexCoord0.xy);
                }
            ''',
        )
        vertices = np.array([
            -1.0, -1.0, 0.5773, 0.0, 0.0,
            0.0, -1.0, -1.15475, 0.5, 0.0,
            1.0, -1.0, 0.5773, 1.0, 0.0,
            0.0, 1.0, 0.0, 0.5, 1.0
        ])
        self.vertices = vertices
        indices = np.array([0, 3, 1, 1, 3, 2, 2, 3, 0, 0, 1, 2], dtype='i4')
        self.scale = 0.0
        self.delta = 0.1
        self.pipline = Pipeline()
        self.vbo = self.ctx.buffer(vertices.astype('f4').tobytes())
        self.ibo = self.ctx.buffer(indices)
        self.vao = self.ctx.vertex_array(self.program, [(self.vbo, '3f 2f', 'Position', 'TexCoord')], self.ibo)
        self.pers_proj_info = PersProjInfo(
            fov=60.0,
            height=800,
            width=800,
            z_near=1.0,
            z_far=100.0
        )
        self.camera = Camera(800, 800)
        self.texture = ImageTexture('ogldev/data/test.png')

    def render(self):
        self.ctx.clear()
        self.ctx.enable(self.ctx.DEPTH_TEST)
        self.texture.use()
        self.camera.on_render()
        self.scale += self.delta
        self.pipline.set_world_pos((0.0, 0.0, 3.0))
        self.pipline.set_rotate((0.0, self.scale, 0.0))
        self.pipline.set_camera(
            **self.camera.get_camera_dict()
        )
        self.pipline.set_perspective_projection(self.pers_proj_info)
        g_wvp = self.pipline.get_wvp_trans()
        g_wvp = glm.mat4(np.ascontiguousarray(g_wvp))
        self.program['gWVP'].write(g_wvp)

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
