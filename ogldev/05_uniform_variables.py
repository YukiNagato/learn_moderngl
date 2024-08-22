import math
import os
import sys

import moderngl
import pygame
import numpy as np

os.environ['SDL_WINDOWS_DPI_AWARENESS'] = 'permonitorv2'

pygame.init()
pygame.display.set_mode((800, 800), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)


class Scene:
    def __init__(self):
        self.ctx = moderngl.get_context()
        self.program = self.ctx.program(
            vertex_shader='''
                #version 330 core

                layout (location = 0) in vec3 Position;

                uniform float gScale;

                void main()
                {
                    gl_Position = vec4(gScale * Position.x, gScale * Position.y, Position.z, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330 core

                out vec4 out_color;

                void main() {
                    out_color = vec4(1.0, 0.0, 0.0, 0.0);
                }
            ''',
        )
        vertices = np.array([
            -1.0, -1.0, 0.0,
            1.0, -1.0, 0.0,
            0.0, 1.0, 0.0
        ])
        self.scale = 0.0
        self.delta = 0.001
        self.vbo = self.ctx.buffer(vertices.astype('f4').tobytes())
        self.vao = self.ctx.vertex_array(self.program, [(self.vbo, '3f', 'Position')])
        self.program['gScale'] = self.scale

    def render(self):
        self.ctx.clear()
        self.ctx.enable(self.ctx.DEPTH_TEST)
        
        self.scale += self.delta
        if self.scale >= 1.0 or self.scale <= -1.0:
            self.delta *= -1.0
        self.program['gScale'] = self.scale

        self.vao.render()


scene = Scene()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    scene.render()

    pygame.display.flip()
