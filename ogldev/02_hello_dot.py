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
                layout (location = 0) in vec3 in_vertex;

                void main() {
                    gl_Position = vec4(in_vertex, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330 core

                layout (location = 0) out vec4 out_color;

                void main() {
                    out_color = vec4(1.0, 1.0, 1.0, 1.0);
                }
            ''',
        )
        vertices = np.array([
            0.0, 0.0, 0.0
        ])
        self.vbo = self.ctx.buffer(vertices.astype('f4').tobytes())
        self.vao = self.ctx.vertex_array(self.program, [(self.vbo, '3f', 'in_vertex')])

    def render(self):
        self.ctx.clear()
        self.ctx.enable(self.ctx.DEPTH_TEST)
        self.vao.render(moderngl.POINTS)


scene = Scene()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    scene.render()

    pygame.display.flip()
