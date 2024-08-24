import math
import os
import sys

import moderngl
import pygame
import numpy as np
import glm

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
                
                uniform mat4 gScaling;
                
                void main()
                {
                    gl_Position = gScaling * vec4(Position, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330

                out vec4 FragColor;

                void main()
                {
                    FragColor = vec4(1.0, 0.0, 0.0, 1.0);
                }
            ''',
        )
        vertices = np.array([
            -1.0, -1.0, 0.0,
            1.0, -1.0, 0.0,
            0.0, 1.0, 0.0
        ])

        self.scale = 1.0
        self.delta = 0.001
        self.vbo = self.ctx.buffer(vertices.astype('f4').tobytes())
        self.vao = self.ctx.vertex_array(self.program, [(self.vbo, '3f', 'Position')])

    def render(self):
        self.ctx.clear()
        self.ctx.enable(self.ctx.DEPTH_TEST)

        self.scale += self.delta
        if self.scale >= 1.5 or self.scale <= 0.5:
            self.delta *= -1.0

        scaling = np.array([
            self.scale, 0.0, 0.0, 0.0,
            0.0, self.scale, 0.0, 0.0,
            0.0, 0.0, self.scale, 0.0,
            0.0, 0.0, 0.0, 1.0]
        ).reshape((4, 4))

        scaling = glm.mat4(np.ascontiguousarray(scaling))
        self.program['gScaling'].write(scaling)

        self.vao.render()


scene = Scene()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    scene.render()

    pygame.display.flip()
