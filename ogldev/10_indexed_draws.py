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
            -1.0, -1.0, 0.0,
            0.0, -1.0, 1.0,
            1.0, -1.0, 0.0,
            0.0, 1.0, 0.0
        ])

        indices = np.array([0, 3, 1, 1, 3, 2, 2, 3, 0, 0, 1, 2], dtype='i4')
        self.scale = 0.0
        self.delta = 0.01
        self.vbo = self.ctx.buffer(vertices.astype('f4').tobytes())
        self.ibo = self.ctx.buffer(indices)
        self.vao = self.ctx.vertex_array(self.program, [(self.vbo, '3f', 'Position')], self.ibo)

    def render(self):
        self.ctx.clear()
        self.ctx.enable(self.ctx.DEPTH_TEST)

        self.scale += self.delta

        world = np.array([
            np.cos(self.scale), 0.0, -np.sin(self.scale), 0.0,
            0.0, 1.0, 0.0, 0.0,
            np.sin(self.scale), 0.0, np.cos(self.scale), 0.0,
            0.0, 0.0, 0.0, 1.0]
        ).reshape((4, 4))

        world = glm.mat4(np.ascontiguousarray(world))
        self.program['gWorld'].write(world)

        self.vao.render()


scene = Scene()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    scene.render()

    pygame.display.flip()
