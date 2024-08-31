import os
import pygame
import sys
from ogldev.lib.ogldev_callbacks import ICallbacks

os.environ['SDL_WINDOWS_DPI_AWARENESS'] = 'permonitorv2'


class PyGameBackend:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        pygame.init()
        pygame.display.set_mode((self.width, self.height),
                                flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)
        self.mouse_button_down = False

    def run(self, app: ICallbacks):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    app.keyboard_callback(event.key)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.mouse_button_down = True
                if event.type == pygame.MOUSEBUTTONUP:
                    self.mouse_button_down = False
                if event.type == pygame.MOUSEMOTION:
                    if self.mouse_button_down:
                        app.passive_mouse_callback(*event.pos)

            app.render_scene_callback()
            pygame.display.flip()



