import numpy as np
from ogldev.lib.ogldev_keys import OGLDEVKeyState, OGLDEVMouse

class ICallbacks:
    def keyboard_callback(self, key, key_status: OGLDEVKeyState=OGLDEVKeyState.OGLDEV_KEY_STATE_PRESS):
        pass

    def passive_mouse_callback(self, x, y):
        pass

    def render_scene_callback(self):
        pass

    def mouse_callback(self, button: OGLDEVMouse, state: OGLDEVKeyState, x, y):
        pass