import numpy as np
from ogldev.lib.ogldev_math_3d import normalize, rotate_vector
import ogldev.lib.ogldev_keys as OGLKeys
from typing import Optional


STEP_SCALE = 0.01
EDGE_STEP = 0.5
MARGIN = 10


class MousePos:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Camera:
    def __init__(self, width, height, pos=None, target=None, up=None):
        self._windows_width = width
        self._windows_height = height
        if pos is None:
            pos = np.array([0, 0, 0], dtype=float)
        if target is None:
            target = np.array([0, 0, 1], dtype=float)
        if up is None:
            up = np.array([0, 1, 0], dtype=float)

        self.pos: np.ndarray = np.array(pos, dtype=float)
        self.target: np.ndarray = normalize(np.array(target, dtype=float))
        self.up: np.ndarray = normalize(np.array(up, dtype=float))
        self._angle_h: Optional[float] = None
        self._angle_v: Optional[float] = None
        self._on_upper_edge: bool = False
        self._on_lower_edge: bool = False
        self._on_left_edge: bool = False
        self._on_right_edge: bool = False
        self._mouse_pos: MousePos = MousePos(self._windows_width / 2, self._windows_height / 2)
        self.init()

    def init(self):
        h_target = np.array([self.target[0], 0, self.target[2]])
        h_target = normalize(h_target)
        if h_target[2] >= 0:
            if h_target[0] >= 0:
                self._angle_h = 360.0 - np.rad2deg(np.arcsin(h_target[2]))
            else:
                self._angle_h = 180.0 + np.rad2deg(np.arcsin(h_target[2]))
        else:
            if h_target[0] >= 0:
                self._angle_h = np.rad2deg(np.arcsin(-h_target[2]))
            else:
                self._angle_h = 180.0 - np.rad2deg(np.arcsin(-h_target[2]))

        self._angle_v = -np.rad2deg(np.arcsin(self.target[1]))

    def on_keyboard(self, key: int):
        ret = False
        match key:
            case OGLKeys.OGLDEV_KEY_UP:
                self.pos += STEP_SCALE * self.target
                ret = True
            case OGLKeys.OGLDEV_KEY_DOWN:
                self.pos -= STEP_SCALE * self.target
                ret = True
            case OGLKeys.OGLDEV_KEY_LEFT:
                left = np.cross(self.target, self.up)
                left = normalize(left)
                left *= STEP_SCALE
                self.pos += left
                ret = True
            case OGLKeys.OGLDEV_KEY_RIGHT:
                right = np.cross(self.up, self.target)
                right = normalize(right)
                right *= STEP_SCALE
                self.pos += right
                ret = True
            case OGLKeys.OGLDEV_KEY_PAGE_UP:
                self.pos[1] += STEP_SCALE
            case OGLKeys.OGLDEV_KEY_PAGE_DOWN:
                self.pos[1] -= STEP_SCALE
        return ret

    def get_camera_dict(self):
        return {
            'pos': self.pos,
            'up': self.up,
            'target': self.target
        }

    def on_mouse(self, x, y):
        delta_x = x - self._mouse_pos.x
        delta_y = y - self._mouse_pos.y

        self._mouse_pos.x = x
        self._mouse_pos.y = y
        # print(self._angle_h, self._angle_v)
        self._angle_h += delta_x / 20.0
        self._angle_v += delta_y / 20.0

        # print(self._angle_h, self._angle_v)
        if delta_x == 0:
            if x <= MARGIN:
                self._on_left_edge = True
            elif x >= self._windows_width - MARGIN:
                self._on_right_edge = True
        else:
            self._on_left_edge = False
            self._on_right_edge = False

        if delta_y == 0:
            if y <= MARGIN:
                self._on_upper_edge = True
            elif y >= self._windows_height - MARGIN:
                self._on_lower_edge = True
        else:
            self._on_upper_edge = False
            self._on_lower_edge = False

        self.update()

    def update(self):
        v_axis = np.array([0.0, 1.0, 0.0])
        view = np.asarray([1.0, 0.0, 0.0])
        view = rotate_vector(view, self._angle_h, v_axis)
        view = normalize(view)
        h_axis = normalize(np.cross(v_axis, view))
        view = rotate_vector(view, self._angle_v, h_axis)
        self.target = normalize(view)
        self.up = normalize(np.cross(self.target, h_axis))

    def on_render(self):
        should_update = False
        if self._on_left_edge:
            self._angle_h -= EDGE_STEP
            should_update = True
        elif self._on_right_edge:
            self._angle_h += EDGE_STEP
            should_update = True

        if self._on_upper_edge:
            if self._angle_v > -90.0:
                self._angle_v -= EDGE_STEP
                should_update = True
        elif self._on_lower_edge:
            if self._angle_v < 90.0:
                self._angle_v += EDGE_STEP
                should_update = True

        if should_update:
            self.update()




