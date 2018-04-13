"""
Module for keyboard control node
"""

import glfw

from transform import vec, rotate, translate, sincos, normalized
from model_loading import Node


class KeyboardControlNode(Node):
    def __init__(self, key_forward, key_backward, key_left, key_right, speed=.5, **param):
        super().__init__(**param)   # forward base constructor named arguments
        self.axis, self.angle = vec(0, 1, 0), 0
        self.key_forward, self.key_backward, self.key_left, self.key_right = key_forward, key_backward, key_left, key_right
        self.transform = translate(0, 0, 0)
        self.speed = speed

    def draw(self, projection, view, model, win=None, **param):
        if (glfw.get_time() > 1 and (glfw.get_key(win, self.key_forward) == glfw.PRESS)) :
            glfw.set_time(0)
        assert win is not None
        # rotation management
        self.angle += 2 * int(glfw.get_key(win, self.key_left) == glfw.PRESS)
        self.angle -= 2 * int(glfw.get_key(win, self.key_right) == glfw.PRESS)
        # translation management
        sin, cos = sincos(self.angle)
        new_x = cos * int(glfw.get_key(win, self.key_forward) == glfw.PRESS)
        new_z = -sin * int(glfw.get_key(win, self.key_forward) == glfw.PRESS)
        new_x += -cos * int(glfw.get_key(win, self.key_backward) == glfw.PRESS)
        new_z += sin * int(glfw.get_key(win, self.key_backward) == glfw.PRESS)

        old_translation = translate(self.transform[0][3], self.transform[1][3], self.transform[2][3])
        new_translation = translate(self.speed*normalized(vec(new_x, 0, new_z)))
        translation = old_translation + new_translation

        self.transform = translation @ rotate(self.axis, self.angle)

        # call Node's draw method to pursue the hierarchical tree calling
        super().draw(projection, view, model, win=win, x=translation[0][3], z=translation[2][3], **param)
