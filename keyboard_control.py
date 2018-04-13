"""
Module for keyboard control node
"""

import glfw

from transform import vec, rotate, translate, sincos, normalized
from model_loading import Node


class KeyboardControlNode(Node):
    def __init__(self, key_forward, key_backward, key_left, key_right, key_toggle, show=True, time=1, speed=.5, **param):
        super().__init__(**param)   # forward base constructor named arguments
        self.axis, self.angle = vec(0, 1, 0), 0
        self.key_forward, self.key_backward, self.key_left, self.key_right, self.key_toggle = key_forward, key_backward, key_left, key_right, key_toggle
        self.transform = translate(0, 0, 0)
        self.speed = speed
        self.show = show
        self.time = time

    def draw(self, projection, view, model, win=None, **param):
        assert win is not None

        if (glfw.get_time() > self.time and (glfw.get_key(win, self.key_forward) == glfw.PRESS)) :
            glfw.set_time(0)
        if (self.show and glfw.get_time() > self.time):
            glfw.set_time(0)

        # rotation management
        self.angle += 2 * int(glfw.get_key(win, self.key_left) == glfw.PRESS)
        self.angle -= 2 * int(glfw.get_key(win, self.key_right) == glfw.PRESS)
        rotation = rotate(self.axis, self.angle)
        # translation management
        movement_magnitude = self.speed * (int(glfw.get_key(win, self.key_forward) == glfw.PRESS) - int(glfw.get_key(win, self.key_backward) == glfw.PRESS))
        forward_vector = -rotation @ vec(0, 0, movement_magnitude, 1) # the - and magnitude on z is here to correct the dae oriention

        old_translation = translate(self.transform[0][3], self.transform[1][3], self.transform[2][3])
        translation = old_translation + translate(forward_vector[0], 0, forward_vector[2])

        self.transform = translation @ rotation

        # call Node's draw method to pursue the hierarchical tree calling
        if ((glfw.get_key(win, self.key_toggle) == glfw.PRESS) != self.show):
            # For whatever reason, we need to divide the coordinates by 2, the keyboard goes too far
            super().draw(projection, view, model, win=win, x=translation[0][3]/2, z=translation[2][3]/2, **param)
