"""
Module for keyboard control node
"""

import glfw

from transform import vec, rotate, translate, sincos, normalized
from model_loading import Node


class KeyboardControlNode(Node):
    def __init__(self, key_forward, key_backward, key_left, key_right, key_toggle, key_toggle2=glfw.KEY_Q, show=True, interact=False, time=1, speed=.5, **param):
        super().__init__(**param)   # forward base constructor named arguments
        self.axis, self.angle = vec(0, 1, 0), 0
        self.key_forward, self.key_backward, self.key_left, self.key_right, self.key_toggle, self.key_toggle2 = key_forward, key_backward, key_left, key_right, key_toggle, key_toggle2
        self.transform = translate(0, 0, 0)
        self.speed = speed
        self.show = show
        self.time = time
        self.interact = interact
        self.first_time = True

    def draw(self, projection, view, model, win=None, **param):
        assert win is not None

        # rotation management
        self.angle += 2 * int(glfw.get_key(win, self.key_left) == glfw.PRESS)
        self.angle -= 2 * int(glfw.get_key(win, self.key_right) == glfw.PRESS)
        rotation = rotate(self.axis, self.angle)
        # translation management
        movement_magnitude = self.speed * (int(glfw.get_key(win, self.key_forward) == glfw.PRESS) - int(glfw.get_key(win, self.key_backward) == glfw.PRESS))
        forward_vector = -rotation @ vec(0, 0, movement_magnitude, 1) # the - and magnitude on z is here to correct the dae oriention

        old_translation = translate(self.transform[0][3], self.transform[1][3], self.transform[2][3])
        translation = old_translation + translate(forward_vector[0], 0, forward_vector[2])

        goalx = 0
        goaly = 0

        if glfw.get_time() > self.time:
            if (glfw.get_key(win, self.key_forward) == glfw.PRESS or glfw.get_key(win, self.key_forward) == PRESS):
                # resets walking animation when key is held
                glfw.set_time(0)
            if (self.show and (glfw.get_key(win, self.key_toggle) != glfw.PRESS and glfw.get_key(win, self.key_toggle2) != glfw.PRESS)):
                # resets idle animation
                glfw.set_time(0)
            if (self.interact):
                glfw.set_time(0)
        # elif self.first_time and self.interact and glfw.get_key(win, self.key_toggle) != glfw.PRESS:
        #     # make sure the dino eating animation starts when toggle is pressed
        #     glfw.set_time(0)
        #     self.first_time = False
        #
        # if self.interact and (glfw.get_key(win, self.key_forward) == glfw.RELEASE or glfw.get_key(win, self.key_toggle) == glfw.RELEASE or glfw.get_key(win, self.key_toggle2) == glfw.RELEASE):
        #     self.first_time = True



        # self.transform = translation @ rotate(self.axis, self.angle)
        self.transform = translation @ rotation


        # call Node's draw method to pursue the hierarchical tree calling
        if self.show:
            if (glfw.get_key(win, self.key_toggle) != glfw.PRESS and glfw.get_key(win, self.key_toggle2) != glfw.PRESS):
                super().draw(projection, view, model, win=win, x=translation[0][3]/2, z=translation[2][3]/2, **param)
        else:
            if (glfw.get_key(win, self.key_toggle) == glfw.PRESS):
                super().draw(projection, view, model, win=win, x=translation[0][3]/2, z=translation[2][3]/2, **param)
            # else:
            #     if (self.interact):
            #         super().draw(projection, view, model, win=win, x=translation[0][3], z=translation[2][3], **param)
