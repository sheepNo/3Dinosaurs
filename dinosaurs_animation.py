#!/usr/bin/env python3
"""
Main module that implements the Viewer class and loads every dinosaurs
"""
# Python built-in modules
import os                           # os function, i.e. checking file status
from itertools import cycle
import sys

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
import numpy as np

from transform import vec, translate, scale, rotate, identity, Trackball, sincos
from transform import (lerp, quaternion_slerp, quaternion_matrix, quaternion,
                       quaternion_from_euler)

from model_loading import load, Node

from renderable import Ground, GroundedNode, Forest
from keyboard_control import KeyboardControlNode

from texture_test import load_textured as load_textured_test
from texture import load_textured as load_textured
from texture import TexturedPlane

from animation_texture import load_skinned as load_skinned_test
from animation import load_skinned as load_skinned

from particles import FireEmiter

# ------------  Viewer class & window management ------------------------------
class GLFWTrackball(Trackball):
    """ Use in Viewer for interactive viewpoint control """

    def __init__(self, win):
        """ Init needs a GLFW window handler 'win' to register callbacks """
        super().__init__()
        self.mouse = (0, 0)
        glfw.set_cursor_pos_callback(win, self.on_mouse_move)
        glfw.set_scroll_callback(win, self.on_scroll)

    def on_mouse_move(self, win, xpos, ypos):
        """ Rotate on left-click & drag, pan on right-click & drag """
        old = self.mouse
        self.mouse = (xpos, glfw.get_window_size(win)[1] - ypos)
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT):
            self.drag(old, self.mouse, glfw.get_window_size(win))
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT):
            self.pan(old, self.mouse)

    def on_scroll(self, win, _deltax, deltay):
        """ Scroll controls the camera distance to trackball center """
        self.zoom(deltay, glfw.get_window_size(win)[1])

class Viewer(Node):
    """ GLFW viewer window, with classic initialization & graphics loop """

    def __init__(self, width=640, height=480):
        super().__init__("Scene")

        # version hints: create GL window with >= OpenGL 3.3 and core profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)
        self.win = glfw.create_window(width, height, 'Viewer', None, None)

        # make win's OpenGL context current; no OpenGL calls can happen before
        glfw.make_context_current(self.win)

        # register event handlers
        glfw.set_key_callback(self.win, self.on_key)

        # useful message to check OpenGL renderer characteristics
        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode() + ', GLSL',
              GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() +
              ', Renderer', GL.glGetString(GL.GL_RENDERER).decode())

        # initialize GL by setting viewport and default render characteristics
        GL.glClearColor(0.1, 0.1, 0.1, 0.1)
        GL.glEnable(GL.GL_DEPTH_TEST)         # depth test now enabled (TP2)
        GL.glEnable(GL.GL_CULL_FACE)          # backface culling enabled (TP2)

        # initialize trackball
        self.trackball = GLFWTrackball(self.win)

        # cyclic iterator to easily toggle polygon rendering modes
        self.fill_modes = cycle([GL.GL_LINE, GL.GL_POINT, GL.GL_FILL])

    def run(self):
        """ Main render loop for this OpenGL window """
        while not glfw.window_should_close(self.win):
            # clear draw buffer and depth buffer (<-TP2)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            winsize = glfw.get_window_size(self.win)
            view = self.trackball.view_matrix()
            projection = self.trackball.projection_matrix(winsize)

            """ draw our scene objects """
            self.draw(projection, view, identity(), win=self.win)

            # flush render commands, and swap draw buffers
            glfw.swap_buffers(self.win)

            # Poll for and process events
            glfw.poll_events()

    def on_key(self, _win, key, _scancode, action, _mods):
        """ 'Q' or 'Escape' quits """
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(self.win, True)
            # if key == glfw.KEY_SPACE:
            #     # if glfw.get_time() > 3:
            #     glfw.set_time(0)
            if key == glfw.KEY_V:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))

# -------------- main program and scene setup --------------------------------
def main():
    """ create a window, add scene objects, then run rendering loop """
    viewer = Viewer()

    ground = Ground("assets/heightmap.pgm")

    viewer.add(ground)

    viewer.add(GroundedNode(ground, 10, 10, y_offset_with_origin=0.5).add(*load_textured_test("TP5/bunny.obj", "TP5/bunny.png")))

    grounded_dino_walk = GroundedNode(ground).add(*load_skinned_test("dino/Dinosaurus_walk.dae", "dino/textures/Donosaurus.jpg"))
    grounded_dino_idle = GroundedNode(ground).add(*load_skinned_test("dino/Dinosaurus_idle2.dae", "dino/textures/Donosaurus.jpg"))
    grounded_dino_eat = GroundedNode(ground).add(*load_skinned_test("dino/Donosaurus_eat.dae", "dino/textures/Donosaurus.jpg"))

    movement_speed = .22

    moving_dino_walk = KeyboardControlNode(glfw.KEY_UP, glfw.KEY_DOWN, glfw.KEY_LEFT, glfw.KEY_RIGHT, glfw.KEY_UP, show=False, time=1, speed=movement_speed)
    moving_dino_walk.add(grounded_dino_walk)
    moving_dino_idle = KeyboardControlNode(glfw.KEY_UP, glfw.KEY_DOWN, glfw.KEY_LEFT, glfw.KEY_RIGHT, glfw.KEY_UP, glfw.KEY_SPACE, time=3, speed=movement_speed)
    moving_dino_idle.add(grounded_dino_idle)
    moving_dino_eat = KeyboardControlNode(glfw.KEY_UP, glfw.KEY_DOWN, glfw.KEY_LEFT, glfw.KEY_RIGHT, glfw.KEY_SPACE, show=False, interact=True, time=5, speed=movement_speed)
    moving_dino_eat.add(grounded_dino_eat)

    viewer.add(moving_dino_walk, moving_dino_idle, moving_dino_eat)
    viewer.add(Forest(ground, n_trees=20, fire=True)) # trees have a probability of 0.2 to be on fire

    print("Ready!\n Press W, A or D to move around\n Press SPACE to eat some grass (or a rabbit)\n")
    # start rendering loop
    viewer.run()

if __name__ == '__main__':
    glfw.init()                # initialize window system glfw
    main()                     # main function keeps variables locally scoped
    glfw.terminate()           # destroy all glfw windows and GL contexts
