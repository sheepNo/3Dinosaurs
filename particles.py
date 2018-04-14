"""
A module for particles systems (here to do some fire)
The module is pretty inefficient
"""

import numpy as np
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw

from math import sin
from random import randint

from model_loading import VertexArray, MonoColorMesh, Node

from transform import vec, translate, scale, identity

from shaders import Shader, UNIFORM_COLOR_VERT, UNIFORM_COLOR_FRAG

class Particle(MonoColorMesh):
    def __init__(self, color_shader, quad_vertex_array, life=0, color=(1,1,1,1), height=15, width=1, x_offset=0, z_offset=0):
        super().__init__(color, quad_vertex_array, color_shader)
        self.color = color
        self.width = width
        self.height = height
        self.life = life # Remaining life of the particle. if < 0 : dead and unused.
        self.total_life = life
        self.x_offset = x_offset
        self.z_offset = z_offset

        self.model = identity()

    def draw(self, projection, view, model, **_kwargs):
        if self.life > 0:
            super().draw(projection, view, model @ self.model, **_kwargs)
    def update(self, delta):
        if self.life > 0:
            self.life -= delta
            
            life_ratio = self.life/self.total_life # life_ratio goes from 1 to 0
            self.color = (1, life_ratio, 0, 1)
            self.model = scale(0.5 + 0.5*life_ratio) @ translate(self.x_offset + self.width*life_ratio*sin(self.height*life_ratio), self.height*(1-life_ratio), self.z_offset + self.width*life_ratio*sin(self.height*life_ratio))
        else:
            self.life = self.total_life
    

class FireEmiter(Node):
    def __init__(self, max_particles=100, width=1, height=15, density=10): # density is the ratio width/particle_size
        super().__init__()

        self.max_particles = max_particles
        self.width = width
        self.height = height
        self.density = density

        self.color_shader = Shader(UNIFORM_COLOR_VERT, UNIFORM_COLOR_FRAG)

        # creating the small cube for a single particle
        self.particle = [(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)]
        self.particle_quad = VertexArray([self.particle], (0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 4, 3, 3, 4, 7, 1, 2, 5, 2, 6, 5, 2, 3, 7, 2, 7, 6, 0, 1, 4, 1, 5, 4))

        self.particles = np.empty([max_particles], Particle)
        for i in range(max_particles):
            self.particles[i] = self.new_particle()
            self.add(self.particles[i])

        self.last_time = glfw.get_time()

    def new_particle(self):
        return Particle(self.color_shader, self.particle_quad, life=0.5+1.5*randint(0, 10)/10, height=randint(int(0.5*self.height), int(1.5*self.height)), width=0.5+1.5*randint(0,10)/10, x_offset = -self.width/2 + randint(0, self.density)/self.density, z_offset = -self.width/2 + randint(0, self.density)/self.density)

    def draw(self, model, view, projection, **_kwargs):
        delta = glfw.get_time() - self.last_time

        for particle in self.particles:
            particle.update(delta)

        super().draw(model, view, projection, **_kwargs)
        self.last_time = glfw.get_time()
