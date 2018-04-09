"""
Module for some special renderable objects (sa the ground)
"""

from model_loading import load, Node, VertexArray

# A cylinder class mainly for debuging and testing
class Cylinder(Node):
    """ Very simple cylinder based on practical 2 load function """
    def __init__(self):
        super().__init__()
        self.add(*load('assets/cylinder.obj'))  # just load the cylinder from file

# TODO
class Ground(VertexArray):
    pass
