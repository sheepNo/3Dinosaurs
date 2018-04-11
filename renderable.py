"""
Module for some special renderable objects (sa the ground)
"""

from model_loading import load, Node, ColorMesh

from transform import translate

# A cylinder class mainly for debuging and testing
class Cylinder(Node):
    """ Very simple cylinder based on practical 2 load function """
    def __init__(self):
        super().__init__()
        self.add(*load('assets/cylinder.obj'))  # just load the cylinder from file
# a cylinder on the ground
class GroundCylinder(Node):
    def __init__(self, ground, x=0, z=0):
        super().__init__()
        cylinderNode = Node(transform=translate(x, ground.get_local_height(x, z) + 1, z))
        cylinderNode.add(*load('assets/cylinder.obj'))
        self.add(cylinderNode)

# A ground generated from a pgm heightmap and rendered with usual triangle mode
# The heightmap should be of type P5 (meaning greyscale binary)
DENSITY = 10 # the density of the heightmap, ie points/m
MAX_HEIGHT = 4 # the maximum height in m

class Ground(Node):
    def __init__(self, heightmap_file):
        # loading the heightmap into a ColorMesh
        with open(heightmap_file, 'rb') as pgmf:
            assert pgmf.readline() == b'P5\n'
            (width, height) = [int(i) for i in pgmf.readline().split()]
            depth = int(pgmf.readline())
            assert depth <= 255

            # boundaries
            N = width * height
            min_x = - (width >> 1) / DENSITY
            min_z = - (height >> 1) / DENSITY
            band_size = 5*DENSITY # bands of colors are 5m long

            vertex_positions = [(0, 0, 0)]*N
            vertex_colors = [(0, 0, 0)]*N
            for h in range(height):
                for w in range(width):
                    vertex_positions[h*width + w] = (min_x + w/DENSITY, ord(pgmf.read(1))/(255/MAX_HEIGHT), min_z + h/DENSITY)
                    vertex_colors[h*width + w] = ((w%band_size)/band_size, (h%band_size)/band_size, 0)

        # index
        indexes = [0]*(6*(width)*(height))
        for h in range(height-1):
            for w in range(width-1):
                i = h*width + w
                indexes[6*i] = i+1
                indexes[6*i +1] = i
                indexes[6*i +2] = (h+1)*width + w

                indexes[6*i +3] = i+1
                indexes[6*i +4] = (h+1)*width + w
                indexes[6*i +5] = (h+1)*(width) + w+1

        self.heights = [vertex_position[1] for vertex_position in vertex_positions]
        self.width = width
        self.min_x = min_x
        self.min_z = min_z

        super().__init__(name="Ground")
        
        self.add(ColorMesh([vertex_positions, vertex_colors], indexes))

    def get_local_height(self, x, z):
        """ Returns the local height of the ground x and z are in meters """
        x_pos, z_pos = int((x-self.min_x)*DENSITY), int((z-self.min_z)*DENSITY)
        return self.heights[z_pos*self.width + x_pos]
