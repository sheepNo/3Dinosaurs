#!/usr/bin/env python3
"""
Python OpenGL practical application.
"""
# Python built-in modules
import os                           # os function, i.e. checking file status
import sys

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
import numpy as np                  # all matrix manipulations & OpenGL args

from transform import translate, rotate, scale, vec, Trackball, identity
from transform import frustum, perspective

from PIL import Image               # load images for textures
from itertools import cycle

import pyassimp
import pyassimp.errors

# ------------ low level OpenGL object wrappers ----------------------------
class Shader:
    """ Helper class to create and automatically destroy shader program """
    @staticmethod
    def _compile_shader(src, shader_type):
        src = open(src, 'r').read() if os.path.exists(src) else src
        src = src.decode('ascii') if isinstance(src, bytes) else src
        shader = GL.glCreateShader(shader_type)
        GL.glShaderSource(shader, src)
        GL.glCompileShader(shader)
        status = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
        src = ('%3d: %s' % (i+1, l) for i, l in enumerate(src.splitlines()))
        if not status:
            log = GL.glGetShaderInfoLog(shader).decode('ascii')
            GL.glDeleteShader(shader)
            src = '\n'.join(src)
            print('Compile failed for %s\n%s\n%s' % (shader_type, log, src))
            return None
        return shader

    def __init__(self, vertex_source, fragment_source):
        """ Shader can be initialized with raw strings or source file names """
        self.glid = None
        vert = self._compile_shader(vertex_source, GL.GL_VERTEX_SHADER)
        frag = self._compile_shader(fragment_source, GL.GL_FRAGMENT_SHADER)
        if vert and frag:
            self.glid = GL.glCreateProgram()  # pylint: disable=E1111
            GL.glAttachShader(self.glid, vert)
            GL.glAttachShader(self.glid, frag)
            GL.glLinkProgram(self.glid)
            GL.glDeleteShader(vert)
            GL.glDeleteShader(frag)
            status = GL.glGetProgramiv(self.glid, GL.GL_LINK_STATUS)
            if not status:
                print(GL.glGetProgramInfoLog(self.glid).decode('ascii'))
                GL.glDeleteProgram(self.glid)
                self.glid = None

    def __del__(self):
        GL.glUseProgram(0)
        if self.glid:                      # if this is a valid shader object
            GL.glDeleteProgram(self.glid)  # object dies => destroy GL object

class Texture:
    """ Helper class to create and automatically destroy textures """
    def __init__(self, file, wrap_mode=GL.GL_REPEAT, min_filter=GL.GL_LINEAR,
                 mag_filter=GL.GL_LINEAR_MIPMAP_LINEAR):
        self.glid = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.glid)
        # helper array stores texture format for every pixel size 1..4
        format = [GL.GL_LUMINANCE, GL.GL_LUMINANCE_ALPHA, GL.GL_RGB, GL.GL_RGBA]

        try:
            # imports image as a numpy array in exactly right format
            tex = np.array(Image.open(file))
            format = format[0 if len(tex.shape) == 2 else tex.shape[2] - 1]
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, tex.shape[1],
                            tex.shape[0], 0, format, GL.GL_UNSIGNED_BYTE, tex)

            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, wrap_mode)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, wrap_mode)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, min_filter)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, mag_filter)
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
            message = 'Loaded texture %s\t(%s, %s, %s, %s)'
            print(message % (file, tex.shape, wrap_mode, min_filter, mag_filter))
        except FileNotFoundError:
            print("ERROR: unable to load texture file %s" % file)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def __del__(self):  # delete GL texture from GPU when object dies
        GL.glDeleteTextures(self.glid)

# -------------- Example texture plane class ----------------------------------
TEXTURE_VERT = """#version 330 core
uniform mat4 modelviewprojection;
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 tex;
out vec2 fragTexCoord;
void main() {
    gl_Position = modelviewprojection * vec4(position, 1);
    fragTexCoord =tex;
}"""

TEXTURE_FRAG = """#version 330 core
uniform sampler2D diffuseMap;
in vec2 fragTexCoord;
out vec4 outColor;
void main() {
    outColor = texture(diffuseMap, fragTexCoord);
}"""

# ------------  Simple color shaders ------------------------------------------
COLOR_VERT = """#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

uniform int color_id;

uniform mat4 PVM;
uniform mat4 M;

out vec4 out_position;
out vec3 out_normal;
out vec3 out_color;

void main() {
    gl_Position = PVM * vec4(position, 1);
    out_position = M * vec4(position, 1);
    out_color = vec3(1,0,0);
	out_normal = normal;
}"""

COLOR_FRAG = """#version 330 core
in vec3 out_normal;
in vec4 out_position;
in vec3 out_color;

out vec4 color;

uniform vec4 camera_position;

vec3 l = normalize(vec3(1, 1, 1));

vec3 Ka = vec3(0.19125, 0.0735, 0.0225);
vec3 Kd = vec3(0.7038, 0.27048, 0.0828);
vec3 Ks = vec3(0.256777, 0.137622, 0.086014);

float shine = 0.01 * 128;

void main() {
    vec4 c = camera_position - out_position;
    vec3 v = vec3(c[0], c[1], c[2]);
    color = vec4( Ka + Kd * dot(out_normal, l) + Ks * pow( dot( reflect(l, out_normal) , v ) , shine) , 1);
}"""

class TexturedPlane:
    """ Simple first textured object """

    def __init__(self, file):
        # feel free to move this up in the viewer as per other practicals
        self.shader = Shader(TEXTURE_VERT, TEXTURE_FRAG)

        # triangle and face buffers
        vertices = 100 * np.array(((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)), np.float32)
        faces = np.array(((0, 1, 2), (0, 2, 3)), np.uint32)
        self.vertex_array = VertexArray([vertices], faces)

        # interactive toggles
        self.wrap = cycle([GL.GL_REPEAT, GL.GL_MIRRORED_REPEAT,
                           GL.GL_CLAMP_TO_BORDER, GL.GL_CLAMP_TO_EDGE])
        self.filter = cycle([(GL.GL_NEAREST, GL.GL_NEAREST),
                             (GL.GL_LINEAR, GL.GL_LINEAR),
                             (GL.GL_LINEAR, GL.GL_LINEAR_MIPMAP_LINEAR)])
        self.wrap_mode, self.filter_mode = next(self.wrap), next(self.filter)
        self.file = file

        # setup texture and upload it to GPU
        self.texture = Texture(file, self.wrap_mode, *self.filter_mode)

    def draw(self, projection, view, model, win=None, **_kwargs):

        # some interactive elements
        if glfw.get_key(win, glfw.KEY_F6) == glfw.PRESS:
            self.wrap_mode = next(self.wrap)
            self.texture = Texture(self.file, self.wrap_mode, *self.filter_mode)

        if glfw.get_key(win, glfw.KEY_F7) == glfw.PRESS:
            self.filter_mode = next(self.filter)
            self.texture = Texture(self.file, self.wrap_mode, *self.filter_mode)

        GL.glUseProgram(self.shader.glid)

        # projection geometry
        loc = GL.glGetUniformLocation(self.shader.glid, 'modelviewprojection')
        GL.glUniformMatrix4fv(loc, 1, True, projection @ view @ model)

        # texture access setups
        loc = GL.glGetUniformLocation(self.shader.glid, 'diffuseMap')
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture.glid)
        GL.glUniform1i(loc, 0)
        self.vertex_array.draw(GL.GL_TRIANGLES)

        # leave clean state for easier debugging
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUseProgram(0)

class TexturedMesh:

        def __init__(self, texture, attributes, index=None):
            # feel free to move this up in the viewer as per other practicals
            self.shader = Shader(TEXTURE_VERT, TEXTURE_FRAG)

            # triangle and face buffers

            self.vertex_array = VertexArray(attributes, index)

            # interactive toggles
            # self.wrap = cycle([GL.GL_REPEAT, GL.GL_MIRRORED_REPEAT,
            #                    GL.GL_CLAMP_TO_BORDER, GL.GL_CLAMP_TO_EDGE])
            # self.filter = cycle([(GL.GL_NEAREST, GL.GL_NEAREST),
            #                      (GL.GL_LINEAR, GL.GL_LINEAR),
            #                      (GL.GL_LINEAR, GL.GL_LINEAR_MIPMAP_LINEAR)])
            # self.wrap_mode, self.filter_mode = next(self.wrap), next(self.filter)

            # setup texture and upload it to GPU
            self.texture = texture

        #def draw(self, projection, view, model, color_shader=None, color=(1, 1, 1, 1), **param):
        def draw(self, projection, view, model, win=None, **_kwargs):

            # some interactive elements
            # if glfw.get_key(win, glfw.KEY_F6) == glfw.PRESS:
            #     self.wrap_mode = next(self.wrap)
            #     self.texture = Texture(self.file, self.wrap_mode, *self.filter_mode)
            #
            # if glfw.get_key(win, glfw.KEY_F7) == glfw.PRESS:
            #     self.filter_mode = next(self.filter)
            #     self.texture = Texture(self.file, self.wrap_mode, *self.filter_mode)

            GL.glUseProgram(self.shader.glid)

            # projection geometry
            loc = GL.glGetUniformLocation(self.shader.glid, 'modelviewprojection')
            GL.glUniformMatrix4fv(loc, 1, True, projection @ view @ model)

            # texture access setups
            loc = GL.glGetUniformLocation(self.shader.glid, 'diffuseMap')
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture.glid)
            GL.glUniform1i(loc, 0)
            self.vertex_array.draw(GL.GL_TRIANGLES)

            # leave clean state for easier debugging
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            GL.glUseProgram(0)

class VertexArray:
    def __init__(self, attributes, index=None, usage=GL.GL_STATIC_DRAW):
        if len(attributes) == 0:
            print("wait what ?")
            sys.exit(1)
        # attributes is a list of np.float32 arrays, index an optional np.uint32 array
        self.buffers = []
        self.index = index

        self.buffer_len = len(attributes[0])

        self.glid = GL.glGenVertexArrays(1)  # create OpenGL vertex array id
        GL.glBindVertexArray(self.glid)      # activate to receive state below

        for layout_index, buffer_data in enumerate(attributes):
            self.buffers += [GL.glGenBuffers(1)]
            buffer_data = np.array(buffer_data, np.float32, copy=False)
            nb_primitives, size = buffer_data.shape
            GL.glEnableVertexAttribArray(layout_index)  # activates for current vao only
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[-1])
            GL.glBufferData(GL.GL_ARRAY_BUFFER, buffer_data, usage)
            GL.glVertexAttribPointer(layout_index, size, GL.GL_FLOAT, False, 0, None)

        if self.index is not None:
            self.buffers += [GL.glGenBuffers(1)]                                 # create GPU index buffer
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.buffers[-1])     # make it active to receive
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self.index, GL.GL_STATIC_DRAW) # our index array here

        # cleanup and unbind so no accidental subsequent state update
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

    def draw(self, primitive):
        GL.glBindVertexArray(self.glid)
        if self.index is None:
            # draw triangle as GL_TRIANGLE vertex array, draw array call
            GL.glDrawArrays(primitive, 0, self.buffer_len)
        else:
            # draw triangle as GL_TRIANGLE vertex array, draw array call
            GL.glDrawElements(primitive, self.index.size, GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(0)

    def __del__(self):
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(1, self.buffers)

class Node:
    """ Scene graph transform and parameter broadcast node """
    def __init__(self, name='', children=(), transform=identity(), **param):
        self.name = name
        self.param = param
        self.transform = transform
        self.children = list(iter(children))

    def add(self, *drawables):
        """ Add drawables to this node, simply updating children list """
        self.children.extend(drawables)

    def draw(self, projection, view, model, **param):
        """ Recursive draw, passing down named parameters & model matrix. """
        # merge named parameters given at initialization with those given here
        param = dict(param, **self.param)
        model = self.transform @ model # what to insert here for hierarchical update?
        for child in self.children:
            child.draw(projection, view, model, **param)

class ColorMesh(VertexArray):
    def __init__(self, attributes, index=None):
        VertexArray.__init__(self, attributes, index)

    def draw(self, projection, view, model, color_shader=None, color=(1, 1, 1, 1), **param):
        # loading of the colors into the fragment shader
        color_id_location = GL.glGetUniformLocation(color_shader.glid, 'color_id')
        GL.glUseProgram(color_shader.glid)
        GL.glUniform1i(color_id_location, color_id_location)

        # loading of the transform matrix into the shader
        matrix_location = GL.glGetUniformLocation(color_shader.glid, 'PVM')
        GL.glUniformMatrix4fv(matrix_location, 1, True, projection @ view @ model)

        model_location = GL.glGetUniformLocation(color_shader.glid, 'M')
        GL.glUniformMatrix4fv(model_location, 1, True, model)

        camera_location = GL.glGetUniformLocation(color_shader.glid, 'camera_position')
        GL.glUniform4fv(camera_location, 1, True, view[:,3])

        VertexArray.draw(self, GL.GL_TRIANGLES)

    def __del__(self):
        VertexArray.__del__(self)


class Cylinder(Node):
    """ Very simple cylinder based on practical 2 load function """
    def __init__(self):
        super().__init__()
        self.add(*load('cylinder.obj'))  # just load the cylinder from file

# ------------  Scene object classes ------------------------------------------
class SimpleTriangle:
    """Hello triangle object"""

    def __init__(self):

        # triangle position buffer
        position = np.array(((0, .5, 0), (.5, -.5, 0), (-.5, -.5, 0)), 'f')

        # triangle color buffer
        color = np.array(((1, 0, 0), (0, 1, 0) ,(0, 0, 1)), 'f')

        self.vertex_array = VertexArray([position, color], None)

    def draw(self, projection, view, model, color_shader, color_id):
        # loading of the colors into the fragment shader
        color_id_location = GL.glGetUniformLocation(color_shader.glid, 'color_id')
        GL.glUseProgram(color_shader.glid)
        GL.glUniform1i(color_id_location, color_id)

		# loading of the transform matrix into the shader
        matrix_location = GL.glGetUniformLocation(color_shader.glid, 'matrix')
        GL.glUniformMatrix4fv(matrix_location, 1, True, rotate(vec(0, 1, 1), 45))

        self.vertex_array.draw(GL.GL_TRIANGLES)

    def __del__(self):
        del self.vertex_array

class Pyramid:
    """Pyramid object"""

    def __init__(self, color_type="smooth"):
        self.color_type = (color_type == "smooth")

        if self.color_type:
            vertices = np.array(((-.5, 0, .5), (.5, 0, .5), (.5, 0, -.5), (-.5, 0, -.5), (0, 1, 0)), 'f')
            indexes = np.array((0, 1, 2, 0, 2, 3, 0, 1, 4, 0, 3, 4, 1, 2, 4, 2, 3, 4), np.uint32)
            colors = np.array(((0, 0, 1), (1, 1, 0), (1, 0, 0), (1, 0, 1), (0, 1, 0)), 'f')
            self.vertex_arrays = VertexArray([vertices, colors], index=indexes)
        else:
            vertices = np.array(((-.5, 0, .5), (.5, 0, .5), (0, 1, 0), (.5, 0, .5), (.5, 0, -.5), (0, 1, 0), (.5, 0, -.5), (-.5, 0, -.5), (0, 1, 0), (-.5, 0, -.5), (-.5, 0, .5), (0, 1, 0), (.5, 0, .5), (-.5, 0 ,.5), (.5, 0, -.5), (-.5, 0, -.5), (-.5, 0, .5), (.5, 0, -.5)), 'f')
            colors = np.array(((0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 1, 0), (0, 1, 0), (0, 1, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 0), (1, 1, 0), (0, 1, 1), (0, 1, 1), (0, 1, 1), (0, 1, 1), (0, 1, 1), (0, 1, 1)), 'f')
            self.vertex_arrays = VertexArray([vertices, colors], None)

    def draw(self, projection, view, model, color_shader, color_id):
        # loading of the colors into the fragment shader
        color_id_location = GL.glGetUniformLocation(color_shader.glid, 'color_id')
        GL.glUseProgram(color_shader.glid)
        GL.glUniform1i(color_id_location, color_id)

		# loading of the transform matrix into the shader
        if self.color_type:
            matrix_location = GL.glGetUniformLocation(color_shader.glid, 'matrix')
            projection_matrix = frustum(-.5, .5, -.5, .5, .01, 1000)
            transform_matrix = rotate(vec(1, 0, 0), 15) @ rotate(vec(0, 1, 0), -15) @ scale(.5) @ translate(-1, 0, 0)
            GL.glUniformMatrix4fv(matrix_location, 1, True, transform_matrix)
        else:
            matrix_location = GL.glGetUniformLocation(color_shader.glid, 'matrix')
            projection_matrix = frustum(-.5, .5, -.5, .5, .01, 1000)
            transform_matrix = rotate(vec(1, 0, 0), 15)  @ rotate(vec(0, 1, 0), 15) @ scale(.5) @ translate(1, 0, 0)
            GL.glUniformMatrix4fv(matrix_location, 1, True, transform_matrix)

        self.vertex_arrays.draw(GL.GL_TRIANGLES)


    def __del__(self):
        del self.vertex_arrays


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

# ------------  Viewer class & window management ------------------------------
class Viewer:
    """ GLFW viewer window, with classic initialization & graphics loop """

    def __init__(self, width=640, height=480):

        # version hints: create GL window with >= OpenGL 3.3 and core profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)
        self.win = glfw.create_window(width, height, 'Viewer', None, None)
        #init trackball
        self.trackball = GLFWTrackball(self.win)

        # make win's OpenGL context current; no OpenGL calls can happen before
        glfw.make_context_current(self.win)

        # register event handlers
        glfw.set_key_callback(self.win, self.on_key)

        # useful message to check OpenGL renderer characteristics
        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode() + ', GLSL',
              GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() +
              ', Renderer', GL.glGetString(GL.GL_RENDERER).decode())

        # initialize GL by setting viewport and default render characteristics
        GL.glClearColor(0.5, 0.5, 0.5, 0.5)

        # compile and initialize shader programs once globally
        self.color_shader = Shader(COLOR_VERT, COLOR_FRAG)

        # initially empty list of object to draw
        self.drawables = []

        self.color_id = 0

        GL.glEnable(GL.GL_DEPTH_TEST)

    def run(self):
        """ Main render loop for this OpenGL window """
        while not glfw.window_should_close(self.win):
            # clear draw buffer
            GL.glClear(GL.GL_COLOR_BUFFER_BIT|GL.GL_DEPTH_BUFFER_BIT)


            winsize = glfw.get_window_size(self.win)
            view = self.trackball.view_matrix()
            projection = self.trackball.projection_matrix(winsize)

            # draw our scene objects
            for drawable in self.drawables:
                drawable.draw(projection, view, identity(), win=self.win)
                #drawable.draw(projection, view, identity(), color_shader=self.color_shader)
                #drawable.draw(projection, view, identity, self.color_shader, self.color_id)

            # flush render commands, and swap draw buffers
            glfw.swap_buffers(self.win)

            # Poll for and process events
            glfw.poll_events()

    def add(self, *drawables):
        """ add objects to draw in this window """
        self.drawables.extend(drawables)

    def on_key(self, _win, key, _scancode, action, _mods):
        """ 'Q' or 'Escape' quits """
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(self.win, True)
            elif key == glfw.KEY_N:
                self.color_id = (self.color_id + 1) % 3


# -------------- 3D ressource loader -----------------------------------------
def load(file):
    """ load resources from file using pyassimp, return list of ColorMesh """
    try:
        option = pyassimp.postprocess.aiProcessPreset_TargetRealtime_MaxQuality
        scene = pyassimp.load(file, option)
    except pyassimp.errors.AssimpError:
        print('ERROR: pyassimp unable to load', file)
        return []     # error reading => return empty list

    meshes = [ColorMesh([m.vertices, m.normals], m.faces) for m in scene.meshes]
    size = sum((mesh.faces.shape[0] for mesh in scene.meshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(scene.meshes), size))

    pyassimp.release(scene)
    return meshes

def load_textured(file):
    """ load resources using pyassimp, return list of TexturedMeshes """
    try:
        option = pyassimp.postprocess.aiProcessPreset_TargetRealtime_MaxQuality
        scene = pyassimp.load(file, option)
    except pyassimp.errors.AssimpError:
        print('ERROR: pyassimp unable to load', file)
        return []  # error reading => return empty list

    # Note: embedded textures not supported at the moment
    path = os.path.dirname(file)
    path = os.path.join('.', '') if path == '' else path
    for mat in scene.materials:
        mat.tokens = dict(reversed(list(mat.properties.items())))
        if 'file' in mat.tokens:  # texture file token
            tname = mat.tokens['file'].split('/')[-1].split('\\')[-1]
            # search texture in file's whole subdir since path often screwed up
            tname = [os.path.join(d[0], f) for d in os.walk(path) for f in d[2]
                     if tname.startswith(f) or f.startswith(tname)]
            if tname:
                mat.texture = tname[0]
            else:
                print('Failed to find texture:', tname)

    # prepare textured mesh
    meshes = []
    for mesh in scene.meshes:
        texture = scene.materials[mesh.materialindex].texture

        # tex coords in raster order: compute 1 - y to follow OpenGL convention
        tex_uv = ((0, 1) + mesh.texturecoords[0][:, :2] * (1, -1)
                  if mesh.texturecoords.size else None)

        # create the textured mesh object from texture, attributes, and indices
        meshes.append(TexturedMesh(Texture(texture), [mesh.vertices, tex_uv], mesh.faces))

    size = sum((mesh.faces.shape[0] for mesh in scene.meshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(scene.meshes), size))

    pyassimp.release(scene)
    return meshes

# -------------- main program and scene setup --------------------------------
def main():
    """ create a window, add scene objects, then run rendering loop """
    viewer = Viewer()

    # place instances of our basic objects
    # viewer.add(SimpleTriangle())
    #viewer.add(Pyramid())
    #viewer.add(Pyramid(color_type="sharp"))

    #meshes = load("suzanne.obj")
    #for m in meshes:
    #    viewer.add(m)
    # start rendering loop

    #cylinder_node = Node(name='my_cylinder', transform=translate(-1, 0, 0), color=(1, 0, 0.5, 1))
    #cylinder_node.add(Cylinder())pss

    meshes = load_textured("bunny.obj")
    for m in meshes:
        viewer.add(m)

    # viewer.add(TexturedPlane("grass.png"))

    # construct our robot arm hierarchy for drawing in viewer
    # cylinder = Cylinder()             # re-use same cylinder instance
    # limb_shape = Node(transform=identity() @ translate(0, 1, 0) @ scale(0.5, 1, 0.5))  # make a thin cylinder
    # limb_shape.add(cylinder)          # common shape of arm and forearm
    #
    # arm_node = Node(transform=identity() @ translate(0, 2, 0) @ scale(0.7, 0.7, 0.7), children=[cylinder])    # robot arm rotation with phi angle
    # arm_node.add(limb_shape)
    #
    # # make a flat cylinder
    # base_shape = Node(transform=identity(), children=[cylinder])
    # base_node = Node(transform=identity())   # robot base rotation with theta angle
    # base_node.add(base_shape, arm_node)
    # viewer.add(base_node)

    viewer.run()


if __name__ == '__main__':
    glfw.init()                # initialize window system glfw
    main()                     # main function keeps variables locally scoped
    glfw.terminate()           # destroy all glfw windows and GL contexts
