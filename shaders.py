"""
Module for the Shader class and all the shaders declaration
"""

import os
import OpenGL.GL as GL              # standard Python OpenGL wrapper

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

    def __init__(self, vertex_source, fragment_source, uniform_loader=None):
        """ Shader can be initialized with raw strings or source file names """
        self.glid = None
        self.uniform_loader = uniform_loader # uniform_loader is a function to be called for special shader uniform loading
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


def create_uniform_loader(names, data_types):
    """ Returns a uniform_loader for the given names and types """
    def loader(color_shader, values):
        loc = {n: GL.glGetUniformLocation(color_shader.glid, n) for n in names}

        for name, value, data_type in zip(names, values, data_types):
            if data_type == "vec4f":
                local_loader = GL.glUniform4f

            local_loader(loc[name], *value)

    return loader


def load_shaders():
    """ Loads all the shaders of this file (need to be add manually)
        Returns a dictionary of shaders with their name """
    shaders = {}

    shaders['simple'] = Shader(SIMPLE_COLOR_VERT, SIMPLE_COLOR_FRAG)
    shaders['color'] = Shader(UNIFORM_COLOR_VERT, UNIFORM_COLOR_FRAG, create_uniform_loader(['color'], ['vec4f']))

    return shaders

# ------------  simple color fragment shader ------
SIMPLE_COLOR_VERT = """#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
out vec3 fragColor;

void main() {
    gl_Position = projection * view * model * vec4(position, 1);
    fragColor = color;
}"""

SIMPLE_COLOR_FRAG = """#version 330 core
in vec3 fragColor;
out vec4 outColor;
void main() {
    outColor = vec4(fragColor, 1);
}"""

UNIFORM_COLOR_VERT = """#version 330 core
layout(location = 0) in vec3 position;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec4 color;
out vec4 fragColor;

void main() {
    gl_Position = projection * view * model * vec4(position, 1);
    fragColor = color;
}"""

UNIFORM_COLOR_FRAG = """#version 330 core
in vec4 fragColor;
out vec4 outColor;
void main() {
    outColor = fragColor;
}"""
