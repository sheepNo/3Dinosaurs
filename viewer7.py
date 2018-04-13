#!/usr/bin/env python3
"""
Python OpenGL practical application.
"""
# Python built-in modules
import os                           # os function, i.e. checking file status
from itertools import cycle
import sys
from bisect import bisect_left      # search sorted keyframe lists

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
import numpy as np                  # all matrix manipulations & OpenGL args
import pyassimp                     # 3D ressource loader
import pyassimp.errors              # assimp error management + exceptions

from transform import translate, scale, identity, Trackball, sincos, vec
from transform import (lerp, quaternion_slerp, quaternion_matrix, quaternion,
                       quaternion_from_euler)

from texture import Texture
from model_loading import Node, VertexArray
from shaders import Shader

# -------------- Keyframing Utilities TP6 ------------------------------------
# Basic keyframe interpolation
class KeyFrames:
    """ Stores keyframe pairs for any value type with interpolation_function"""
    def __init__(self, time_value_pairs, interpolation_function=lerp):
        if isinstance(time_value_pairs, dict):  # convert to list of pairs
            time_value_pairs = time_value_pairs.items()
        keyframes = sorted(((key[0], key[1]) for key in time_value_pairs))
        self.times, self.values = zip(*keyframes)  # pairs list -> 2 lists
        self.interpolate = interpolation_function

    def value(self, time):
        """ Computes interpolated value from keyframes, for a given time """

        # 1. ensure time is within bounds else return boundary keyframe
        if time < self.times[0]:
            return self.values[0]
        if time > self.times[-1]:
            return self.values[-1]
        # 2. search for closest index entry in self.times, using bisect_left function
        index = bisect_left(self.times, time)
        # 3. using the retrieved index, interpolate between the two neighboring values
        # in self.values, using the initially stored self.interpolate function
        return self.interpolate(self.values[index-1], self.values[index], (time-self.times[index-1])/(self.times[index]-self.times[index-1]))

# Transformation interpolation
class TransformKeyFrames:
    """ KeyFrames-like object dedicated to 3D transforms """
    def __init__(self, translate_keys, rotate_keys, scale_keys):
        """ stores 3 keyframe sets for translation, rotation, scale """
        self.translation = KeyFrames(translate_keys, lerp)
        self.rotation = KeyFrames(rotate_keys, quaternion_slerp)
        self.scale = KeyFrames(scale_keys, lerp)

    def value(self, time):
        """ Compute each component's interpolation and compose TRS matrix """
        T = self.translation.value(time)
        R = quaternion_matrix(self.rotation.value(time))
        S = self.scale.value(time)

        TRS = np.zeros([4,4])

        TRS[3][3] = 1.

        for i in range(3):
            TRS[i][3] = T[i]

        for i in range(3):
            for j in range(3):
                TRS[i][j]=R[i][j]*S[i]

        return TRS

# Node for keyframing animation
class KeyFrameControlNode(Node):
    """ Place node with transform keys above a controlled subtree """
    def __init__(self, trans_keys, rotat_keys, scale_keys, **kwargs):
        super().__init__(**kwargs)
        self.keyframes = TransformKeyFrames(trans_keys, rotat_keys, scale_keys)

    def draw(self, projection, view, model, **param):
        """ When redraw requested, interpolate our node transform from keys """
        self.transform = self.keyframes.value(glfw.get_time())
        super().draw(projection, view, model, **param)


# -------------- Linear Blend Skinning : TP7 ---------------------------------
MAX_VERTEX_BONES = 4
MAX_BONES = 128

# new shader for skinned meshes, fully compatible with previous color fragment
SKINNING_TEXTURE_VERT = """#version 330 core
// ---- camera geometry
uniform mat4 modelviewprojection;

// ---- skinning globals and attributes
const int MAX_VERTEX_BONES=%d, MAX_BONES=%d;
uniform mat4 boneMatrix[MAX_BONES];

// ---- vertex attributes
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec4 bone_ids;
layout(location = 3) in vec4 bone_weights;
layout(location = 4) in vec2 tex_uv;

// ----- interpolated attribute variables to be passed to fragment shader
out vec2 fragTexCoord;

void main() {

    // ------ creation of the skinning deformation matrix
    mat4 skinMatrix;
    for (int i=0;i<4;i++) {
        for (int j=0; j<4; j++) {
            skinMatrix[i][j] = 0;
        }
    }
    mat4 weightMatrix;
        int id;
        for (int k=0; k<4; k++){
        id = int(bone_ids[k]);
        for (int i=0;i<4;i++) {
            for (int j=0; j<4; j++) {
                weightMatrix[i][j] = 0;
            }
        }
        for (int i=0;i<4;i++) {
            weightMatrix[i][i] = bone_weights[k];
        }
        skinMatrix = skinMatrix + (bone_weights[k]*boneMatrix[id]);
    }  //complete shader here for exercise 1!
    // mat4 skinMatrix = mat4(1.);  // TODO complete shader here for exercise 1!

    // ------ compute world and normalized eye coordinates of our vertex
    vec4 wPosition4 = skinMatrix * vec4(position, 1.0);
    gl_Position = modelviewprojection * wPosition4;

    fragTexCoord = tex_uv;
}
""" % (MAX_VERTEX_BONES, MAX_BONES)

TEXTURE_FRAG = """#version 330 core
uniform sampler2D diffuseMap;
in vec2 fragTexCoord;
out vec4 outColor;
void main() {
    outColor = texture(diffuseMap, fragTexCoord);
}"""

class SkinnedMesh:
    """class of skinned mesh nodes in scene graph """
    def __init__(self, attributes, texture, bone_nodes, bone_offsets, index=None):

        # setup shader attributes for linear blend skinning shader
        self.vertex_array = VertexArray(attributes, index)

        self.skinning_texture_shader = Shader(SKINNING_TEXTURE_VERT, TEXTURE_FRAG)

        # store skinning data
        self.bone_nodes = bone_nodes
        self.bone_offsets = bone_offsets
        # for texture
        self.texture = texture


    def draw(self, projection, view, model, **_kwargs):
        """ skinning object draw method """

        shid = self.skinning_texture_shader.glid
        GL.glUseProgram(shid)

        # setup camera geometry parameters
        loc = GL.glGetUniformLocation(shid, 'modelviewprojection')
        GL.glUniformMatrix4fv(loc, 1, True, projection @ view @ model)

        # bone world transform matrices need to be passed for skinning
        for bone_id, node in enumerate(self.bone_nodes):
            bone_matrix = node.world_transform @ self.bone_offsets[bone_id]

            bone_loc = GL.glGetUniformLocation(shid, 'boneMatrix[%d]' % bone_id)
            GL.glUniformMatrix4fv(bone_loc, 1, True, bone_matrix)

        # texture access setups
        loc = GL.glGetUniformLocation(shid, 'diffuseMap')
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture.glid)
        GL.glUniform1i(loc, 0)

        # draw mesh vertex array
        self.vertex_array.draw(GL.GL_TRIANGLES)

        # leave with clean OpenGL state, to make it easier to detect problems
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUseProgram(0)


# -------- Skinning Control for Keyframing Skinning Mesh Bone Transforms ------
class SkinningControlNode(Node):
    """ Place node with transform keys above a controlled subtree """
    def __init__(self, *keys, **kwargs):
        super().__init__(**kwargs)
        self.keyframes = TransformKeyFrames(*keys) if keys[0] else None
        self.world_transform = identity()

    def draw(self, projection, view, model, **param):
        """ When redraw requested, interpolate our node transform from keys """
        if self.keyframes:  # no keyframe update should happens if no keyframes
            self.transform = self.keyframes.value(glfw.get_time())

        # store world transform for skinned meshes using this node as bone
        self.world_transform = model @ self.transform

        # default node behaviour (call children's draw method)
        super().draw(projection, view, model, **param)


# -------------- 3D resource loader -------------------------------------------
def load_skinned(file):
    """load resources from file using pyassimp, return node hierarchy """
    try:
        option = pyassimp.postprocess.aiProcessPreset_TargetRealtime_MaxQuality
        scene = pyassimp.load(file, option)
    except pyassimp.errors.AssimpError:
        print('ERROR: pyassimp unable to load', file)
        return []

    # ----- load animations
    def conv(assimp_keys, ticks_per_second):
        """ Conversion from assimp key struct to our dict representation """
        return {key.time / ticks_per_second: key.value for key in assimp_keys}

    # load first animation in scene file (could be a loop over all animations)
    transform_keyframes = {}
    if scene.animations:
        anim = scene.animations[0]
        for channel in anim.channels:
            # for each animation bone, store trs dict with {times: transforms}
            # (pyassimp name storage bug, bytes instead of str => convert it)
            transform_keyframes[channel.nodename.data.decode('utf-8')] = (
                conv(channel.positionkeys, anim.tickspersecond),
                conv(channel.rotationkeys, anim.tickspersecond),
                conv(channel.scalingkeys, anim.tickspersecond)
            )

    # ---- prepare scene graph nodes
    # create SkinningControlNode for each assimp node.
    # node creation needs to happen first as SkinnedMeshes store an array of
    # these nodes that represent their bone transforms
    nodes = {}  # nodes: string name -> node dictionary

    def make_nodes(pyassimp_node):
        """ Recursively builds nodes for our graph, matching pyassimp nodes """
        trs_keyframes = transform_keyframes.get(pyassimp_node.name, (None,))

        node = SkinningControlNode(*trs_keyframes, name=pyassimp_node.name,
                                   transform=pyassimp_node.transformation)
        nodes[pyassimp_node.name] = node, pyassimp_node
        node.add(*(make_nodes(child) for child in pyassimp_node.children))
        return node

    root_node = make_nodes(scene.rootnode)
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

    # ---- create SkinnedMesh objects
    for mesh in scene.meshes:
        # -- skinned mesh: weights given per bone => convert per vertex for GPU
        # first, populate an array with MAX_BONES entries per vertex
        v_bone = np.array([[(0, 0)]*MAX_BONES] * mesh.vertices.shape[0],
                          dtype=[('weight', 'f4'), ('id', 'u4')])
        for bone_id, bone in enumerate(mesh.bones[:MAX_BONES]):
            for entry in bone.weights:  # weight,id pairs necessary for sorting
                v_bone[entry.vertexid][bone_id] = (entry.weight, bone_id)

        v_bone.sort(order='weight')             # sort rows, high weights last
        v_bone = v_bone[:, -MAX_VERTEX_BONES:]  # limit bone size, keep highest

        # prepare bone lookup array & offset matrix, indexed by bone index (id)
        bone_nodes = [nodes[bone.name][0] for bone in mesh.bones]
        bone_offsets = [bone.offsetmatrix for bone in mesh.bones]

        if mesh.materialindex == 1:
            texture =  scene.materials[1].texture
        if mesh.materialindex == 2:
            texture = scene.materials[2].texture
        print(texture)
        # tex coords in raster order: compute 1 - y to follow OpenGL convention
        tex_uv = ((0, 1) + mesh.texturecoords[0][:, :2] * (1, -1)
                  if mesh.texturecoords.size else None)

        # initialize skinned mesh and store in pyassimp_mesh for node addition
        mesh.skinned_mesh = SkinnedMesh(
                [mesh.vertices, mesh.normals, v_bone['id'], v_bone['weight'], tex_uv],
                Texture(texture), bone_nodes, bone_offsets, mesh.faces
        )


    # ------ add each mesh to its intended nodes as indicated by assimp
    for final_node, assimp_node in nodes.values():
        final_node.add(*(_mesh.skinned_mesh for _mesh in assimp_node.meshes))

    nb_triangles = sum((mesh.faces.shape[0] for mesh in scene.meshes))
    print('Loaded', file, '\t(%d meshes, %d faces, %d nodes, %d animations)' %
          (len(scene.meshes), nb_triangles, len(nodes), len(scene.animations)))
    pyassimp.release(scene)
    return [root_node]


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
