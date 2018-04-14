#!/usr/bin/env python
"""
Module encapsulating animation
"""

import glfw
import numpy as np
import pyassimp                     # 3D ressource loader
import pyassimp.errors              # assimp error management + exceptions
import OpenGL.GL as GL

from transform import (lerp, vec, quaternion_slerp, quaternion_matrix,
                        quaternion, quaternion_from_euler)
from transform import identity
from bisect import bisect_left      # search sorted keyframe lists

from model_loading import Node, VertexArray

from shaders import Shader, SIMPLE_COLOR_FRAG

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
    def __init__(self, translate_keys, rotate_keys, scale_keys, **kwargs):
        super().__init__(**kwargs)
        self.keyframes = TransformKeyFrames(translate_keys, rotate_keys, scale_keys)

    def draw(self, projection, view, model, **param):
        """ When redraw requested, interpolate our node transform from keys """
        self.transform = self.keyframes.value(glfw.get_time())
        super().draw(projection, view, model, **param)

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

class SkinnedMesh:
    """class of skinned mesh nodes in scene graph """
    def __init__(self, attributes, bone_nodes, bone_offsets, index=None):

        # setup shader attributes for linear blend skinning shader
        self.vertex_array = VertexArray(attributes, index)

        # feel free to move this up in Viewer as shown in previous practicals
        self.skinning_shader = Shader(SKINNING_VERT, SIMPLE_COLOR_FRAG)

        # store skinning data
        self.bone_nodes = bone_nodes
        self.bone_offsets = bone_offsets

    def draw(self, projection, view, _model, **_kwargs):
        """ skinning object draw method """

        shid = self.skinning_shader.glid
        GL.glUseProgram(shid)

        # setup camera geometry parameters
        loc = GL.glGetUniformLocation(shid, 'projection')
        GL.glUniformMatrix4fv(loc, 1, True, projection)
        loc = GL.glGetUniformLocation(shid, 'view')
        GL.glUniformMatrix4fv(loc, 1, True, view)

        # bone world transform matrices need to be passed for skinning
        for bone_id, node in enumerate(self.bone_nodes):
            bone_matrix = node.world_transform @ self.bone_offsets[bone_id]

            bone_loc = GL.glGetUniformLocation(shid, 'boneMatrix[%d]' % bone_id)
            GL.glUniformMatrix4fv(bone_loc, 1, True, bone_matrix)

        # draw mesh vertex array
        self.vertex_array.draw(GL.GL_TRIANGLES)

        # leave with clean OpenGL state, to make it easier to detect problems
        GL.glUseProgram(0)

# 3D Loading
MAX_VERTEX_BONES = 4
MAX_BONES = 128
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

        # initialize skinned mesh and store in pyassimp_mesh for node addition
        mesh.skinned_mesh = SkinnedMesh(
                [mesh.vertices, mesh.normals, v_bone['id'], v_bone['weight']],
                bone_nodes, bone_offsets, mesh.faces
        )

    # ------ add each mesh to its intended nodes as indicated by assimp
    for final_node, assimp_node in nodes.values():
        final_node.add(*(_mesh.skinned_mesh for _mesh in assimp_node.meshes))

    nb_triangles = sum((mesh.faces.shape[0] for mesh in scene.meshes))
    print('Loaded', file, '\t(%d meshes, %d faces, %d nodes, %d animations)' %
          (len(scene.meshes), nb_triangles, len(nodes), len(scene.animations)))
    pyassimp.release(scene)
    return [root_node]

SKINNING_VERT = """#version 330 core
// ---- camera geometry
uniform mat4 projection, view;

// ---- skinning globals and attributes
const int MAX_VERTEX_BONES=%d, MAX_BONES=%d;
uniform mat4 boneMatrix[MAX_BONES];

// ---- vertex attributes
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec4 bone_ids;
layout(location = 3) in vec4 bone_weights;

// ----- interpolated attribute variables to be passed to fragment shader
out vec3 fragColor;

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
    gl_Position = projection * view * wPosition4;

    fragColor = color;
}
""" % (MAX_VERTEX_BONES, MAX_BONES)
