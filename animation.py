#!/usr/bin/env python
"""
Module encapsulating animation
"""

import glfw
import numpy as np

from transform import (lerp, vec, quaternion_slerp, quaternion_matrix,
                        quaternion, quaternion_from_euler)
from bisect import bisect_left      # search sorted keyframe lists

from model_loading import Node

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
                TRS[i][j]=R[i][j]*S

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



""" Testing """
if __name__ == "__main__":
    my_keyframes = KeyFrames({0: 1, 3: 7, 6: 20})
    print(my_keyframes.value(1.5))

    vector_keyframes = KeyFrames({0: vec(1, 0, 0), 3: vec(0, 1, 0), 6: vec(0, 0, 1)})
    print(vector_keyframes.value(1.5))   # should display numpy vector (0.5, 0.5, 0)

    from transform import quaternion_from_axis_angle

    translate_keys = {0: vec(0, 0, 0), 1: vec(0, 0, 0)}
    rotate_keys = {0: quaternion_from_axis_angle(vec(1, 0, 0), 0), 1: quaternion_from_axis_angle(vec(1, 0, 0), 90)}
    scale_keys = {0: 1, 1: 1}
    transform_keyframes = TransformKeyFrames(translate_keys, rotate_keys, scale_keys)
    print(transform_keyframes.value(0.5))
