#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 13:11:17 2023

@author: luyin
"""
import taichi as ti
from taichi.math import cos, sin, pi

ti.init(ti.cpu)

@ti.func
def quat_from_vecs(vec1,z_axis):
    angle = ti.math.acos(vec1.dot(z_axis))
    axis = (z_axis.cross(vec1)) / ti.math.sin(angle)
    axis_angle = ti.Vector([angle,axis[0],axis[1],axis[2]])
    #print(axis_angle)
    quat = to_quaternion_func(axis_angle)
    return quat

@ti.func
def to_quaternion_func(arg):
    sin_theta = sin(arg[0]/2)
    return ti.Vector([cos(arg[0]/2),sin_theta * arg[1], sin_theta * arg[2], sin_theta * arg[3]])

@ti.func
def quaternion_2_R(quaternion: ti.types.vector(4,ti.f32)):
    s,x,y,z = quaternion[0],quaternion[1],quaternion[2],quaternion[3]
    r11 = 1 - 2 * y ** 2 - 2 * z ** 2
    r12 = 2 * x * y - 2 * s * z
    r13 = 2 * x * z + 2 * s * y
    r21 = 2 * x * y + 2 * s * z
    r22 = 1 - 2 * x ** 2 - 2 * z ** 2
    r23 = 2 * y * z - 2 * s * x
    r31 = 2 * x * z - 2 * s * y
    r32 = 2 * y * z + 2 * s * x
    r33 = 1 - 2 * x ** 2 - 2 * y ** 2
    R = ti.Matrix([[r11, r12, r13],
         [r21, r22, r23],
         [r31, r32, r33],
                         ])
    return R

@ti.kernel
def test():
    vec1 = ti.Vector([0.5 * 2 **0.5, 0.5 * 2 **0.5, 0.0])
    z_axis = ti.Vector([0.0, 0.0, 1.0])
    quat = quat_from_vecs(vec1,z_axis)
    print(quat)
    R = quaternion_2_R(quat)
    #print(R)
    z = R @ ti.Vector([0.0, 0.0, 1.0])
    print(z)

test()