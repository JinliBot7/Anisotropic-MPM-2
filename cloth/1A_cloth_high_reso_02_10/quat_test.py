#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 13:57:37 2023

@author: luyin
"""

import taichi as ti
ti.init(ti.cpu)

def axis_to_quaternion(arg):
    sin_theta = ti.math.sin(arg[0]/2)
    return ti.Vector([ti.math.cos(arg[0]/2),sin_theta * arg[1], sin_theta * arg[2], sin_theta * arg[3]])

def quaternion_to_axis(arg):
    length = (arg[0] ** 2 + arg[1] ** 2 + arg[2] ** 2 + arg[3] ** 2) ** 0.5
    a, b, c, d = arg[0]/length, arg[1]/length, arg[2]/length, arg[3]/length
    #print(a, b, c, d, length)

    theta = 2 * ti.math.acos(a)
    x = b / (ti.math.sin(theta/2))
    y = c / (ti.math.sin(theta/2))
    z = d / (ti.math.sin(theta/2))
    axis_angle = [theta / ti.math.pi * 180, x, y, z]
    return axis_angle

def quat_mul(q1, q2):# quaternion multiplication from wiki
    a1, b1, c1, d1 = q1[0], q1[1], q1[2], q1[3]
    a2, b2, c2, d2 = q2[0], q2[1], q2[2], q2[3]
    t1 = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
    t2 = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
    t3 = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
    t4 = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2
    result = [t1, t2, t3, t4]
    return result
# @ti.kernel
# def rotate_z(counter:ti.i32):
#     obj_0.rotate(to_quaternion([1e-2, 0.0, 0.0, 1.0]))
A = [90.0 / 180 * ti.math.pi, 1.0, 0.0, 0.0]
B = [135.0 / 180 * ti.math.pi, 0.0, 1.0, 0.0]
quat_A = axis_to_quaternion(A)
quat_B = axis_to_quaternion(B)
#print(quat_A, quat_B)  
quat_final = quat_mul(quat_A,quat_B)

axis_final_2 = quaternion_to_axis(quat_final)
# #print(quat_A)
# print(axis_final)