#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 17:07:08 2023

@author: luyin
"""

import taichi as ti
import numpy as np
ti.init()

max_iter = 5
v_input_num = 10
#dv = np.zeros((max_iter,v_input_num,3))

dv = ti.ndarray(dtype=ti.math.vec3, shape=(max_iter, v_input_num))
print(dv[0,0])

A = ti.Vector([1.0, 1.0, 1.0])
print(A)
B = A + dv[0,0]
print(B)