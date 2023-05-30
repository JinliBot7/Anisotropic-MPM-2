#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 16:53:31 2022

@author: luyin
"""

import taichi as ti
ti.init(ti.cpu)

floor = ti.Vector.field(3,ti.f32,shape = 4)
floor_index = ti.field(ti.f32,shape = 6)
floor[0] = [0.0, 0.0, 0.0]
floor[1] = [1.0, 0.0, 0.0]
floor[2] = [0.0, 1.0, 0.0]
floor[3] = [1.0, 1.0, 0.0]
floor_index = [0, 1, 3, 3, 2, 0]

print(floor_index)