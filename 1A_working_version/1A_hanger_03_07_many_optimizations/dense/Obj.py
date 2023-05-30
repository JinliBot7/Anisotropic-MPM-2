#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 21:48:39 2023

@author: luyin
"""
import taichi as ti

real = ti.f32
scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(3, dtype=real)
vec_int = lambda: ti.Vector.field(3, dtype=ti.i32)
mat = lambda: ti.Matrix.field(3, 3, dtype=real)

@ti.data_oriented
class Obj:
    def __init__(self):
        self.n = 2
        self.x = vec()
        ti.root.pointer(ti.l, 10).dense(ti.axes(4), self.n).place(self.x)