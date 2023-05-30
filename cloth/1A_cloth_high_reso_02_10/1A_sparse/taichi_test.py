#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:58:01 2023

@author: luyin
"""

import taichi as ti

ti.init(ti.cuda)

A = ti.field(ti.f32, shape = ())
A[None] = 0
B = 3

@ti.kernel
def test():
    for i in range(3):
        A[None] += 1
        print(B)

test()
#print(A[None])