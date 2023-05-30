#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 16:18:26 2023

@author: luyin
"""

import taichi as ti
ti.init(ti.cpu)

@ti.kernel
def test():
    test_func

@ti.func
def test_func():
    A = ti.Vector([0.0, 0.0, 0.0])
    print(A)
test()