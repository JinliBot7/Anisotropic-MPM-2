#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 11:06:52 2022

@author: luyin
"""
import taichi as ti
ti.init(ti.gpu)

@ti.kernel
def test():
    A = 1
    A = 2
    print(A)

test()