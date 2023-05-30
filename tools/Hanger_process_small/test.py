#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:25:11 2023

@author: luyin
"""
import taichi as ti
ti.init(arch = ti.cpu, cpu_max_num_threads=1)

x = 0

@ti.kernel
def math1():
    for i in ti.static(range(3)):
        y = i
        print(x[None],i)
    print(x[None])

math1()