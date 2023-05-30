#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 14:46:05 2023

@author: luyin
"""
import taichi as ti
ti.init()

real = ti.f32
scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(3, dtype=real)

target = ti.Vector([0.5, 0.5, 0.5])

n = 2

v = vec()
loss = scalar()
ti.root.place(v, loss)
ti.root.lazy_grad()

v[None] = [2.0, 0.0, 0.0]



def compute_loss():
    loss[None] += v[None][0]




    
with ti.ad.Tape(loss):
    compute_loss()

print('dloss/dv =', v.grad)

