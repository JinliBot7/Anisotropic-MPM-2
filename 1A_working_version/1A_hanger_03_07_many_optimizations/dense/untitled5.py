#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 14:38:06 2023

@author: luyin
"""

import taichi as ti
ti.init()

real = ti.f32
scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(3, dtype=real)

target = ti.Vector([0.5, 0.5, 0.5])

n = 2 # particle number

x, v, x_avg = vec(), vec(), vec()
loss = scalar()
ti.root.place(loss, x_avg)
ti.root.dense(ti.i, n).place(x,v)

ti.root.lazy_grad()

@ti.kernel
def ini_v():
    for p in v:
        v[p] = [0.1, 0.2, 0.3]
        #v[p] += [0.1, 0.2, 0.3] # if ini_v() is inside tape, use atomic add so the v.grad will be non-zero

@ti.kernel
def compute_x_avg():
    for p in x:
        x_avg[None] += 1 / n * x[p]

@ti.kernel
def compute_loss():
    dist = x_avg[None] - target
    loss[None] = (dist[0] ** 2 + dist[1] ** 2 + dist[2] ** 2) ** 0.5

@ti.kernel
def substep():
    for p in x:
        x[p] += v[p]


ini_v()    
with ti.ad.Tape(loss):
    #ini_v() # v.grad will become 0!
    substep()
    compute_x_avg()
    compute_loss()

print('dloss/dv =', v.grad)