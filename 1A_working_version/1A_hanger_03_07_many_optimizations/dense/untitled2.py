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

n = 2 # particle number

x, v, x_avg = vec(), vec(), vec()
loss = scalar()
ti.root.place(v, loss, x_avg)
ti.root.dense(ti.i, n).place(x)
#v[None] = [2.0, 1.0, 3.0]
ti.root.lazy_grad()
v[None] = [2.0, 1.0, 3.0]


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
        x[p] += v[None]


    
with ti.ad.Tape(loss):
    substep()
    compute_x_avg()
    compute_loss()

print('dloss/dv =', v.grad)

