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

loss, v = scalar(), scalar()
ti.root.place(v, loss)
ti.root.lazy_grad()

v[None] = 2.0

@ti.kernel
def compute_loss():
    for p in range(1):
        loss[None] += v[None]

def compute_loss_not_kernel():
    for p in range(1):
        loss[None] += v[None]



    
with ti.ad.Tape(loss):
    compute_loss_not_kernel()

print('dloss/dv =', v.grad)

