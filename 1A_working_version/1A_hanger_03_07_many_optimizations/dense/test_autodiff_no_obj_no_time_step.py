#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 21:47:35 2023

@author: luyin
"""

import taichi as ti

ti.init(ti.cpu, cpu_max_num_threads=1)

max_step = 3
n = 2

real = ti.f32
scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(3, dtype=real)

loss = scalar()
x_avg = vec()
x = vec()
v = vec()
target = [0.5, 0.5, 0.5]


# loss 
def compute_loss():
    compute_x_avg()
    dist = (x_avg[None] - ti.Vector(target))
    loss[None] =  0.5 * (dist[0] + dist[1] + dist[2])
@ti.kernel
def compute_x_avg():
    for p in range(n):
        x_avg[None] +=  (1 / n) * x[p]

# simulation
@ti.kernel
def substep():
    for p in range(n):
        x[p] += v[None]
def forward():
    substep()

def main():
    #ti.root.pointer(ti.l, max_step).dense(ti.axes(4), n).place(x)
    ti.root.dense(ti.i, n).place(x)
    ti.root.place(x_avg, loss, v)
    ti.root.lazy_grad()
    with ti.ad.Tape(loss=loss):
        v[None] = [1.0, 1.0, 1.0]
        forward()
        compute_loss()
        
    print(x)
    print(loss[None])
    print(v.grad)

if __name__ == "__main__":
    main()

























