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
v = scalar()
target = [0.5, 0.5, 0.5]


# loss 
@ti.kernel
def compute_loss():
    dist = (x_avg[None] - ti.Vector(target)) ** 2
    loss[None] =  0.5 * (dist[0] + dist[1] + dist[2])

@ti.kernel
def compute_x_avg():
    for p in range(n):
        x_avg[None] +=  (1 / n) * x[max_step - 1, p]

# simulation
@ti.func
def substep():
    for p in range(n):
        x[max_step - 1, p] += v[None]

@ti.kernel
def forward():
    substep()

def main():
    ti.root.dense(ti.i, max_step).dense(ti.j, n).place(x)
    ti.root.place(x_avg, loss, v)
    ti.root.lazy_grad()
    with ti.ad.Tape(loss=loss):
        v[None] = 3
        forward()
        compute_x_avg()
        compute_loss()

    print(loss[None])
    print(x_avg.grad)

if __name__ == "__main__":
    main()

























