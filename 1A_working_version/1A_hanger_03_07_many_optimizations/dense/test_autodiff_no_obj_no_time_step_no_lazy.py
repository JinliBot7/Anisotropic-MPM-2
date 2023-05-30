#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 21:47:35 2023

@author: luyin
"""

import taichi as ti

ti.init(debug = True)

max_step = 3
n = 2

real = ti.f32
scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(3, dtype=real)

loss = ti.field(real,shape = (), needs_grad = True)
x_avg = ti.Vector.field(3,real, shape = (), needs_grad = True)
x = ti.Vector.field(3,real, shape = n, needs_grad = True)
v = ti.Vector.field(3,real,shape = (), needs_grad = True)
target = [0.5, 0.5, 0.5]


# loss 
def compute_loss():
    compute_x_avg()
    dist = (x_avg[None] - ti.Vector(target))
    #loss[None] =  0.5 * (dist[0] + dist[1] + dist[2])
    loss[None] = v[None][0]

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
    with ti.ad.Tape(loss=loss, validation=True):
        #v[None] = [1.0, 1.0, 1.0]
        forward()
        compute_loss()
        
    print(x_avg[None])
    print(loss[None])
    print(v.grad)

if __name__ == "__main__":
    main()

























