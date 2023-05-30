#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 21:47:35 2023

@author: luyin
"""

import taichi as ti
from Obj import Obj

ti.init(ti.cpu, cpu_max_num_threads=1)

max_step = 10

real = ti.f32
scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(3, dtype=real)

loss = scalar()
x_avg = vec()
#x_this = vec()
#ti.root.pointer(ti.l, max_step).dense(ti.axes(4), 2).place(x_this)
v = vec()
target = [0.5, 0.5, 0.5]



@ti.kernel
def compute_loss(obj:ti.template()):
    dist = (x_avg[None] - ti.Vector(target)) ** 2
    loss[None] =  0.5 * (dist[0] + dist[1] + dist[2])

@ti.kernel
def compute_x_avg(obj: ti.template()):
    for p in range(obj.n):
        x_avg[None] +=  (1 / obj.n) * obj.x[max_step - 1, p]
        #x_avg[None] +=  (1 / obj.n) * x_this[max_step - 1, p]

# simulation
@ti.kernel
def substep(obj: ti.template()):
    for p in range(obj.n):
        obj.x[max_step - 1, p] += v[None]
        #x_this[max_step - 1, p] += v[None]

def forward(obj):
    substep(obj)


def main():
    
    obj = Obj()
    ti.root.place(x_avg, loss, v)
    ti.root.lazy_grad()
    with ti.ad.Tape(loss=loss):
        forward(obj)
        compute_x_avg(obj)
        compute_loss(obj)

    print(loss[None])
    print(v.grad)

if __name__ == "__main__":
    main()

























