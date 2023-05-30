#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 14:22:07 2023

@author: luyin
"""

import taichi as ti
ti.init()

x = ti.field(ti.f32)
total = ti.field(ti.f32)
n = 128
ti.root.dense(ti.i, n).place(x)
ti.root.place(total)
ti.root.lazy_grad()

@ti.kernel
def func(mul: ti.f32):
    for i in range(n):
        ti.atomic_add(total[None], x[i] * mul)

@ti.ad.grad_replaced
def forward(mul):
    func(mul)
    func(mul)

@ti.ad.grad_for(forward)
def backward(mul):
    func.grad(mul)

with ti.ad.Tape(loss=total):
    forward(4)

assert x.grad[0] == 4