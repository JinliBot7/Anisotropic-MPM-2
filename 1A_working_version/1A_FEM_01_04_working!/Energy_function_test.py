#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 21:20:18 2022

@author: luyin
"""

import taichi as ti
#from compute_stress import compute_stress
from compute_stress_113 import compute_stress

ti.init(ti.cpu)

F = ti.Matrix([[1.1, 0.0, 0.0],[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
#F = ti.Matrix([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0], [0.0, 0.0, 0.9]])

E, eta = 10e2, 0.1
lam = E * eta / ((1 + eta) * (1 - 2 * eta)) 
mu = E / (2 * (1 + eta))

dim = 3
neighbour = (3,) * dim
dx = 1 / 256
inv_dx = 256
vol = 1

@ti.kernel
def test():
    
    F_p = F
    #U, sig, V = ti.svd(F_p)
    J = F_p.determinant()
    F_T = F_p.transpose()
    F_inv_T = F_T.inverse()
    P = mu * (F_p - F_inv_T) + lam * ti.log(J) * F_inv_T
    print('P: ', P)
    
    #F_plastic = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    P_new, F_elastic, F_plastic = compute_stress(F,mu,lam)
    print('P_new: ', P_new)
    
    dpos = ti.Vector([0.0, -1.0, 0.0])
    
    force = - P @ F_T @ dpos
    force_new = - P_new @ F_T @ dpos
    print('force',force)
    print('force_new',force_new)
    
    # x = ti.Vector([0.51, 0.52, 0.53])
    # base = (x * inv_dx - 0.5).cast(int)
    # fx = x * inv_dx - base.cast(float)
    # print('base: ', base)
    # print('fx:', fx)
    # #fx = ti.Vector([1.416000, 0.584000, 1.223591])
    # #fx = ti.Vector([1.416000, 0.584000, 1.223594])
    # w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2,]  # quadratic b spline
    # for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
    #     dpos = (offset - fx) * dx
    #     weight = 1.0
    #     for i in ti.static(range(dim)):
    #         weight *= w[offset[i]][i]
        

            


test()
