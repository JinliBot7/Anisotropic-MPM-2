#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 19:27:14 2023

@author: luyin
"""
import numpy as np
A = np.ones((100,4))

A[0,0] = 0
A[1,2] = -1
A[2,1] = -3
A[3,3] = -2

G = np.load('pio_real_dist_np.npy')

#A[4,:] = 2

B = np.amin(G, axis = 0)

C = np.where(G  == B)

E = C[0]

D = C[1]