#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 00:03:49 2023

@author: luyin
"""
import numpy as np
grid_m = np.load('./target_path/target_m_0.npy')

for i in range(128):
    print(i)
    for j in range(128):
        for k in range(128):
            if grid_m[i,j,k] != 0.0:
                print('yes!',i,j,k,grid_m[i,j,k])