#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 11:14:45 2022

@author: luyin
"""

import numpy as np

D_inv = np.load('D_inv.npy')
F_quad = np.load('F_quad.npy')
x = np.load('x.npy')
x_anchor = np.load('x_anchor.npy')

print(F_quad[193])

# 17 193 223