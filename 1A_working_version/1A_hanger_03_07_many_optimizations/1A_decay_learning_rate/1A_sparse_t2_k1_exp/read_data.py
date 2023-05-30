#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:57:24 2023

@author: luyin
"""

import numpy as np
import matplotlib.pyplot as plt

loss = np.load('./data/loss.npy')
v_input = np.load('./data/v_input.npy')[0:84,0,:]
delta_v = []

for i in range(11):
    delta_v.append(v_input[i+1] - v_input[i])

delta_v_np = np.array(delta_v)
plt.plot(v_input,'bo',markersize = 2)
plt.plot(v_input)
plt.show()