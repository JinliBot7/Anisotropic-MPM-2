#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 20:02:37 2023

@author: luyin
"""

import numpy as np
import matplotlib.pyplot as plt

ini_v = np.load('v_input_np.npy')
loss = np.load('loss.npy') * 1e3
grad = np.load('grad.npy') * 500
x_avg = np.load('x_avg.npy') 


plt.plot(loss,'b-')
plt.ylabel('loss')
plt.show()
