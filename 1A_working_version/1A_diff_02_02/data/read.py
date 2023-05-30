#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 20:02:37 2023

@author: luyin
"""

import numpy as np
import matplotlib.pyplot as plt

ini_v = np.load('ini_v.npy')
loss = np.load('loss.npy')[0:35]
grad = np.load('grad.npy')
x_avg = np.load('x_avg.npy')


plt.plot(loss)
plt.ylabel('loss')
plt.show()
