#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:57:24 2023

@author: luyin
"""

import numpy as np
import matplotlib.pyplot as plt

loss_05 = np.load('./k0.5/loss.npy')[0:30]
loss_1 = np.load('./k1/loss.npy')
loss_4 = np.load('./k4/loss.npy')
loss_8 = np.load('./k8/loss.npy')
# v_input = np.load('./k1/v_input.npy')[0:49,:,2]
# delta_v = []

# for i in range(11):
#     delta_v.append(v_input[i+1] - v_input[i])

#delta_v_np = np.array(delta_v)
plt.plot(loss_05, label = 'k = 0.5')
plt.plot(loss_1, label = 'k = 1')
plt.plot(loss_4, label = 'k = 4')
plt.plot(loss_8, label = 'k = 8')
plt.legend()
plt.show()