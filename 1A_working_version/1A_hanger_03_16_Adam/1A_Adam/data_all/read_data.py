#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:57:24 2023

@author: luyin
"""

import numpy as np
import matplotlib.pyplot as plt



loss_1 = np.load('./vis_exp_10/loss_standard.npy') / 64 * 100
v_input_1 = np.load('./vis_exp_10/v_input.npy')[1,:,0]
grad_1 = np.load('./vis_exp_10/grad.npy')[0,:,0]

loss_2 = np.load('./vis_exp_20/loss_standard.npy')
v_input_2 = np.load('./vis_exp_20/v_input.npy')[:,:,0]
grad_2 = np.load('./vis_exp_20/grad.npy')[:,:,0]



delta_v = []
for i in range(99):
    delta_v.append(v_input_1[i+1] - v_input_1[i])
delta_v_np = np.array(delta_v)

plt.figure(dpi = 400)

# plt.plot(grad_1)



plt.plot(loss_1, label = "exp = 10") 
#plt.plot(loss_2, label = "exp = 20") 

# plt.plot(loss_01, label = 'k = 0.1')
plt.legend()
plt.show()