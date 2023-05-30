#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:57:24 2023

@author: luyin
"""

import numpy as np
import matplotlib.pyplot as plt



loss_01 = np.load('./t2k01/loss.npy')
v_input_01 = np.load('./t2k01/v_input.npy')[:,:,0]
grad_01 = np.load('./t2k01/grad.npy')[:,:,0]

loss_005 = np.load('./t2k005/loss.npy')
v_input_005 = np.load('./t2k005/v_input.npy')[:,:,0]
grad_005 = np.load('./t2k005/grad.npy')[:,:,0]

loss_0025 = np.load('./t2k0025/loss.npy')
v_input_0025 = np.load('./t2k0025/v_input.npy')[:,:,0]
grad_0025 = np.load('./t2k0025/grad.npy')[:,:,0]

# with good initialization
loss_gi01 = np.load('./git2k01/loss.npy')
v_input_gi01 = np.load('./git2k01/v_input.npy')[0,:,0]
grad_gi01 = np.load('./git2k01/grad.npy')[:,:,0]




# delta_v = []
# for i in range(29):
#     delta_v.append(v_input[i+1] - v_input[i])
# delta_v_np = np.array(delta_v)

plt.figure(dpi = 400)


# plt.plot(loss_01, label = 'k = 0.1')
# plt.plot(loss_005, label = 'k = 0.05')
# plt.plot(loss_0025, label = 'k = 0.025')
# plt.plot(loss_gi01, label = 'k = 0.1, tuned initialization')


plt.plot(grad_gi01)

# plt.plot(loss_01, label = 'k = 0.1')
plt.legend()
plt.show()