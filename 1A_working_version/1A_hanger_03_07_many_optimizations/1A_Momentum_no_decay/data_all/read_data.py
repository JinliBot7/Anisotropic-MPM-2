#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:57:24 2023

@author: luyin
"""

import numpy as np
import matplotlib.pyplot as plt

loss_05 = np.load('./k05/loss.npy')
v_input_05 = np.load('./k05/v_input.npy')[:,:,0]
grad_05 = np.load('./k05/grad.npy')[:,:,0]

loss_1 = np.load('./k1/loss.npy')
v_input_1 = np.load('./k1/v_input.npy')[:,:,0]
grad_1 = np.load('./k1/grad.npy')[:,:,0]

loss_2 = np.load('./k2/loss.npy')
v_input_2 = np.load('./k2/v_input.npy')[:,:,0]
grad_2 = np.load('./k2/grad.npy')[:,:,0]


loss_4 = np.load('./k4/loss.npy')
v_input_4 = np.load('./k4/v_input.npy')[:,:,0]
grad_4 = np.load('./k4/grad.npy')[:,:,0]

# loss_01 = np.load('./k01/loss.npy')
# v_input_01 = np.load('./k01/v_input.npy')[:,:,0]
# grad_01 = np.load('./k01/grad.npy')[:,:,0]



# delta_v = []
# for i in range(29):
#     delta_v.append(v_input[i+1] - v_input[i])
# delta_v_np = np.array(delta_v)

plt.figure(dpi = 400)

#plt.plot(grad_01)
plt.plot(loss_05, label = 'k = 0.5')
plt.plot(loss_1, label = 'k = 1')
plt.plot(loss_2, label = 'k = 2')
plt.plot(loss_4, label = 'k = 4')

# plt.plot(loss_01, label = 'k = 0.1')
plt.legend()
plt.show()