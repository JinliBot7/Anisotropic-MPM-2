#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:57:24 2023

@author: luyin
"""

import numpy as np
import matplotlib.pyplot as plt

parameter = np.load('./data/parameter.npy')

# loss_1 = np.load('./vis_exp_10/loss_standard.npy') / 64 * 100
# v_input_1 = np.load('./vis_exp_10/v_input.npy')[1,:,0]
# grad_1 = np.load('./vis_exp_10/grad.npy')[0,:,0]



plt.figure(dpi = 400)

loss_list = []
for i in range(parameter.shape[0]):
    loss = np.load(f'./data/loss_standard_{i}.npy')
    loss_list.append(np.sum(loss))
    #plt.plot(loss, label = str(parameter[i]))
    plt.plot(loss)

print(np.min(np.array(loss_list)))
print(parameter[10])
plt.legend()
plt.show()