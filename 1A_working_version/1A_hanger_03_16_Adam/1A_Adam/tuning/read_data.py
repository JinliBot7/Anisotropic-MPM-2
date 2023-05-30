#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:57:24 2023

@author: luyin
"""

import numpy as np
import matplotlib.pyplot as plt

parameter = np.load('./data/parameter.npy',)

# loss_1 = np.load('./vis_exp_10/loss_standard.npy') / 64 * 100
# v_input_1 = np.load('./vis_exp_10/v_input.npy')[1,:,0]
# grad_1 = np.load('./vis_exp_10/grad.npy')[0,:,0]



plt.figure(dpi = 400,figsize=(5,3))

# loss_list = []
# for i in range(parameter.shape[0]):
#     loss = np.load(f'./data/loss_standard_{i}.npy')
#     loss_list.append(np.sum(loss))
#     #plt.plot(loss, label = str(parameter[i]))
#     plt.plot(loss)



# loss_list_np = np.array(loss_list)
# print(np.min(loss_list_np))
# print(np.where(loss_list_np == np.min(loss_list_np)))
# print(parameter[188])

for i in range(300):
    loss = np.load(f'./data/loss_{i}.npy')
    v_input = np.load(f'./data/v_input_{i}.npy')[1,:,0]
    grad = np.load(f'./data/grad_{i}.npy')[1,:,0]
#plt.plot(loss, 'ko')
    plt.plot(loss)
plt.grid(color='dimgray', linestyle='--', linewidth=0.5)
#plt.legend()
plt.xlabel('iteration')
plt.ylabel('loss (m)')
plt.title('Feasible Parameter Search')
plt.show()
