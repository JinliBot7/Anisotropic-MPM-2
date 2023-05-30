#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:57:24 2023

@author: luyin
"""

import numpy as np
import matplotlib.pyplot as plt



fig, ax = plt.subplots(dpi = 400)

for i in range(8):
    loss_position = np.load(f'./data_fixed_ini_zero_v_{i}/loss_position.npy')[1:200]
    ax.plot(loss_position)
    print(i,np.amin(loss_position),np.where(loss_position == np.amin(loss_position))[0])
    #print(np.where(loss_position == np.amin(loss_position)))

loss_position = np.load(f'./data_fixed_ini_zero_v_6/loss_position.npy')[1:200]
# loss_position = np.load(f'./{k}/loss_position.npy')
# loss_velocity = np.load(f'./{k}/loss_velocity.npy')
# v_input = np.load(f'./{k}/v_input.npy')[:,:,2].transpose()
# minimum_index = np.load(f'./{k}/minimum_index_np.npy')
# grad = np.load(f'./{k}/grad.npy')[:,:,0]

# pio_coe_all = np.load(f'./{k}/pio_coe.npy') 
# pio_dist = np.load(f'./{k}/pio_dist.npy')
# pio_dist_hat = np.load(f'./{k}/pio_dist_hat.npy')




