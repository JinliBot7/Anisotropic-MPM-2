#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:57:24 2023

@author: luyin
"""

import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots(dpi = 400)


k = 'data'
loss = np.load(f'./{k}/loss.npy')
loss_position = np.load(f'./{k}/loss_position.npy')
loss_velocity = np.load(f'./{k}/loss_velocity.npy')
v_input = np.load(f'./{k}/v_input.npy')[1,:,0].transpose()
minimum_index = np.load(f'./{k}/minimum_index_np.npy')
# grad = np.load(f'./{k}/grad.npy')[:,99,0]
#ax.plot(loss, linestyle="-", label='learning decay = 0.00')
ax.plot(v_input, linestyle="-")

# k = 1
# loss = np.load(f'./{k}/loss.npy')
# v_input = np.load(f'./{k}/v_input.npy')[:,99,0]
# grad = np.load(f'./{k}/grad.npy')[:,99,0]
# #ax.plot(loss, linestyle="-", label='learning decay = 0.01')
# #ax.plot(v_input, linestyle="-")


# ax.set_xlabel('iteration')
# ax.set_ylabel('loss')
# ax.legend()
# ax.grid(color='dimgray', linestyle='--', linewidth=0.5)