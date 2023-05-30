#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:57:24 2023

@author: luyin
"""

import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots(dpi = 400)


# k = 0
# loss = np.load(f'./{k}/loss.npy')
# v_input = np.load(f'./{k}/v_input.npy')[:,:,2]
# grad = np.load(f'./{k}/grad.npy')[:,99,0]
# ax.plot(loss, linestyle="-", label='learning decay = 0.00')
# #ax.plot(v_input, linestyle="-")

k = 1
loss = np.load(f'./{k}/loss.npy')
v_input = np.load(f'./{k}/v_input.npy')[:,:,0]
grad = np.load(f'./{k}/grad.npy')[:,99,0]
#ax.plot(loss, linestyle="-", label='learning decay = 0.01')
ax.plot(v_input, linestyle="-")

# k = 2
# loss = np.load(f'./{k}/loss_position.npy')
# v_input = np.load(f'./{k}/v_input.npy')[:,:,2]
# grad = np.load(f'./{k}/grad.npy')[:,99,0]
# ax.plot(loss, linestyle="-", label='base = 4')
# #ax.plot(v_input, linestyle="-")

# k = 3
# loss = np.load(f'./{k}/loss_position.npy')
# v_input = np.load(f'./{k}/v_input.npy')[:,:,2]
# grad = np.load(f'./{k}/grad.npy')[:,99,0]
# ax.plot(loss, linestyle="-", label='base = 3')
# #ax.plot(v_input, linestyle="-")

# k = 4
# loss = np.load(f'./{k}/loss_position.npy')
# v_input = np.load(f'./{k}/v_input.npy')[:,:,2]
# grad = np.load(f'./{k}/grad.npy')[:,99,0]
# ax.plot(loss, linestyle="-", label='base = 2')
# #ax.plot(v_input, linestyle="-")

# k = 5
# loss = np.load(f'./{k}/loss_position.npy')
# v_input = np.load(f'./{k}/v_input.npy')[:,:,2]
# grad = np.load(f'./{k}/grad.npy')[:,99,0]
# ax.plot(loss, linestyle="-", label='base = 1.5')
# #ax.plot(v_input, linestyle="-")


# k = 6
# loss = np.load(f'./{k}/loss_position.npy')
# v_input = np.load(f'./{k}/v_input.npy')[:,:,0]
# grad = np.load(f'./{k}/grad.npy')[:,99,0]
# #ax.plot(loss, linestyle="-", label='base = 1')
# ax.plot(v_input, linestyle="-")

plt.title('loss-iteration variation using different hyperparameters')
#plt.ylim([-0.05,0.1])
ax.set_xlabel('iteration')
ax.set_ylabel('loss')
ax.legend()
ax.grid(color='dimgray', linestyle='--', linewidth=0.5)

# plt.xlim([0,50])
#plt.ylim([1,2])
# plt.grid(color='dimgray', linestyle='--', linewidth=0.5)

# plt.show()