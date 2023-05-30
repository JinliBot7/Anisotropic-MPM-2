#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:57:24 2023

@author: luyin
"""

import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots(dpi = 400)


k = 0
loss = np.load(f'./{k}/loss.npy')[1:]
v_input = np.load(f'./{k}/v_input.npy')[:,:,2]
grad = np.load(f'./{k}/grad.npy')[:,0,0]
ax.plot(loss, linestyle="-", label='beta1 = 0.9')

k = 1
loss = np.load(f'./{k}/loss.npy')[1:]   
v_input = np.load(f'./{k}/v_input.npy')
grad = np.load(f'./{k}/grad.npy')[:,0,0]
ax.plot(loss, linestyle="-", label='beta1 = 0.5')

k = 2
loss = np.load(f'./{k}/loss.npy')[1:]
v_input = np.load(f'./{k}/v_input.npy')
grad = np.load(f'./{k}/grad.npy')[:,0,0]
ax.plot(loss, linestyle="-", label='beta1 = 0.1')

k = 3
loss = np.load(f'./{k}/loss.npy')[1:]
v_input = np.load(f'./{k}/v_input.npy')
grad = np.load(f'./{k}/grad.npy')[:,0,0]
ax.plot(loss, linestyle="-", label='beta1 = 0.7')


ax.set_xlabel('iteration')
ax.set_ylabel('loss')
ax.legend()
ax.grid(color='dimgray', linestyle='--', linewidth=0.5)

# plt.xlim([0,50])
# plt.ylim([0,8])
# plt.grid(color='dimgray', linestyle='--', linewidth=0.5)

# plt.show()