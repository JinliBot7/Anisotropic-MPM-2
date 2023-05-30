#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:57:24 2023

@author: luyin
"""

import numpy as np
import matplotlib.pyplot as plt





k = 'data'
loss = np.load(f'./{k}/loss.npy')
loss_position = np.load(f'./{k}/loss_position.npy')
loss_velocity = np.load(f'./{k}/loss_velocity.npy')
v_input = np.load(f'./{k}/v_input.npy')[:,:,1].transpose()
minimum_index = np.load(f'./{k}/minimum_index_np.npy')
grad = np.load(f'./{k}/grad.npy')[:,:,0]

pio_coe_all = np.load(f'./{k}/pio_coe.npy') 
pio_dist = np.load(f'./{k}/pio_dist.npy')
pio_dist_hat = np.load(f'./{k}/pio_dist_hat.npy')
i = 0
dist_hat = pio_dist_hat[:,i]
dist = pio_dist[:,i]
fig, ax = plt.subplots(dpi = 400)
ax.plot(loss)
ax.plot(loss_position)
print(np.amin(loss_position))
#ax.plot(dist_hat)



# max_this = 0.0
# min_this = 1.0

# iter_num = 250
# for i in range(1,iter_num,1):
#     pio_coe = pio_coe_all[i]
#     max_this = max(np.amax(pio_coe),max_this)
#     min_this = min(np.amin(pio_coe),min_this)
#     print(i)

# for i in range(1,iter_num,1):
#     fig, ax = plt.subplots(dpi = 400)
#     pio_coe = pio_coe_all[i].reshape(14,14)
#     im = ax.imshow(pio_coe,vmax=max_this,vmin=min_this)
#     fig.colorbar(im, orientation='vertical')
#     fig.savefig(f'./heatmap/{i}.jpg')
#     print(i)
#     #plt.clf()

