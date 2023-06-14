#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:57:24 2023

@author: luyin
"""

import numpy as np
import matplotlib.pyplot as plt





trial_num = 13
loss = np.load(f'../trials/trial_{trial_num}/loss.npy')
v_input = np.load(f'../trials/trial_{trial_num}/v_input.npy')
# obj_center = np.load(f'../trial_{trial_num}/obj_center.npy')
# target_num = np.load(f'../trial_{trial_num}/target_num.npy')


# for i in range(1,iter_num,1):
#     fig, ax = plt.subplots(dpi = 400)
#     pio_coe = pio_coe_all[i].reshape(14,14)
#     im = ax.imshow(pio_coe,vmax=max_this,vmin=min_this)
#     fig.colorbar(im, orientation='vertical')
#     fig.savefig(f'./heatmap/{i}.jpg')
#     print(i)
#     #plt.clf()

