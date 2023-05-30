#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:57:24 2023

@author: luyin
"""

import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots(dpi = 400)


k = 'target_path'
energy_list = []
for counter in range(100,100000, 100):
    
    energy = np.load(f'./{k}/deformation_energy_{counter}.npy')
    #print(energy)
    energy_list.append(energy)

energy_np = np.array(energy_list)
ax.plot(energy_np, linestyle="-", label='learning decay = 0.00')
#ax.plot(v_input, linestyle="-")




ax.set_xlabel('iteration')
ax.set_ylabel('loss')
ax.legend()
ax.grid(color='dimgray', linestyle='--', linewidth=0.5)