#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 22:17:02 2023

@author: luyin
"""

import numpy as np
import math
import matplotlib.pyplot as plt

center_list =[]
rot_list = []

for i in range(10000):
    center_z = np.load(f'target_path/center_{i}.npy')[2]
    center_list.append(center_z)
    rot = np.load(f'target_path/rot_{i}.npy')[0] / math.pi * 180
    rot_list.append(rot)

#%%

center_z_np = np.array(center_list)
rot_np = np.array(rot_list)

fig, axs = plt.subplots(1, 1, tight_layout=True)

plt.title('Intermediate state sampling')
plt.xlabel('height')
plt.ylabel('rotation')
plt.xlim(0.34,0.46)
plt.ylim(-190, -90)
axs.scatter(center_z_np,rot_np,s=1)
