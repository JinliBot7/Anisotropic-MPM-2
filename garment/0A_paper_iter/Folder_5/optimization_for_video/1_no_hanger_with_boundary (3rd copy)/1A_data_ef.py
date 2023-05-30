#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:57:24 2023

@author: luyin
"""

import numpy as np
import matplotlib.pyplot as plt




ef_left = np.load('data/ef_left.npy')


ax1 = plt.subplot(311)
ax1.plot(ef_left[:,0] * 100)

ax2 = plt.subplot(312)
ax2.plot(ef_left[:,1] * 100)

ax2 = plt.subplot(313)
ax2.plot(ef_left[:,2] * 100)



