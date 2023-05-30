#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:57:24 2023

@author: luyin
"""

import numpy as np
import matplotlib.pyplot as plt

loss = np.load('./data/loss.npy')
v_input = np.load('./data/v_input.npy')


plt.plot(loss)
plt.show()