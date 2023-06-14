#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 23:24:26 2023

@author: luyin
"""

import numpy as np

A = np.array([0])
np.save('./trial_num.npy',A)

B = np.load('./trial_num.npy')
# B[0] += 1
# np.save(f'./trial_num_{B[0]}.npy',B)
