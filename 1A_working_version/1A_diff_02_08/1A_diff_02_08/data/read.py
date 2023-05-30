#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 20:02:37 2023

@author: luyin
"""

import numpy as np
import matplotlib.pyplot as plt

#ini_v = np.load('v_input_np.npy')
loss3 = np.load('loss_3.npy') * 10e3
loss4 = np.load('loss_4.npy') * 10e3
loss5 = np.load('loss_5.npy') * 10e3
loss6 = np.load('loss_6.npy') * 10e3
loss7 = np.load('loss_7.npy') * 10e3
loss8 = np.load('loss_8.npy') * 10e3
loss9 = np.load('loss_9.npy') * 10e3

loss_all = np.concatenate((loss3,loss4,loss5,loss6,loss7,loss8,loss9))
#grad = np.load('grad.npy') * 500
#x_avg = np.load('x_avg.npy') 

#plt.figure(figsize=(5,3))

plt.plot(loss_all,'b-')
plt.ylabel('loss',fontsize=14)
plt.xlabel('iteration',fontsize=14)
plt.grid(linestyle ='dashed')
#plt.show()
#plt.xlim([-20, 720])

plt.savefig('loss.jpg', dpi=400)
#plt.ylim([-20, 3100])