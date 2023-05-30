#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:57:24 2023

@author: luyin
"""

import numpy as np
import matplotlib.pyplot as plt




ef_left = np.load('data/ef_left.npy')

num = 20000

ef = np.zeros((num,3))
ef_v = np.zeros((num,3))
for i in range(num - 1):
    ef_v[i] = ((ef_left[(i+1) * int(20000/num)] - ef_left[(i) * int(20000/num)]) * num) * 1000
    ef[i] = ef_left[i*int(20000/num)] * 1000 - ef_left[0] * 1000

ef[num - 1] = ef[num - 2] 
ax1 = plt.subplot(311)
ax1.plot(ef[:,0])
ax2 = plt.subplot(312)
ax2.plot(ef[:,1])
ax3= plt.subplot(313)
ax3.plot(ef[:,2])
# ax4 = plt.subplot(614)
# ax4.plot(ef_v[:,0])
# ax5 = plt.subplot(615)
# ax5.plot(ef_v[:,1])
# ax6= plt.subplot(616)
# ax6.plot(ef_v[:,2])

plt.savefig('0A.png',dpi = 300)

np.savetxt(f'ef_{num}.txt', ef)
np.savetxt(f'ef_v_{num}.txt', ef_v)
np.save('ef.npy',ef)



