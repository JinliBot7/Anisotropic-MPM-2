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
v_input = np.load(f'./{k}/v_input.npy')
grad = np.load(f'./{k}/grad.npy')[:,:,0]

pio_coe_all = np.load(f'./{k}/pio_coe.npy') 
pio_dist = np.load(f'./{k}/pio_dist.npy')
pio_dist_hat = np.load(f'./{k}/pio_dist_hat.npy')

v_cor = 0

v_input_cor =  v_input[:,:,v_cor]

max_this = 0.0
min_this = 1.0

iter_num = 200
for i in range(0,iter_num,1):
    v_input_this = v_input_cor[i]
    max_this = max(np.amax(v_input_this),max_this)
    min_this = min(np.amin(v_input_this),min_this)


# v_input_failed = np.load(f'./{k}/v_input_failed.npy')
# v_input_cor_failed = v_input_failed[:,:,v_cor]
# for i in range(0,iter_num,1):
#     v_input_this = v_input_cor_failed[i]
#     max_this = max(np.amax(v_input_this),max_this)
#     min_this = min(np.amin(v_input_this),min_this)


for i in range(0,iter_num,1):
    fig, ax = plt.subplots(dpi = 400)
    ax.plot(v_input_cor[i],'k-',label = 'success')
    #ax.plot(v_input_cor_failed[i],'r-', label = 'failed')
    plt.title(f'iteration {i}')
    plt.xlabel('velocity interval number')
    plt.ylabel(r'$v_x$')
    plt.xlim(-0.5,10.5)
    plt.ylim(min_this-0.02, max_this+0.02)
    plt.xticks(np.arange(-1, 11, 1))
    plt.legend()
    fig.savefig(f'./vis_v_{v_cor}/{i}.jpg')
    print(v_cor,i)
    plt.clf()

v_cor = 1

v_input_cor =  v_input[:,:,v_cor]

max_this = 0.0
min_this = 1.0

iter_num = 200
for i in range(0,iter_num,1):
    v_input_this = v_input_cor[i]
    max_this = max(np.amax(v_input_this),max_this)
    min_this = min(np.amin(v_input_this),min_this)


# v_input_failed = np.load(f'./{k}/v_input_failed.npy')
# v_input_cor_failed = v_input_failed[:,:,v_cor]
# for i in range(0,iter_num,1):
#     v_input_this = v_input_cor_failed[i]
#     max_this = max(np.amax(v_input_this),max_this)
#     min_this = min(np.amin(v_input_this),min_this)


for i in range(0,iter_num,1):
    fig, ax = plt.subplots(dpi = 400)
    ax.plot(v_input_cor[i],'k-',label = 'success')
    #ax.plot(v_input_cor_failed[i],'r-', label = 'failed')
    plt.title(f'iteration {i}')
    plt.xlabel('velocity interval number')
    plt.ylabel(r'$v_y$')
    plt.xlim(-0.5,10.5)
    plt.ylim(min_this-0.02, max_this+0.02)
    plt.xticks(np.arange(-1, 11, 1))
    plt.legend()
    fig.savefig(f'./vis_v_{v_cor}/{i}.jpg')
    print(v_cor,i)
    plt.clf()

v_cor = 2

v_input_cor =  v_input[:,:,v_cor]

max_this = 0.0
min_this = 1.0

iter_num = 200
for i in range(0,iter_num,1):
    v_input_this = v_input_cor[i]
    max_this = max(np.amax(v_input_this),max_this)
    min_this = min(np.amin(v_input_this),min_this)


# v_input_failed = np.load(f'./{k}/v_input_failed.npy')
# v_input_cor_failed = v_input_failed[:,:,v_cor]
# for i in range(0,iter_num,1):
#     v_input_this = v_input_cor_failed[i]
#     max_this = max(np.amax(v_input_this),max_this)
#     min_this = min(np.amin(v_input_this),min_this)


for i in range(0,iter_num,1):
    fig, ax = plt.subplots(dpi = 400)
    ax.plot(v_input_cor[i],'k-',label = 'success')
    #ax.plot(v_input_cor_failed[i],'r-', label = 'failed')
    plt.title(f'iteration {i}')
    plt.xlabel('velocity interval number')
    plt.ylabel(r'$v_z$')
    plt.xlim(-0.5,10.5)
    plt.ylim(min_this-0.02, max_this+0.02)
    plt.xticks(np.arange(-1, 11, 1))
    plt.legend()
    fig.savefig(f'./vis_v_{v_cor}/{i}.jpg')
    print(v_cor,i)
    plt.clf()

