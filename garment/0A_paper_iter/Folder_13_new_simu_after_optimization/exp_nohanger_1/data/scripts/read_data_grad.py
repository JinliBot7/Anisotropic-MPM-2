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
grad = np.load(f'./{k}/grad.npy')


v_cor = 0
grad_cor =  grad[:,:,v_cor]

max_this = 0.0
min_this = 1.0

iter_num = 200
for i in range(0,iter_num,1):
    grad_this = grad_cor[i]
    if np.isnan(np.amax(grad_this)) or np.isnan(np.amin(grad_this)):
        pass
    else:
        max_this = max(np.amax(grad_this),max_this)
        min_this = min(np.amin(grad_this),min_this)

# grad_failed = np.load(f'./{k}/grad_failed.npy')
# grad_cor_failed = grad_failed[:,:,v_cor]
# for i in range(0,iter_num,1):
#     grad_this = grad_cor_failed[i]
#     if np.isnan(np.amax(grad_this)) or np.isnan(np.amin(grad_this)):
#         pass
#     else:
#         max_this = max(np.amax(grad_this),max_this)
#         min_this = min(np.amin(grad_this),min_this)


for i in range(0,iter_num,1):
    fig, ax = plt.subplots(dpi = 400)
    ax.plot(grad_cor[i],'k-',label = 'success')
    #ax.plot(grad_cor_failed[i],'r-', label = 'failed')
    plt.title(f'iteration {i}')
    plt.xlabel('velocity interval number')
    plt.ylabel(r'$grad_x$')
    plt.xlim(-0.5,10.5)
    plt.ylim(min_this-0.02, max_this+0.02)
    plt.xticks(np.arange(-1, 11, 1))
    plt.legend()
    fig.savefig(f'./vis_grad_{v_cor}/{i}.jpg')
    print(v_cor,i)
    plt.clf()

v_cor = 1

grad_cor =  grad[:,:,v_cor]

max_this = 0.0
min_this = 1.0

iter_num = 200
iter_num = 200
for i in range(0,iter_num,1):
    grad_this = grad_cor[i]
    if np.isnan(np.amax(grad_this)) or np.isnan(np.amin(grad_this)):
        pass
    else:
        max_this = max(np.amax(grad_this),max_this)
        min_this = min(np.amin(grad_this),min_this)

# grad_failed = np.load(f'./{k}/grad_failed.npy')
# grad_cor_failed = grad_failed[:,:,v_cor]
# for i in range(0,iter_num,1):
#     grad_this = grad_cor_failed[i]
#     if np.isnan(np.amax(grad_this)) or np.isnan(np.amin(grad_this)):
#         pass
#     else:
#         max_this = max(np.amax(grad_this),max_this)
#         min_this = min(np.amin(grad_this),min_this)


for i in range(0,iter_num,1):
    fig, ax = plt.subplots(dpi = 400)
    ax.plot(grad_cor[i],'k-',label = 'success')
    #ax.plot(grad_cor_failed[i],'r-', label = 'failed')
    plt.title(f'iteration {i}')
    plt.xlabel('velocity interval number')
    plt.ylabel(r'$grad_y$')
    plt.xlim(-0.5,10.5)
    plt.ylim(min_this-0.02, max_this+0.02)
    plt.xticks(np.arange(-1, 11, 1))
    plt.legend()
    fig.savefig(f'./vis_grad_{v_cor}/{i}.jpg')
    print(v_cor,i)
    plt.clf()

v_cor = 2

grad_cor =  grad[:,:,v_cor]

max_this = 0.0
min_this = 1.0

iter_num = 200
for i in range(0,iter_num,1):
    grad_this = grad_cor[i]
    if np.isnan(np.amax(grad_this)) or np.isnan(np.amin(grad_this)):
        pass
    else:
        max_this = max(np.amax(grad_this),max_this)
        min_this = min(np.amin(grad_this),min_this)

# grad_failed = np.load(f'./{k}/grad_failed.npy')
# grad_cor_failed = grad_failed[:,:,v_cor]
# for i in range(0,iter_num,1):
#     grad_this = grad_cor_failed[i]
#     if np.isnan(np.amax(grad_this)) or np.isnan(np.amin(grad_this)):
#         pass
#     else:
#         max_this = max(np.amax(grad_this),max_this)
#         min_this = min(np.amin(grad_this),min_this)


for i in range(0,iter_num,1):
    fig, ax = plt.subplots(dpi = 400)
    ax.plot(grad_cor[i],'k-',label = 'success')
    #ax.plot(grad_cor_failed[i],'r-', label = 'failed')
    plt.title(f'iteration {i}')
    plt.xlabel('velocity interval number')
    plt.ylabel(r'$grad_z$')
    plt.xlim(-0.5,10.5)
    plt.ylim(min_this-0.02, max_this+0.02)
    plt.xticks(np.arange(-1, 11, 1))
    plt.legend()
    fig.savefig(f'./vis_grad_{v_cor}/{i}.jpg')
    print(v_cor,i)
    plt.clf()

