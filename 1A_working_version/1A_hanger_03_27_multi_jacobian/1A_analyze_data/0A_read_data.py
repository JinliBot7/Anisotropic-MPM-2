#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:57:24 2023

@author: luyin
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'


fig, ax = plt.subplots(dpi = 400)

n_iter = 30 # number of the iteration
n_step = 0 # number of step within a iteration
n_point = 0 # number of the point
dim_p = 0 # dimension of the point position
dim_v = 0 # dimension of the velocity

# iter 0, dim_p 0, dim_v 0, all point variation along step
#for n_iter in range(1):
n_iter = 0
iter_list = []


for n_step in range(100):
    print(n_iter,n_step)
    step_list = []
    heatmap = np.ones((8,8))
    for n_point in range(64):
        grad = np.load(f'./data/grad_{n_iter}_{n_point}_{dim_p}.npy')[n_step,dim_v]
        step_list.append(grad)
        colum = n_point // 8
        row = n_point - colum * 8
        heatmap[colum,row] = grad
    fig = px.imshow(heatmap, text_auto=True)
    fig.show()
    iter_list.append(step_list)
iter_list_np = np.array(iter_list)
#np.save(f'./data/xx_{n_iter}.npy',iter_list_np)



# loss = np.load(f'./{k}/loss.npy')
# v_input = np.load(f'./{k}/v_input.npy')[:,:,0]
# # grad = np.load(f'./{k}/grad.npy')[:,99,0]
# #ax.plot(loss, linestyle="-", label='learning decay = 0.00')
# ax.plot(v_input, linestyle="-")

# # k = 1
# # loss = np.load(f'./{k}/loss.npy')
# # v_input = np.load(f'./{k}/v_input.npy')[:,99,0]
# # grad = np.load(f'./{k}/grad.npy')[:,99,0]
# # #ax.plot(loss, linestyle="-", label='learning decay = 0.01')
# # #ax.plot(v_input, linestyle="-")


# ax.set_xlabel('iteration')
# ax.set_ylabel('loss')
# ax.legend()
# ax.grid(color='dimgray', linestyle='--', linewidth=0.5)