#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 14:24:00 2023

@author: luyin
"""


import numpy as np
import cv2 as cv
import plotly.graph_objects as go
import plotly.io as pio
import colorsys
from numpy import pi, cos, sin
pio.renderers.default='browser'

# aruco_0 exp_num 0: 32 exp_num 1: 36
exp_name = 'data_0612'
exp_num = 3
n_frame_start = 32
n_frame_end = 100


T_rc = np.load('T_cam2base.npy')

x_ef = 0.0
y_ef = 13.15
z_ef = -91.336

x_c = 10.759 + 1.5
y_c = 2.0   
z_c = -11.493

# theta_x = pi / 2
# rot_x = np.array([
#     [1.0, 0.0, 0.0],
#     [0.0, cos(theta_x), -sin(theta_x)],
#     [0.0, sin(theta_x), cos(theta_x)]
#     ])
theta_z = pi
rot_z = np.array([
    [cos(theta_z), -sin(theta_z), 0.0],
    [sin(theta_z), cos(theta_z), 0.0],
    [0.0, 0.0, 1.0]
    ])

for n_frame in range(n_frame_start,n_frame_end + 1,1):
#for n_frame in range(44,45,1):
    vert = np.load(f'../../../0AAA_all_exp_data/{exp_name}/exp_{exp_num}/vert/vert_{n_frame}.npy') * 100
    uv = np.load(f'../../../0AAA_all_exp_data/{exp_name}/exp_{exp_num}/uv/uv_{n_frame}.npy')
    bgr = cv.imread(f'../../../0AAA_all_exp_data/{exp_name}/exp_{exp_num}/rgb/rgb_img_{n_frame}.jpg')
    #time = np.load(f'data/{exp_name}/{n_exp}/time.npy')

    roi_flag = np.zeros(vert.shape[0])
    
    new_vert = []
    color = []
    for i in range(vert.shape[0]):
        print(n_frame,round(i/vert.shape[0] * 100))
        u, v = min(round(640 * uv[i][0]), 639), min(round(480 * uv[i][1]), 479)
        color_this = bgr[v, u, :]
        b,g,r = color_this[0], color_this[1], color_this[2]
        h,s,v = colorsys.rgb_to_hsv(r,g,b)
        #print(i,h,s,v)
        h, s, v = int(h * 179), int(s * 255), v
        #if h > 60 and h < 81 and s > 70 and s < 255 and v > 70 and v < 255:
        if h > 28 and h < 60 and s > 18 and s < 135 and v > 129 and v < 255:
            roi_flag[i] = 1.0
            #print(i,v)
        
        if vert[i][2] == 0.0 or vert[i][2] > 130.0 or vert[i][2] <= 35.0:
            roi_flag[i] = 0.0
            
    for i in range(vert.shape[0]):
        if roi_flag[i] == 1.0:
            new_vert.append(vert[i])
            u, v = min(round(640 * uv[i][0]), 639), min(round(480 * uv[i][1]), 479)
            color_this = bgr[v, u, :]
            color.append(color_this)
    
    
    X_c_3d = np.array(new_vert)
    color = np.array(color)
    #%%
    X_c = np.concatenate((X_c_3d,np.ones((X_c_3d.shape[0],1))),axis=1)
    X_r = np.matmul(T_rc, X_c.transpose()).transpose()
    
    T_rg = np.array([[1.0, 0.0, 0.0, x_ef + x_c],
                      [0.0, 1.0, 0.0, y_ef + y_c],
                      [0.0, 0.0, 1.0, z_ef +z_c],
                      [0.0, 0.0, 0.0, 1.0]])
    
    T_gr = np.linalg.inv(T_rg)
    X_g = np.matmul(T_gr, X_r.transpose()).transpose()
    
    R_sg = rot_z
    T_sg = np.array([[R_sg[0,0], R_sg[0,1], R_sg[0,2],39.55],
                     [R_sg[1,0], R_sg[1,1], R_sg[1,2],60.182375],
                     [R_sg[2,0], R_sg[2,1], R_sg[2,2],60.448408],
                     [0.0, 0.0, 0.0, 1.0],])
    X_s = np.matmul(T_sg, X_g.transpose()).transpose()
    X_s *= 0.01
    X_s_3d = X_s[:, 0:3]

    
    np.save(f'../../../0AAA_all_exp_data/{exp_name}/exp_{exp_num}/pcd/x_{n_frame}.npy',X_s_3d.astype('float32'))
    np.save(f'../../../0AAA_all_exp_data/{exp_name}/exp_{exp_num}/pcd/color_{n_frame}.npy',color.astype('float32'))

#%%
fig = go.Figure(data=[go.Scatter3d(x=X_s[:,0], y=X_s[:,1], z=X_s[:,2],
                                    mode='markers',marker=dict(
        size=1, color = color            # set color to an array/list off desired values
        # opacity=0.8
    ))])


point = np.array([0.0, 0.0, 0.0])

trace = go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]], mode='markers',marker=dict(size=10, color = 'red'))
fig.add_trace(trace)
fig.show()

