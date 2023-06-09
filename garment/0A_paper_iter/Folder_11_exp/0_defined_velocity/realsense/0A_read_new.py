#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 14:24:00 2023

@author: luyin
"""

import open3d as o3d
import numpy as np
import cv2 as cv
import plotly.graph_objects as go
import plotly.io as pio
import colorsys
from numpy import cos, sin, pi
pio.renderers.default='browser'

# color_raw = o3d.io.read_image('data/0/rgb/rgb_img_0.jpg')
# depth_raw = o3d.io.read_image('data/0/depth/depth_img_0.png')
# depth_img = cv.imread('data/0/depth/depth_img_0.png')
# rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,convert_rgb_to_intensity=False)
# camera_model = o3d.camera.PinholeCameraIntrinsic()
# camera_model.set_intrinsics(640,480,462.80078125,461.91015625,299.10546875,244.515625)
# pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,camera_model)
# o3d.visualization.draw_geometries([pcd])
#depth_raw = cv.imread('data/0/depth/depth_img_0.png')

n_exp = 0
n_frame = 0
vert = np.load(f'data/{n_exp}/vert/vert_{n_frame}.npy')
uv = np.load(f'data/{n_exp}/uv/uv_{n_frame}.npy')
bgr = cv.imread(f'data/{n_exp}/rgb/rgb_img_{n_frame}.jpg')
roi_flag = np.zeros(vert.shape[0])

new_vert = []
color = []
for i in range(vert.shape[0]):
    print(n_exp,round(i/vert.shape[0] * 100))
    u, v = min(round(640 * uv[i][0]), 639), min(round(480 * uv[i][1]), 479)
    color_this = bgr[v, u, :]
    b,g,r = color_this[0], color_this[1], color_this[2]
    h,s,v = colorsys.rgb_to_hsv(r,g,b)
    #print(i,h,s,v)
    h, s, v = int(h * 179), int(s * 255), v
    #if h > 60 and h < 81 and s > 70 and s < 255 and v > 70 and v < 255:
    if h > 27 and h < 54 and s > 33 and s < 91 and v > 118 and v < 221:
        roi_flag[i] = 1.0
        #print(i,v)
    
    if vert[i][2] == 0.0 or vert[i][2] > 0.8:
        roi_flag[i] = 0.0
        
for i in range(vert.shape[0]):
    if roi_flag[i] == 1.0:
        new_vert.append(vert[i])
        u, v = min(round(640 * uv[i][0]), 639), min(round(480 * uv[i][1]), 479)
        color_this = bgr[v, u, :]
        color.append(color_this)


new_vert = np.array(new_vert)
color = np.array(color)
#%%
rvec, tvec = np.load(f'./data/{n_exp}/rvec.npy'), np.load(f'./data/{n_exp}/tvec.npy')
def rtvec_to_matrix(rvec, tvec):
 	T = np.eye(4)
 	R, jac = cv.Rodrigues(rvec)
 	T[:3, :3] = R
 	T[:3, 3] = tvec.reshape(3,)
 	return T

T = rtvec_to_matrix(rvec, tvec)
T_inv = np.linalg.inv(T)
new_vert_homo = np.concatenate((new_vert,np.ones((new_vert.shape[0],1))),axis=1)
new_vert_T_homo = np.matmul(T_inv,new_vert_homo.transpose()).transpose()
new_vert_T = np.divide(new_vert_T_homo[:, 0:3],new_vert_T_homo[:, 3].reshape(new_vert_T_homo[:, 3].shape[0],1))
 
#%%

theta_x = pi / 2
rot_x = np.array([
    [1.0, 0.0, 0.0],
    [0.0, cos(theta_x), -sin(theta_x)],
    [0.0, sin(theta_x), cos(theta_x)]
    ])
theta_z = pi
rot_z = np.array([
    [cos(theta_z), -sin(theta_z), 0.0],
    [sin(theta_z), cos(theta_z), 0.0],
    [0.0, 0.0, 1.0]
    ])

vert_robot = np.matmul(np.matmul(rot_x, rot_z), new_vert_T.transpose()).transpose()

vert_robot_sim = vert_robot + np.array([0.3955, 0.801824, 0.604484]) 

fig = go.Figure(data=[go.Scatter3d(x=vert_robot_sim[:,0], y=vert_robot_sim[:,1], z=vert_robot_sim[:,2],
                                    mode='markers',marker=dict(
        size=5, color = color            # set color to an array/list of desired values
        # opacity=0.8
    ))])

# fig.update_layout(
#     scene = dict(
#         xaxis = dict(nticks=4, range=[0,1],),
#                       yaxis = dict(nticks=4, range=[0,1],),
#                       zaxis = dict(nticks=4, range=[0,1],),),)
fig.show()
np.save(f'../0A_real_path/{n_exp}/pcd/x_{n_frame}.npy',vert_robot_sim.astype('float32'))
np.save(f'../0A_real_path/{n_exp}/pcd/color_{n_frame}.npy',color.astype('float32'))