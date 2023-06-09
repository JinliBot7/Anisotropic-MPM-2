# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 16:20:53 2019

@author: lancerhly
"""
from __future__ import division
import open3d as o3d
import numpy as np
import cv2
import timeit

num = 1
dist = np.load('thermal_dist.npy')
rvec = np.load('rvec.npy')
tvec = np.load('tvec.npy')
t_mtx = np.load('dist_thermal_mtx.npy')
extrinsic = np.load('extrinsic.npy')
objpoints = np.float64(np.load('world_%i.npy'%num))
rgb = cv2.imread('rgb_%i.jpg'%num,-1)
thermal = cv2.imread('thermal_%i.jpg'%num,-1)
start = timeit.default_timer()
objpoints = objpoints.reshape(307200,3)
imagePoints, jacobian	=	cv2.projectPoints(objpoints, rvec, tvec, t_mtx, dist)
imagePoints = imagePoints[:,0,:]



imagePoints =np.round((imagePoints).astype(int))
imagePoints = np.flip(imagePoints,axis=1)
row = imagePoints[:,0]
row[row>=479] = 479
row[row<0] = 0
col = imagePoints[:,1]
col[col>=639] = 639
col[col<0] = 0
projected_image = thermal[row,col]
projected_img_color_1 = cv2.applyColorMap(projected_image, cv2.COLORMAP_JET)
projected_img_color = cv2.cvtColor(projected_img_color_1,cv2.COLOR_RGB2BGR)
ret,mask = cv2.threshold(projected_image,70,255,cv2.THRESH_BINARY)
rgb_img = cv2.imread('rgb_%i.jpg'%num)
fustion_img = cv2.bitwise_not(projected_img_color,rgb_img,mask =mask)
fustion_img = fustion_img[:,0,:]/256
world_raw = np.load('world_%i.npy'%num).reshape(307200,3)
fusion_point_cloud = np.concatenate((world_raw,fustion_img),axis=1)
np.savetxt('fusion_point_cloud.txt',fusion_point_cloud)

pcd = o3d.io.read_point_cloud("fusion_point_cloud.txt", format='xyzrgb')
print(pcd)
o3d.visualization.draw_geometries([pcd])
