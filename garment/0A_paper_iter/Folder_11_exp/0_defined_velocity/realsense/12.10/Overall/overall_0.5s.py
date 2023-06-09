# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 23:19:00 2019

@author: lancerhly
"""
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
#projected_img = np.zeros((480,640),np.uint8)
objpoints = objpoints.reshape(307200,3)
imagePoints, jacobian	=	cv2.projectPoints(objpoints, rvec, tvec, t_mtx, dist)
#imagePoints = imagePoints.reshape(307200,2)
imagePoints = imagePoints[:,0,:]

a = imagePoints[:,0]
a[a>=639] = 639
a[a<0] = 0
b = imagePoints[:,1]
b[b>=479] = 479
b[b<0] = 0
#projected_img = np.zeros((480,640),np.int8)
imagePoints =np.round((imagePoints).astype(int))
imagePoints = np.flip(imagePoints,axis=1)
projected_image = []
for location in imagePoints:
    a_this = thermal[(location[0],location[1])]
    projected_image.append([a_this])
projected_image = np.array(projected_image).reshape(480,640)
projected_img_color_1 = cv2.applyColorMap(projected_image, cv2.COLORMAP_JET)
projected_img_color = cv2.cvtColor(projected_img_color_1,cv2.COLOR_RGB2BGR)
ret,mask = cv2.threshold(projected_image,70,255,cv2.THRESH_BINARY)
rgb_img = cv2.imread('rgb_%i.jpg'%num)
fustion_img = cv2.bitwise_not(projected_img_color,rgb_img,mask =mask)
world_raw = np.load('world_%i.npy'%num)
world_reshape = world_raw.reshape((307200,3))
depth = world_reshape [:,2]
depth_raw = depth.reshape(480,640)
depth_milimeter = np.multiply(100,depth_raw)
depth_img = np.int16(depth_milimeter)

cv2.imwrite('fustion_img.jpg',fustion_img)
cv2.imwrite('depth_img.png',depth_img)

color_raw = o3d.io.read_image('fustion_img.jpg')
#color_raw = o3d.io.read_image('rgb_%i.jpg'%num)
depth_raw = o3d.io.read_image('depth_img.png')
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,convert_rgb_to_intensity=False)
#print(rgbd_image)
camera_model = o3d.camera.PinholeCameraIntrinsic()
camera_model.set_intrinsics(640,480,652.532,665.369,295.413,249.478)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,camera_model)
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
stop = timeit.default_timer()
print('Time: ', stop - start)
o3d.visualization.draw_geometries([pcd])
cv2.imwrite('0.jpg',projected_img_color_1)


