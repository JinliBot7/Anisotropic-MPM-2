#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 16:15:13 2023

@author: luyin
"""

import numpy as np
import open3d as o3d


grid_n = 128
dx = 1 / grid_n

sdf = np.load('../Hanger/rio_sdf.npy')
max_sdf, min_sdf = np.amax(sdf), np.amin(sdf)
sdf *= 1/(max_sdf - min_sdf)
sdf_n = np.load('../Hanger/rio_sdf_n.npy')
sdf_color = np.zeros((sdf_n.shape[0],3))
sdf_color[:,0]  = 1 - sdf
grid_position = np.load('../Hanger/rio_grid_position.npy')
grid_index = np.load('../Hanger/rio_grid_index.npy')

pcd = o3d.io.read_triangle_mesh('../Hanger/hanger.ply')
x_rotated = np.load('../Hanger/x_rotated.npy')
x_rotated += np.array([0.5, 0.5, 0.5])
pcd.vertices = o3d.utility.Vector3dVector(x_rotated)
pcd.compute_vertex_normals()

sdf_pcd = o3d.geometry.PointCloud()
sdf_pcd.points = o3d.utility.Vector3dVector(grid_position)
sdf_pcd.colors = o3d.utility.Vector3dVector(sdf_color)
sdf_pcd.normals = o3d.utility.Vector3dVector(sdf_n)
# sdf_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
#         radius=0.1, max_nn=30))

o3d.visualization.draw_geometries([pcd,sdf_pcd])
