#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 15:01:32 2022

@author: luyin
"""
import zarr
import open3d as o3d
import numpy as np



pcd = o3d.io.read_triangle_mesh('./Hanger/hanger.ply')
pcd.compute_vertex_normals(normalized=True)
points_np = np.asarray(pcd.vertices)
o3d.io.write_triangle_mesh('./o3d_mesh.ply',pcd)
# #%%

sampled_point = pcd.sample_points_uniformly(number_of_points= 40000)

sampled_point_np = np.asarray(sampled_point.points)

diff = 0
mid_point = []
for i in range(3):
    colum = points_np[:,i]
    max_this = np.amax(colum)
    min_this = np.amin(colum)
    mid_point_this = (max_this + min_this) / 2
    mid_point.append(mid_point_this)
    diff_this = max_this - min_this
    if diff_this > diff:
        diff = diff_this
#print(diff)
points_np -= np.array(mid_point)
points_np *= 0.7 * 1/diff



diff = 0
mid_point = []
for i in range(3):
    colum = sampled_point_np[:,i]
    max_this = np.amax(colum)
    min_this = np.amin(colum)
    mid_point_this = (max_this + min_this) / 2
    mid_point.append(mid_point_this)
    diff_this = max_this - min_this
    if diff_this > diff:
        diff = diff_this
#print(diff)
sampled_point_np -= np.array(mid_point)
sampled_point_np *= 0.7 * 1/diff
# #%%



points_np = points_np.astype('float32')
mesh_np = np.asarray(pcd.triangles)    
normals_np = np.asarray(pcd.vertex_normals).astype('float32')

mesh_np = mesh_np.reshape((mesh_np.shape[0] * 3,))

np.save('./Hanger/x.npy',points_np)
np.save('./Hanger/mesh.npy',mesh_np)
# np.save('./Hanger/normal.npy',normals_np)
np.save('./Hanger/sampled_point.npy',sampled_point_np)    
    

o3d.visualization.draw_geometries([pcd])

