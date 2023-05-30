#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 15:01:32 2022

@author: luyin
"""
import zarr
import open3d as o3d
import numpy as np

pcd = o3d.io.read_triangle_mesh('Sports_Tee_Shirt_-_Loose_v1.0_natural_pose_grey.obj')
#pcd.compute_vertex_normals(normalized=True)
for i in range(0,400000,1000):
    print(i)
    points_np = np.load(f'obj_npy/{i}.npy')
    pcd.vertices = o3d.utility.Vector3dVector(points_np)
    o3d.io.write_triangle_mesh(f'obj/Tshirt_{i}.obj', pcd)

# #%%

# diff = 0

# for i in range(3):
#     colum = points_np[:,i]
#     max_this = np.amax(colum)
#     min_this = np.amin(colum)
#     diff_this = max_this - min_this
#     if diff_this > diff:
#         diff = diff_this
# points_np *= 0.3 * 1/diff
# #%%
# avgx = np.average(points_np, axis = 0)

# points_np -= avgx
# points_np = points_np.astype('float32')
# mesh_np = np.asarray(pcd.triangles)    
# normals_np = np.asarray(pcd.vertex_normals).astype('float32')

# mesh_np = mesh_np.reshape((mesh_np.shape[0] * 3,))

# np.save('./data/TShirt/x.npy',points_np)
# np.save('./data/TShirt/mesh.npy',mesh_np)
# np.save('./data/TShirt/normal.npy',normals_np)
    
    

#o3d.visualization.draw_geometries([pcd])

