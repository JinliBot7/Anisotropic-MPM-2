#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 15:01:32 2022

@author: luyin
"""
import zarr
import open3d as o3d
import numpy as np
import taichi as ti

ti.init(ti.cuda)


#pcd = o3d.io.read_triangle_mesh('./Hanger/hanger.ply')
#print(pcd)
#pcd.compute_vertex_normals(normalized=True)
points_np = np.load('../Hanger/x.npy').astype(np.float32)
sampled_points_np = np.load('../Hanger/sampled_point.npy').astype(np.float32)

x = ti.Vector.field(3, dtype = ti.f32, shape = points_np.shape[0])
x.from_numpy(points_np)
sampled_x = ti.Vector.field(3, dtype = ti.f32, shape = sampled_points_np.shape[0])
sampled_x.from_numpy(sampled_points_np)

axis = [90 / 180 * ti.math.pi, 1.0, 0.0, 0.0]


def axis_to_quaternion(arg):
    sin_theta = ti.math.sin(arg[0]/2)
    return ti.Vector([ti.math.cos(arg[0]/2),sin_theta * arg[1], sin_theta * arg[2], sin_theta * arg[3]])

@ti.kernel
def rotate(x:ti.template(),quaternion: ti.types.vector(4,ti.f32)): # quaternion rotation from wiki
    for p in x:
        complex_p = [0.0, x[p][0], x[p][1], x[p][2]]
        complex_q = quaternion
        complex_q_inv = [quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]]
        p_prime = quat_mul(quat_mul(complex_q, complex_p),complex_q_inv)
        x[p][0], x[p][1], x[p][2] = p_prime[1] , p_prime[2] , p_prime[3] 

@ti.func
def quat_mul(q1, q2):# quaternion multiplication from wiki
    a1, b1, c1, d1 = q1[0], q1[1], q1[2], q1[3]
    a2, b2, c2, d2 = q2[0], q2[1], q2[2], q2[3]
    t1 = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
    t2 = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
    t3 = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
    t4 = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2
    result = [t1, t2, t3, t4]
    return result

rotate(x,axis_to_quaternion(axis))
x_new_np = x.to_numpy()

rotate(sampled_x,axis_to_quaternion(axis))
sampled_points_np_new = sampled_x.to_numpy()

# center to 0




np.save('../Hanger/x_rotated.npy',x_new_np)
np.save('../Hanger/sampled_points_rotated.npy',sampled_points_np_new)
# np.save('./Hanger/mesh.npy',mesh_np)
# np.save('./Hanger/normal.npy',normals_np)
    
    

#o3d.visualization.draw_geometries([pcd])

