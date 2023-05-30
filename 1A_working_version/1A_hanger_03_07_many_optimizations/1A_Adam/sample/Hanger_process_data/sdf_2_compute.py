#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 16:29:51 2023

@author: luyin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 12:09:30 2023
@author: luyin
"""

import numpy as np
import taichi as ti
ti.init(ti.cuda,device_memory_fraction = 0.5)

center = [0.5, 0.5, 0.5]
center_np  = np.array(center)
center_ti = ti.Vector(center)
grid_n = 128
dx = 1 / grid_n



points = np.load('../Hanger/sampled_points_rotated.npy')

block_dim = []
for i in range(3):
    diff = 0
    colum = points[:,i]
    max_this = np.amax(colum)
    min_this = np.amin(colum)
    diff_this = max_this - min_this
    if diff_this > diff:
        diff = diff_this
    diff /= dx
    diff += 10
    diff = round(diff)
    if diff % 2 != 0:
        diff += 1
    block_dim.append(diff)

block_dim_ti = ti.Vector([0.0, 0.0, 0.0])
block_dim_ti[0], block_dim_ti[1], block_dim_ti[2] = block_dim[0], block_dim[1], block_dim[2]

N = points.shape[0]
points_ti = ti.Vector.field(3, dtype = ti.f32, shape = N)

for p in range(N):
    points[p] += center_np
points_ti.from_numpy(points)

@ti.kernel
def compute(i: ti.i32, j: ti.i32, k: ti.i32, dx: ti.f32, center: ti.template(), block: ti.template()):
    disx, disy, disz = center[0], center[1], center[2]
    offsetx, offsety, offsetz = block[0],  block[1],  block[2]
    for p in grid_per_point:
        gp = [(i + disx * grid_n - offsetx * 0.5) * dx, (j + disy * grid_n - offsety * 0.5) * dx, (k + disz * grid_n - offsetz * 0.5) * dx]
        p0, p1, p2 = points_ti[p][0], points_ti[p][1], points_ti[p][2]
        dis = ((gp[0] - p0) ** 2 + (gp[1] - p1) ** 2 + (gp[2] - p2) ** 2) ** 0.5
        grid_per_point[p] = dis

@ti.kernel
def reset():
    for p in grid_per_point:
        grid_per_point[p] = 0.0
dis_list = []
normal_list = []
gird_index_list = []
grid_position = []

grid_per_point = ti.field(ti.f32,N)
#%%
for i, j, k in np.ndindex((block_dim[0], block_dim[1], block_dim[2])):
#for i, j, k in np.ndindex((2, 2, 2)):
    print(round((i * block_dim[1] * block_dim[2] + j * block_dim[2] + k) / (block_dim[0] * block_dim[1] * block_dim[2]) * 10000)/100, i, j, k)

    compute(i, j, k, dx, center_ti, block_dim_ti)
    grid_per_point_np = grid_per_point.to_numpy()
    min_dis = np.amin(grid_per_point_np)
    min_ind = np.argmin(grid_per_point_np)
    
    disx, disy, disz = center[0], center[1], center[2]
    offsetx, offsety, offsetz = block_dim[0],  block_dim[1],  block_dim[2]
    
    gp = [(i + disx * grid_n - offsetx * 0.5) * dx, (j + disy * grid_n - offsety * 0.5) * dx, (k + disz * grid_n - offsetz * 0.5) * dx]
    normal = np.array([gp[0] - points[min_ind][0], gp[1] - points[min_ind][1], gp[2] - points[min_ind][2]])
    length = (normal[0]**2 +  normal[1]**2 +  normal[2]**2) ** 0.5
    normal /= length
    
    dis_list.append(min_dis)
    normal_list.append(normal)
    gird_index_list.append([(i + disx * grid_n - offsetx * 0.5),(j + disy * grid_n - offsety * 0.5),(k + disz * grid_n - offsetz * 0.5)])
    grid_position.append(gp)

np.save('../Hanger/sdf.npy', np.array(dis_list))
np.save('../Hanger/sdf_normal.npy', np.array(normal_list))
np.save('../Hanger/grid_index.npy', np.array(gird_index_list))
np.save('../Hanger/grid_position.npy', np.array(grid_position))
# min_dis()
# for i, j, k in np.ndindex((grid_n,grid_n,grid_n)):
#     gp = [i * dx, j * dx, k * dx]
