#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 16:15:13 2023

@author: luyin
"""

import numpy as np
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio
pio.renderers.default='browser'


grid_n = 128
dx = 1 / grid_n

sdf = np.load('../Hanger/sdf.npy')
sdf *= grid_n

sdf_n = np.load('../Hanger/sdf_normal.npy')
points = np.load('../Hanger/sampled_points_rotated.npy')
grid_index = np.load('../Hanger/grid_index.npy')
grid_position = np.load('../Hanger/grid_position.npy')


rio_sdf = []
rio_sdf_n = []
rio_grid_index = []
rio_grid_position = []
threshold = 2
for i in range(sdf.shape[0]):
    if sdf[i] <= threshold and sdf[i] >=1:
        rio_sdf.append(sdf[i])
        rio_sdf_n.append(sdf_n[i])
        rio_grid_index.append(grid_index[i])
        rio_grid_position.append(grid_position[i])

rio_sdf = np.array(rio_sdf)
rio_sdf_n = np.array(rio_sdf_n)
rio_grid_index = np.array(rio_grid_index)
rio_grid_position = np.array(rio_grid_position)

center = [0.5, 0.5, 0.5]
center_np = np.array(center)
N = points.shape[0]
for p in range(N):
    points[p] += center_np

# def get_block_dim():
#     block_dim = []
#     for i in range(3):
#         diff = 0
#         colum = points[:,i]
#         max_this = np.amax(colum)
#         min_this = np.amin(colum)
#         diff_this = max_this - min_this
#         if diff_this > diff:
#             diff = diff_this
#         diff /= dx
#         diff += 4
#         diff = round(diff)
#         if diff % 2 != 0:
#             diff += 1
#         block_dim.append(diff)
#     return block_dim

# block_dim = get_block_dim()
# #print(block_dim)

# disx, disy, disz = center[0], center[1], center[2]
# offsetx, offsety, offsetz = block_dim[0],  block_dim[1],  block_dim[2]

# grid_point = []

# for i, j, k in np.ndindex((block_dim[0], block_dim[1], block_dim[2])):
#     gp = [(i + disx * grid_n - offsetx * 0.5) * dx, (j + disy * grid_n - offsety * 0.5) * dx, (k + disz * grid_n - offsetz * 0.5) * dx]
#     grid_point.append(gp)
#     #print(i)

# grid_point = np.array(grid_point)    

fig = go.Figure(data=[go.Scatter3d(x=rio_grid_position[:,0], y=rio_grid_position[:,1], z=rio_grid_position[:,2],
                                   mode='markers',marker=dict(
        size=2.0,
        color=rio_sdf_n,                # set color to an array/list of desired values
        colorscale='gray',   # choose a colorscale
        colorbar=dict(thickness=20)
        # opacity=0.8
    ))])

# trace2 = go.Scatter3d(
#     x=points[:,0],
#     y=points[:,1],
#     z=points[:,2],
#     mode='markers',marker=dict(size=2.0))
# fig.add_trace(trace2)

#fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)

fig.update_layout(
    scene = dict(
        xaxis = dict(nticks=4, range=[0,1],),
                      yaxis = dict(nticks=4, range=[0,1],),
                      zaxis = dict(nticks=4, range=[0,1],),),)

# fig.update_layout(
#     scene = dict(
#         xaxis = dict(nticks=4, range=[-0.5,0.5],),
#                      yaxis = dict(nticks=4, range=[-0.5,-0.5],),
#                      zaxis = dict(nticks=4, range=[-0.5,0.5],),),)

fig.show()



np.save('../Hanger/rio_sdf.npy', rio_sdf)
np.save('../Hanger/rio_sdf_n.npy', rio_sdf_n)
np.save('../Hanger/rio_grid_index.npy', rio_grid_index)
np.save('../Hanger/rio_grid_position.npy', rio_grid_position)

