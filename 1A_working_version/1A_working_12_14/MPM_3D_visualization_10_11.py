#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 18:30:50 2022

@author: luyin
"""

import taichi as ti
import numpy as np
from numpy import pi
import time
from compute_stress_113 import compute_stress

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(ti.cpu, cpu_max_num_threads=1)

N_grid =64
dx = 1 / N_grid
inv_dx = 1 / dx
dt = 1e-4


# Numpy variable initialization
# subscript '_list' dimention is N_l, number of lines
N_l = 1 # number of lines
L_list = [0.1] #  line lenth
pho_list = [1.0] # line density
radius_list = [0.04] # particle radius
start_pos_list = np.array([[0.5, 0.5, 0.4]])
orientation_list = np.array([[0.0, 0.0, -1.0]])
N_p_list = np.zeros(N_l) # number of points per line
M_list = np.zeros(N_l) # mass of a particle on a line
N_p = 0 # total point number initialization
for i in range(N_l):
    radius = radius_list[i] 
    length = L_list[i]
    pho = pho_list[i]
    # number of points and mass per point are calculated
    N_p_list[i] = L_list[i] // (radius_list[i] * 2 ) # number of points per line
    M_list[i] = pi * radius_list[i]**2 * L_list[i] * pho_list[i] / N_p_list[i] # mass per particle  
    N_p += int(N_p_list[i]) # accumulate total point number

# subscript '_list_total' dimention is N_p, number of points
# initialise positions
position_list_total = []
mass_list_total = []
volume_list_total = []
for i in range(N_l):
    N_o_p = int(N_p_list[i])
    length = L_list[i]
    start_pos = start_pos_list[i]
    orientation = orientation_list[i]
    mass = M_list[i]
    pho = pho_list[i]
    for j in range(N_o_p):
        position_list_total.append(list(start_pos + 2 * radius * j * orientation))
        mass_list_total.append(mass)
        volume_list_total.append(mass / pho)

position_list_total = np.array(position_list_total)
mass_list_total = np.array(mass_list_total)
volume_list_total = np.array(volume_list_total)

# fl
#%%
# Taichi variables intialization
x = ti.Vector.field(3, dtype=float, shape=N_p)  # position
m = ti.field(dtype=float, shape=N_p)  # mass
vol = ti.field(dtype=float, shape=N_p)  # volume
v = ti.Vector.field(3, dtype=float, shape=N_p)  # velocity
C = ti.Matrix.field(3, 3, dtype=float, shape=N_p)  # affine velocity field
F = ti.Matrix.field(3, 3, dtype=float, shape=N_p)  # deformation gradient


time = ti.field(dtype=float,shape=())

grid_v = ti.Vector.field(3, dtype=float, shape=(N_grid, N_grid, N_grid))  # grid velocity
grid_m = ti.field(dtype=float, shape=(N_grid, N_grid, N_grid))  # grid mass
grid_f = ti.Vector.field(3, dtype=float, shape=(N_grid, N_grid, N_grid))  # grid force


dim = 3
neighbour = (3,) * dim


E = 500
eta = 0.2 # poisson ratio
lam = E * eta / ((1 + eta) * (1 - 2 * eta))
mu = E / (2 * (1 + eta))
#mu = 0


x.from_numpy(position_list_total)
m.from_numpy(mass_list_total)
vol.from_numpy(volume_list_total)
@ti.kernel
def initialize():
    for p in x:
        v[p] = ti.Vector([0.0, 0.0, 0.0])
        C[p] = ti.Matrix.zero(float, 3, 3)
        F[p] = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    


@ti.kernel
def Particle_To_Grid():
    for p in x:

        # Deformation update
        F[p] += dt * C[p] @ F[p]
        #print(F[p])
        # Momentum and mass update
        base = (x[p] * inv_dx - 0.5).cast(int)  # x2[p]/dx transfer world coordinate to mesh coordinate
        fx = x[p] * inv_dx - base.cast(float)  # fx is displacement vector
        w = [
            0.5 * (1.5 - fx) ** 2,
            0.75 - (fx - 1) ** 2,
            0.5 * (fx - 0.5) ** 2,
        ]  # quadratic b spline
        mass = m[p]
        volume = vol[p]
        # Compute Piola Kirchhoff Stress
        U, sig, V = ti.svd(F[p])
        J = F[p].determinant()
        F_T = F[p].transpose()
        F_inv_T = F_T.inverse()
        P = mu*(F[p]-F_inv_T)+lam*ti.log(J)*F_inv_T
        
        P_new, F_elastic, F_plastic = compute_stress(F[p],mu,lam)
        P_new = P_new @ F_plastic.transpose().inverse()
        F[p] = F_elastic

        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]

            grid_m[base + offset] += weight * mass  # mass conservation
            grid_v[base + offset] += weight * mass * v[p] + weight * mass* C[p] @ dpos 
            grid_v[base + offset] -= weight * 4* dt * inv_dx * inv_dx * volume * P_new @ F_T @ dpos# momentum conservation


        #print('force:', force * 1e9)

        

bound = 4


@ti.kernel
def Grid_Operations():
    # Add gravity
    for i, j, k in grid_v:
        #grid_v[i,j,k].z -= 9.8*dt

        # boundary check
        if i < bound and grid_v[i, j, k].x < 0:
            grid_v[i, j, k].x = 0
            grid_v[i, j, k] *= 0.1  # friction
        if i > N_grid - bound and grid_v[i, j, k].x > 0:
            grid_v[i, j, k].x = 0
            grid_v[i, j, k] *= 0.1  # friction
        if j < bound and grid_v[i, j, k].y < 0:
            grid_v[i, j, k].y = 0
            grid_v[i, j, k] *= 0.1  # friction
        if j > N_grid - bound and grid_v[i, j, k].y > 0:
            grid_v[i, j, k].y = 0
            grid_v[i, j, k] *= 0.1  # friction
        if k < bound and grid_v[i, j, k].z < 0:
            grid_v[i, j, k].z = -grid_v[i, j, k].z
            grid_v[i, j, k] *= 0.1  # friction
        if k > N_grid - bound and grid_v[i, j, k].z > 0:
            grid_v[i, j, k].z = 0
            grid_v[i, j, k] *= 0.1  # friction


@ti.kernel
def Grid_To_Particle():
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 3)
        new_C = ti.Matrix.zero(float, 3, 3)
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            g_v = grid_v[base + offset] / grid_m[base + offset]
            g_v.z -= 9.8*dt*5 # Add gravity
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx * inv_dx

        C[p] = new_C
        #print(C[p])
        #print(F[p])
        v[p] = new_v
        x[p] += dt * v[p]
        time[None] +=dt
        #print(v[p].z,time[None])


@ti.kernel
def Reset():
    for i, j, k in grid_m:
        grid_v[i, j, k] = [0.0, 0.0, 0.0]
        grid_m[i, j, k] = 0.0
        grid_f[i, j, k] = [0.0, 0.0, 0.0]


# GUI
res = (1080, 720)
window = ti.ui.Window("MPM 3D", res, vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
gui = window.get_gui()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(0.9, 0.9, 0.3)
camera.lookat(0, 0, 0)
camera.up(0, 0, 1)
# camera.fov(60)
axisLength = 0.5
axisWideht = 2
axisX = ti.Vector.field(3, float, (2,))
axisX[1][0], axisX[1][1], axisX[1][2] = axisLength, 0, 0
colorX = (0.5, 0, 0)
axisY = ti.Vector.field(3, float, (2,))
axisY[1][0], axisY[1][1], axisY[1][2] = 0, axisLength, 0
colorY = (0, 0.5, 0)
axisZ = ti.Vector.field(3, float, (2,))
axisZ[1][0], axisZ[1][1], axisZ[1][2] = 0, 0, axisLength
colorZ = (0, 0, 0.5)


def render():
    camera.track_user_inputs(window, movement_speed=0.01, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    # scene.ambient_light((1, 1, 1))
    # scene.point_light(pos=(0, 0, 0.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.5))
    # scene.lines(x2, 0.01)
    scene.particles(x, radius=0.5*dx, color=(0, 0, 0))
    scene.lines(axisX, axisWideht, color=colorX)
    scene.lines(axisY, axisWideht, color=colorY)
    scene.lines(axisZ, axisWideht, color=colorZ)
    canvas.scene(scene)


def main():
    initialize()
    counter = 0
    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.GUI.ESCAPE:
                window.destroy()

        Reset()
        Particle_To_Grid()
        Grid_Operations()
        Grid_To_Particle()
        render()
        #time.sleep(0.1)
        window.show()
        
        # if counter % 100 == 0:
        #     print('F',F[0][2,2])
        #     print('v',v[0])


if __name__ == "__main__":
    main()
