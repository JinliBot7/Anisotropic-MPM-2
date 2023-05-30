#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:57:43 2023

@author: luyin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import taichi as ti
from geometries import Obj, Hanger
from render import set_render, render
from compute_stress import compute_stress
from math import pi
import time
import numpy as np
import sys 

#ti.init(ti.cpu, cpu_max_num_threads=1) # for debug
ti.init(ti.cuda,device_memory_fraction = 0.95, offline_cache = True, kernel_profiler = False)

# sparse part
real = ti.f32
scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(3, dtype=real)



N_grid, dt = 128, 5e-5 # Grid number, substep time

dx, inv_dx = 1 / N_grid, N_grid

E, eta = 15e2, 0.3 # Young's modulus, Poisson's ratio
lam = E * eta / ((1 + eta) * (1 - 2 * eta)) # defined in https://encyclopediaofmath.org/wiki/Lam%C3%A9_constants
mu = E / (2 * (1 + eta))

damping = 0.0 # paper Eq. (2)

grid_v = vec()
grid_m = scalar()
piexel_scale = 4
block = ti.root.pointer(ti.ijk, (int(N_grid/piexel_scale),int(N_grid/piexel_scale),int(N_grid/piexel_scale)))
pixel = block.dense(ti.ijk, (piexel_scale,piexel_scale,piexel_scale))
pixel.place(grid_v, grid_m)

# Object
dimension = ti.Vector([0.25, 0.25, 0.0005]) # dimension changes shape. if dimension[i] < e_radius, single layer
e_radius, total_mass, pho = 0.0019, 4356, 1.0 # e_radius is the radius of the element. The distance between two elements is designed as 2 * e_radius.
color = (0.5,0.8,0.5)

# Hanger
hanger = Hanger('./Hanger',[0.5, 0.5, 0.5], (0.4, 0.4, 0.4))

boundary_list = [hanger]
path_list = []

#%%
dim = 3
neighbour = (3,) * dim

speed = 2

@ti.kernel
def Particle_To_Grid(obj:ti.template(), counter:ti.f32):
    m = obj.m
    vol = obj.vol
    #ti.block_local(grid_v)
    for p in obj.x:
        x_p = obj.x[p]
        v_p = obj.v[p]
        C_p = obj.C[p]
        F_p = obj.F[p]
        
        # if p == 0 and counter % 100 == 0:
        # #     print(counter, counter*dt, error_length)
        # if counter > 4e3 and counter < 1e4:
        #     if (obj.is_grasping_point(p)):
        #         #error = ti.Vector([0.375, 0.55, 0.6]) - obj.x[0]
        #         error = ti.Vector([0.375, 0.51, 0.25]) - obj.x[0]
        #         v_p = [speed * error[0], speed * error[1], speed * error[2]]
        
        #Deformation update
        F_p += dt * C_p @ F_p
        
        base = (x_p * inv_dx - 0.5).cast(int)  # x2[p]/dx transfer world coordinate to mesh coordinate
        fx = x_p * inv_dx - base.cast(float)  # fx is displacement vector
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2,]  # quadratic b spline
        
        # Anisotropic
        P, F_elastic, F_plastic = compute_stress(F_p,mu,lam)
        obj.set_F(F_elastic,p)
        dF_dC = F_elastic.transpose() 
        #dF_dC = F_plastic.inverse().transpose() @ F_p_ori.transpose() # identical to above equation
        
        # # # Neo-Hookean
        # J = F_p.determinant()
        # F_T = F_p.transpose()
        # dF_dC = F_p.transpose()
        # F_inv_T = F_T.inverse()
        # P = mu * (F_p - F_inv_T) + lam * ti.log(J) * F_inv_T
        # obj.set_F(F_p,p)
        
        # FTF = F_T @ F_p
        # energy = (0.5 * mu * (FTF[0,0] + FTF[1,1] + FTF[2,2] - 3) - mu * ti.log(J) + 0.5 * lam * ti.log(J) ** 2) * (0.27 * 0.27 * 0.005) / (obj.n)
        # # if energy < 0:
        # #     print('wrong!', FTF[0,0] + FTF[1,1] + FTF[2,2])
        # #obj.energy[0] += energy
        # obj.energy[0] += x_p[2]


        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]

            grid_m[base + offset] += weight * m  # mass conservation
            grid_v[base + offset] += weight * m * v_p + weight * m * C_p @ dpos # grid_v is acatually momentum, not velocity
            grid_v[base + offset] -= weight * 4 * dt * inv_dx * inv_dx * vol * P @ dF_dC @ dpos
            grid_v[base + offset].z += weight * dt * m * -9.8
            


Circle_Center = ti.Vector.field(3, dtype = float, shape = 1)
Circle_Center[0] = [0.45, 0.5, 0.3]
Circle_Radius = 0.02
bound = 5
@ti.kernel
def Grid_Operations(boundary:ti.template()):
    for i, j, k in grid_v:
        if i < bound and grid_v[i, j, k].x < 0:
            grid_v[i, j, k].x *= 0
            grid_v[i, j, k].y *= 0.1
            grid_v[i, j, k].z *= 0.1
        if i > N_grid - bound and grid_v[i, j, k].x > 0:
            grid_v[i, j, k].x *= 0
            grid_v[i, j, k].y *= 0.1
            grid_v[i, j, k].z *= 0.1
            
        if j < bound and grid_v[i, j, k].y < 0:
            grid_v[i, j, k].y *= 0
            grid_v[i, j, k].x *= 0.1
            grid_v[i, j, k].z *= 0.1
        if j > N_grid - bound and grid_v[i, j, k].y > 0:
            grid_v[i, j, k].y *= 0
            grid_v[i, j, k].x *= 0.1
            grid_v[i, j, k].z *= 0.1
            
        if k < bound and grid_v[i, j, k].z < 0:
            grid_v[i, j, k].z *= 0
            grid_v[i, j, k].x *= 0.1
            grid_v[i, j, k].y *= 0.1
        if k > N_grid - bound and grid_v[i, j, k].z > 0:
            grid_v[i, j, k].z *= 0
            grid_v[i, j, k].x *= 0.1
            grid_v[i, j, k].y *= 0.1
        
        # dist = ti.Vector([i * dx, j * dx, k * dx]) - Circle_Center[0]
        # if dist.x ** 2 + dist.y ** 2 + dist.z ** 2 < Circle_Radius * Circle_Radius :
        #     dist = dist.normalized()
        #     grid_v[i, j, k] -= dist * ti.min(0, grid_v[i, j, k].dot(dist))
        #     grid_v[i, j, k] *= 0.9  #friction
        
    for p in boundary.sdf:
        grid_index = boundary.sdf_index[p]
        if grid_m[grid_index] != 0:
            v_proj = grid_v[grid_index].dot(boundary.sdf_n[p])            
            if v_proj < 0:
                v_normal = v_proj * boundary.sdf_n[p]
                v_tangent = grid_v[grid_index] - v_normal # no friciton now
                grid_v[grid_index] -= v_normal * ti.pow(3,-boundary.sdf[p])
                # grid_v[grid_index] -= v_normal * 1
                grid_v[grid_index] -= v_tangent * 0.05

@ti.kernel
def Check_Collision(boundary: ti.template(), collision_flag: ti.template()):
    for p in boundary.sdf:
        grid_index = boundary.sdf_index[p]
        if grid_m[grid_index] > 0.999:
            collision_flag[None] += 1.0          
            
@ti.kernel
def Grid_To_Particle(obj:ti.template()):
    for p in obj.x:
        x_p = obj.x[p]
        
        base = (x_p * inv_dx - 0.5).cast(int)

        fx = x_p * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 3)
        new_C = ti.Matrix.zero(float, 3, 3)
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            g_v = ti.Vector([0.0, 0.0, 0.0])
            if grid_m[base + offset] != 0:
                g_v = grid_v[base + offset] / grid_m[base + offset]

            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx * inv_dx
        
        sym_C, skew_C = (new_C + new_C.transpose()) * 0.5, (new_C - new_C.transpose()) * 0.5
        next_C = skew_C + (1 - damping) * sym_C
        obj.set_C(next_C,p)
        obj.set_v(new_v,p)
    
            
        x_p += dt * obj.v[p]
        obj.set_x(x_p,p)
        
    for p in range(obj.pio_n):
        obj.pio_x[p] = obj.x[p % obj.pio_nx * obj.pio_scale + p // obj.pio_nx * obj.nx * obj.pio_scale]


@ti.kernel
def Reset():
    for i, j, k in grid_m:
        grid_v[i, j, k] = [0.0, 0.0, 0.0]
        grid_m[i, j, k] = 0.0

@ti.func
def to_quaternion(arg):
    sin_theta = ti.math.sin(arg[0]/2)
    return ti.Vector([ti.math.cos(arg[0]/2),sin_theta * arg[1], sin_theta * arg[2], sin_theta * arg[3]])


collision_flag = scalar()
ti.root.place(collision_flag)

def main():
    counter = 0
    camera, scene, canvas, window, axisX, axisY, axisZ, colorX, colorY, colorZ = set_render()

    #ti.profiler.clear_kernel_profiler_info()  # Clears all records
    n_sample = 0
    sample_counter = 0
    while n_sample < 1000:
        print(sample_counter)
        sample_counter += 1
        z_rnd = np.random.uniform(low = 0.35, high = 0.45)
        rot_rnd = np.random.uniform(low = -180, high = -100)
        center_np = np.load(f'target_path/center_{n_sample}.npy')
        rot_np = np.load(f'target_path/rot_{n_sample}.npy')
        center, axis_angle = ti.Vector([0.5, 0.5, center_np[2]]), ti.Vector([rot_np[0], 1.0, 0.0, 0.0])
        print(center_np[2], rot_np[0])
        #center, axis_angle = ti.Vector([0.5, 0.5, 0.45]), ti.Vector([-130 / 180 * pi, 1.0, 0.0, 0.0]) 
        obj_0 = Obj(center, dimension, axis_angle, e_radius, total_mass, pho, color)
        obj_list = [obj_0]
        for counter in range(1):
            if window.get_event(ti.ui.PRESS):
                if window.event.key == ti.GUI.ESCAPE:
                    window.destroy()
            # obj_0.energy[0] = 0.0
            collision_flag[None] = 0.0

            
            #Reset()
    
            #Particle_To_Grid(obj_0, counter)
            
            #Check_Collision(hanger, collision_flag)

            
            
            #Grid_Operations(hanger)
            
            #Grid_To_Particle(obj_0)
            
            #update_dt()
            # if counter % 100 == 0: 
            render(obj_list, boundary_list, path_list, camera, scene, canvas, window, axisX, axisY, axisZ, colorX, colorY, colorZ,Circle_Center,Circle_Radius)
            #scene.mesh(floor, indices = floor_index, color = floor_color)
            window.show()
                
        
            #if counter > 2000 and (counter - 2000) % 100 == 0:
            # if collision_flag[None] <= 30.0:
            #     target_point_np = obj_0.pio_x.to_numpy()
            #     np.save(f'target_path/target_{n_sample}.npy',target_point_np)
                
            #     path_np = obj_0.x.to_numpy()
            #     np.save(f'target_path/path_{n_sample}.npy',path_np)
                
            #     center_np = np.array([0.5, 0.5, z_rnd])
            #     np.save(f'target_path/center_{n_sample}.npy',center_np)
                
            #     rot_np = np.array([rot_rnd / 180 * pi, 1.0, 0.0, 0.0])
            #     np.save(f'target_path/rot_{n_sample}.npy',rot_np)
                
            #     print(n_sample, z_rnd, rot_rnd)
            #     n_sample += 1
                
                # grid_m_np = grid_m.to_numpy()
                # np.save(f'target_path/target_m_{counter}.npy',grid_m_np)
                
                # energy_np = obj_0.energy.to_numpy()
                # np.save(f'target_path/deformation_energy_{counter}.npy',energy_np)
                
                # compute_pio_hanger_vec(obj_0,hanger)
                # pio_hanger_vec_np = obj_0.pio_hanger_vec.to_numpy()
                # np.save(f'target_path/pio_hanger_vec_{counter}.npy',pio_hanger_vec_np)
            n_sample += 1

@ti.kernel
def compute_pio_hanger_vec(obj: ti.template(), hanger: ti.template()):
    for h in range(hanger.n_sdf):
        target = dx * hanger.sdf_index[h]
        for o in range(obj.pio_n):
            obj.pio_hanger_vec[o,h] = target - obj.pio_x[o]
            

if __name__ == "__main__":
    main()
    # ti.sync()
    # ti.profiler.print_kernel_profiler_info()  # The default mode: 'count'

