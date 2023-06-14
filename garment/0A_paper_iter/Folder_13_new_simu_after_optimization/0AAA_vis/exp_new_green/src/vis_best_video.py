#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 21:47:35 2023

@author: luyin
"""

import taichi as ti
import numpy as np
import sys
import time
import math
import os


sys.path.append('/home/luyin/Desktop/Anisotropic-MPM-2/garment/0A_paper_iter/Folder_13_new_simu_after_optimization/module')

from geometries import Obj, Hanger, Path, Path_Real
from compute_stress import compute_stress
from render_sim_real import set_render, render


trial_number = 0
drag_coe = 0.00024224343204456116
lift_coe = 7.542315237908129


ti.init(ti.cuda,device_memory_fraction = 0.9)
real = ti.f32
scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(3, dtype=real)

N_grid, dt = 128, 1e-4 # Grid number, substep time
dx, inv_dx = 1 / N_grid, N_grid
E, eta = 8e2, 0.3 # Young's modulus, Poisson's ratio
lam = E * eta / ((1 + eta) * (1 - 2 * eta)) # defined in https://encyclopediaofmath.org/wiki/Lam%C3%A9_constants
mu = E / (2 * (1 + eta))
damping = 0.0 # paper Eq. (2)

max_step = 20000 # step per iteration
max_iter = 200 # max iteration in a MPC update

beta1 = 0.5 # Adam parameters
beta2 = 0.999
k_ori = 0.04
decay_rate = 0.01

# record end effector
end_effector_right, end_effector_left = vec(), vec()
ti.root.dense(ti.i,max_step).place(end_effector_right,end_effector_left)


# Grid field place
grid_v_in, grid_v_out, grid_m = vec(), vec(), scalar()
block = ti.root.pointer(ti.ijk, (int(N_grid/4),int(N_grid/4),int(N_grid/4)))
pixel = block.dense(ti.ijk, (4,4,4))
pixel.place(grid_v_in,grid_v_out, grid_m)

loss= scalar()
ti.root.place(loss)

dim = 3
neighbour = (3,) * dim

#Set gripper initial velocity
v_input_num = 10
v_input = vec()
ti.root.dense(ti.l, v_input_num).place(v_input)

# initialize object
center, dimension = ti.Vector([0.5, 0.6, 0.5]), ti.Vector([0.25, 0.25, 0.0005]) # dimension changes shape. if dimension[i] < e_radius, single layer
axis_angle = ti.Vector([-91 / 180 * math.pi, 1.0, 0.0, 0.0]) # angle and rotation axis
e_radius, total_mass, pho = 0.0019, dimension[0] * dimension[1] * dimension[2], 2.0 # e_radius is the radius of the element. The distance between two elements is designed as 2 * e_radius.
color = (0.5,0.5,0.8)
obj = Obj(center, dimension, axis_angle, e_radius, total_mass, pho, color, max_step)

pio_target = vec()
ti.root.dense(ti.l, obj.pio_n).place(pio_target)

loss_compute = scalar()
ti.root.place(loss_compute)



@ti.kernel
def Particle_To_Grid(t: ti.i32, obj:ti.template(), target_far: ti.template(), target_near: ti.template()):
    for p in range(obj.n):
        x_p = obj.x[t,p]
        v_p = obj.v[t,p]
        #v_p += obj.grasp_flag[p] * (v_input[int(t / max_step * v_input_num)] - obj.v[t,p])
        
        # to keep the garment streched
        

        diff_far = target_far[0] - obj.x[t,0][0]
        diff_near = target_near[0] - obj.x[t,65][0]
        ef_v_far = ti.Vector([diff_far, 0.0, 0.0])
        ef_v_near = ti.Vector([diff_near, 0.0, 0.0])
        v_p += obj.grasp_flag[p] * (v_input[int(t / max_step * v_input_num)] - obj.v[t,p]) +  obj.grasp_flag_far[p] * ef_v_far + obj.grasp_flag_near[p] * ef_v_near
        v_p += obj.grasp_flag[p] * (-ti.Vector([v_p[0], 0.0, 0.0]))
        
        C_p = obj.C[t,p]
        F_p = obj.F[t,p]

        #Deformation update
        F_p += dt * C_p @ F_p
        
        # current normal
        normal  = ti.math.normalize(F_p @ ti.Vector([0.0, 0.0, 1.0]))
        proj_v = v_p.dot(normal)
        
        base = (x_p * inv_dx - 0.5).cast(int)  # x2[p]/dx transfer world coordinate to mesh coordinate
        fx = x_p * inv_dx - base.cast(float)  # fx is displacement vector
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2,]  # quadratic b spline
        
        # # Anisotropic
        P, F_elastic, F_plastic = compute_stress(F_p,mu,lam)
        obj.set_F(t + 1,F_elastic,p)
        dF_dC = F_elastic.transpose() 
        #dF_dC = F_plastic.inverse().transpose() @ F_p_ori.transpose() # identical to above equation
        
        # Neo-Hookean
        # J = F_p.determinant()
        # F_T = F_p.transpose()
        # dF_dC = F_p.transpose()
        # F_inv_T = F_T.inverse()
        # P = mu * (F_p - F_inv_T) + lam * ti.log(J) * F_inv_T
        # obj.set_F(t + 1,F_p,p)
        
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]

            grid_m[base + offset] += weight * obj.m  # mass conservation
            grid_v_in[base + offset] += weight * obj.m * v_p + weight * obj.m * C_p @ dpos # grid_v is acatually momentum, not velocity
            grid_v_in[base + offset] -= weight * 4 * dt * inv_dx * inv_dx * obj.vol * P @ dF_dC @ dpos
            grid_v_in[base + offset].z += weight * dt * obj.m * -lift_coe
            grid_v_in[base + offset] -= drag_coe * weight * obj.m * proj_v * normal

bound = 5
@ti.kernel
def Grid_Operations(boundary:ti.template()):
    for i, j, k in grid_v_in:
        v_out = grid_v_in[i, j, k]
        if i < bound and v_out.x < 0:
            v_out.x *= 0
            v_out.y *= 0.1
            v_out.z *= 0.1
        if i > N_grid - bound and v_out.x > 0:
            v_out.x *= 0
            v_out.y *= 0.1
            v_out.z *= 0.1
        if j < bound and v_out.y < 0:
            v_out.y *= 0
            v_out.x *= 0.1
            v_out.z *= 0.1
        if j > N_grid - bound and v_out.y > 0:
            v_out.y *= 0
            v_out.x *= 0.1
            v_out.z *= 0.1
        if k < bound and v_out.z < 0:
            v_out.z *= 0
            v_out.x *= 0.1
            v_out.y *= 0.1
        if k > N_grid - bound and v_out.z > 0:
            v_out.z *= 0
            v_out.x *= 0.1
            v_out.y *= 0.1
        
        grid_v_out[i, j, k] = v_out
        
    # for p in boundary.sdf:
    #     grid_index = boundary.sdf_index[p]
    #     if grid_m[grid_index] != 0:
    #         v_out = grid_v_in[grid_index]
    #         v_proj = v_out.dot(boundary.sdf_n[p])            
    #         if v_proj < 0:
    #             v_normal = v_proj * boundary.sdf_n[p]
    #             #v_tangent = v_out - v_normal # no friciton now
    #             v_out -= v_normal * ti.pow(3,-boundary.sdf[p])
    #             #v_out -= v_tangent * 0.5
                
    #             grid_v_out[grid_index] = v_out

@ti.kernel
def Grid_To_Particle(t: ti.i32, obj:ti.template()):
    for p in range(obj.n):
        x_p = obj.x[t,p]
        
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
                g_v = grid_v_out[base + offset] / (grid_m[base + offset] + 1e-16)

            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx * inv_dx
        
        sym_C, skew_C = (new_C + new_C.transpose()) * 0.5, (new_C - new_C.transpose()) * 0.5
        next_C = skew_C + (1 - damping) * sym_C
        
        obj.set_C(t + 1,next_C,p)
        obj.set_v(t + 1,new_v,p)
        x_p += dt * obj.v[t,p]
        obj.set_x(t + 1,x_p,p)
    
    for p in range(obj.pio_n):
        obj.pio_x[t + 1,p] = obj.x[t + 1,p % obj.pio_nx * obj.pio_scale + p // obj.pio_nx * obj.nx * obj.pio_scale]
        obj.pio_x_t[p] = obj.x[t + 1,p % obj.pio_nx * obj.pio_scale + p // obj.pio_nx * obj.nx * obj.pio_scale]

        
    
@ti.kernel
def Reset():
    for i, j, k in grid_m:
        grid_v_in[i, j, k] = [0.0, 0.0, 0.0]
        grid_v_out[i, j, k] = [0.0, 0.0, 0.0]
        grid_m[i, j, k] = 0.0
        
        grid_v_in.grad[i, j, k] = [0.0, 0.0, 0.0]
        grid_m.grad[i, j, k] = 0.0
        grid_v_out.grad[i, j, k] = [0.0, 0.0, 0.0]

@ti.ad.grad_replaced
def substep(counter, obj_0, hanger, target_far, target_near):
    Reset()
    #update_is_boundary(counter,obj_0)
    Particle_To_Grid(counter, obj_0, target_far, target_near)
    Grid_Operations(hanger)
    Grid_To_Particle(counter, obj_0)

@ti.ad.grad_for(substep)
def substep_grad(counter, obj_0, hanger, target_far, target_near):
    Reset()
    #update_is_boundary(counter,obj_0)
    Particle_To_Grid(counter, obj_0, target_far, target_near)
    Grid_Operations(hanger)
    
    Grid_To_Particle.grad(counter, obj_0)
    Grid_Operations.grad(hanger)
    Particle_To_Grid.grad(counter, obj_0, target_far, target_near)

@ti.kernel
def ini_v_input(obj:ti.template(), iter_num: ti.i32, v_input_ti: ti.template()):
    for i in range(10):
        v_input[i] = v_input_ti[iter_num,i]

@ti.kernel
def compute_loss(obj:ti.template(), target:ti.template()):
    for p in range(obj.pio_n):
        dist = ti.math.length(obj.pio_x[max_step - 1, p] - target[p])
        loss_compute[None] += dist

    
    
def record_data(iteration, r_loss,  r_v_input, r_grad,trial_number):
    r_loss[iteration] = loss[None]
    v_input_np = r_v_input.to_numpy()
    loss_np = r_loss.to_numpy()
    grad_np = r_grad.to_numpy()
    
    np.save(f'../data/trials/trial_{trial_number}/v_input.npy',v_input_np)
    np.save(f'../data/trials/trial_{trial_number}/loss.npy',loss_np)
    np.save(f'../data/trials/trial_{trial_number}/grad.npy',grad_np)
    # print(loss[None], r_loss[iteration],loss_np[iteration])
@ti.kernel
def record_v_input(iteration: ti.i32, v_input:ti.template(), grad: ti.template(), r_v_input: ti.template(), r_grad: ti.template()):
    for p in v_input:
        r_v_input[iteration,p] = v_input[p]
    for p in grad:
        r_grad[iteration,p] = grad[p]

# @ti.kernel
# def update_is_boundary(t: ti.i32, obj: ti.template()):
#     for p in range(obj.n):
#         if obj.x[t,p][1] < 0.55 and obj.v[t,p][1] < 0.0:
#             dist = 0.53 - obj.x[t,p][1]
#             obj.is_boundary[t,p] = ti.pow(4,100 * dist) + 0.1
#         else:
#             obj.is_boundary[t,p] = 0.0


@ti.kernel        
def record_end_effector(obj_0:ti.template()):
    for step in range(max_step):
        end_effector_right[step] = obj_0.pio_x[step,15]
        end_effector_left[step] = obj_0.pio_x[step,26]


def main():
    ti.root.lazy_grad()
   
    hanger = Hanger('../asset//Hanger',[0.5, 0.5, 0.5], (0.4, 0.4, 0.4))
    obj.initialize()
    target_far = obj.x[0,0] 
    target_near = obj.x[0,65] 
    path = Path('../asset/target_path', trial_number, (0.9,0.5,0.5), obj.n, obj.pio_n, obj.nx)
    path.initialize()
    pio_target_np = np.load(f'../asset/target_path/target_{trial_number}.npy')
    pio_target.from_numpy(pio_target_np)
            
    camera, scene, canvas, window, axisX, axisY, axisZ, colorX, colorY, colorZ = set_render()
    # fields for record
    
    loss = np.load(f'../data/trials/trial_{trial_number}/loss.npy')
    for i in range(loss.shape[0]):
        if loss[i] == 0.0:
            loss[i] += 100.0
    target_iter_num = np.where(loss == np.amin(loss))[0]
    

    iter_num = 1
    
    v_input_np = np.load(f'../data//trials/trial_{trial_number}/v_input.npy')
    v_input_ti = ti.Vector.field(3, ti.f32, shape=(200,10))
    v_input_ti.from_numpy(v_input_np)
    
    # Real Path
    exp_name = 'data_0612'
    exp_num = 0
    real_start_index = 25
    path_real = Path_Real(exp_name,exp_num,real_start_index)
    time_list = np.load(f'../../../0AAA_all_exp_data/{exp_name}/exp_{exp_num}/time.npy')
    
    # compute key frame number
    n_kf = 0
    kf_step_list = []
    for t in range(max_step - 1):
        
        next_real_time = time_list[n_kf + real_start_index + 1] - time_list[real_start_index]
        if t / max_step > next_real_time:
            n_kf += 1
            kf_step_list.append(t)
   
            
    while iter_num <= max_iter:

        if iter_num == target_iter_num:
            ini_v_input(obj, iter_num - 1, v_input_ti)
            loss_compute[None] = 0.0

            
            video_manager = ti.tools.VideoManager('../data/0A_suc_videos',video_filename=f'trial_{trial_number}',framerate = 100, automatic_build=False)
            kf_count = 0 # key frame count
            for t in range(max_step - 1):
                substep(t, obj, hanger, target_far, target_near)
                
                next_real_time = time_list[kf_count + real_start_index + 1] - time_list[real_start_index]
                if t / max_step * 2 > next_real_time:
                    path_real.update(exp_name,exp_num,real_start_index + kf_count + 1)
                    kf_count += 1
                
                if t % 50 == 0: 
                    render(t,[obj], [], [path], [path_real], camera, scene, canvas, window, axisX, axisY, axisZ, colorX, colorY, colorZ)
                    window.show()
                    img = window.get_image_buffer_as_numpy()
                    video_manager.write_frame(img)
                    
            video_manager.make_video(gif=False, mp4=True)   
            compute_loss(obj, pio_target)
            print(np.amin(loss), loss_compute[None])


        
        iter_num += 1
    
        
    #%%


if __name__ == "__main__":
    main()

























