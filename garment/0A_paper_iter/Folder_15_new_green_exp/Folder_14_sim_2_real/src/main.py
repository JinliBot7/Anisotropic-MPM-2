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


sys.path.append('/home/luyin/Desktop/Anisotropic-MPM-2/garment/0A_paper_iter/Folder_14_s2r_grid/module')

from geometries import Obj, Hanger, Path, Path_Real
from compute_stress import compute_stress
from render_diff import set_render, render


ti.init(ti.cuda,device_memory_fraction = 0.9)
real = ti.f32
scalar = lambda: ti.field(dtype = real)
scalar_int = lambda: ti.field(dytpe = ti.i32)
vec = lambda: ti.Vector.field(3, dtype = real)

N_grid, dt = 128, 1e-4 # Grid number, substep time
dx, inv_dx = 1 / N_grid, N_grid
E, eta = 8e2, 0.3 # Young's modulus, Poisson's ratio
lam = E * eta / ((1 + eta) * (1 - 2 * eta)) # defined in https://encyclopediaofmath.org/wiki/Lam%C3%A9_constants
mu = E / (2 * (1 + eta))
damping = 0.0 # paper Eq. (2)

max_step = 10000 # step per iteration
max_move_step = 5000
max_iter = 10 # max iteration in a MPC update

beta1 = 0.5 # Adam parameters
beta2 = 0.999
k_ori = 0.04
decay_rate = 0.01


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

# Parameter to optimize
opt_para = scalar()
n_opt_para = 2
ti.root.dense(ti.l, n_opt_para).place(opt_para)

pio_real_dist = scalar()
ti.root.dense(ti.l, 50000).dense(ti.axes(4), obj.pio_n).place(pio_real_dist)

# opt_para[0] is the drag coefficient, opt_para[1] is the lift coefficient in the fomr of gravitation

@ti.kernel
def Particle_To_Grid(t: ti.i32, obj:ti.template(), opt_para: ti.template()):
    for p in range(obj.n):
        x_p = obj.x[t,p]
        v_p = obj.v[t,p]
        
        if t < max_move_step:
            target_far = obj.pio_x[0,15] + ti.Vector([0.0, -0.2 * (t+1)/ max_move_step, 0.0])
            diff_far = target_far - obj.pio_x[t,15]
            target_near = obj.pio_x[0,26] + ti.Vector([0.0, -0.2 * (t+1)/ max_move_step, 0.0])
            diff_near = target_near - obj.pio_x[t,26]
            
            ef_v_far = 1.0 * ti.math.normalize(diff_far)
            ef_v_near = 1.0 * ti.math.normalize(diff_near)
            v_p += obj.grasp_flag_far[p] * (ef_v_far - obj.v[t,p])
            v_p += obj.grasp_flag_near[p] * (ef_v_near - obj.v[t,p])
        else:
            target_far = obj.pio_x[0,15] + ti.Vector([0.0, -0.2, 0.0])
            diff_far = target_far - obj.pio_x[t,15]
            target_near = obj.pio_x[0,26] + ti.Vector([0.0, -0.2, 0.0])
            diff_near = target_near - obj.pio_x[t,26]
        
            ef_v_far = 1.0 * ti.math.normalize(diff_far)
            ef_v_near = 1.0 * ti.math.normalize(diff_near)
            v_p += obj.grasp_flag_far[p] * (ef_v_far - obj.v[t,p])
            v_p += obj.grasp_flag_near[p] * (ef_v_near - obj.v[t,p])
            
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
            grid_v_in[base + offset].z += weight * dt * obj.m * -opt_para[1]  
            grid_v_in[base + offset] -= opt_para[0] * weight * obj.m * proj_v * normal

bound = 5
@ti.kernel
def Grid_Operations(boundary:ti.template()):
    for i, j, k in grid_v_in:
        v_out = 1.0 * grid_v_in[i, j, k]
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
def substep(counter, obj_0, hanger, opt_para):
    Reset()
    #update_is_boundary(counter,obj_0)
    Particle_To_Grid(counter, obj_0, opt_para)
    Grid_Operations(hanger)
    Grid_To_Particle(counter, obj_0)

@ti.ad.grad_for(substep)
def substep_grad(counter, obj_0, hanger, opt_para):
    Reset()
    #update_is_boundary(counter,obj_0)
    Particle_To_Grid(counter, obj_0, opt_para)
    Grid_Operations(hanger)
    
    Grid_To_Particle.grad(counter, obj_0)
    Grid_Operations.grad(hanger)
    Particle_To_Grid.grad(counter, obj_0, opt_para)

@ti.kernel
def ini_v_input():
    for p in range(v_input_num):
        #rd1, rd2, rd3 = ti.random() - 0.5, ti.random() - 0.5, ti.random() - 0.5
        v_input[p] = ti.Vector([0.0, -0.4, 0.0])

@ti.kernel
def compute_dist(obj:ti.template(), path_real:ti.template(),step:ti.i32):
    for p_real in range(50000):
        for p_obj in range(obj.pio_n):
            diff = obj.pio_x[step, p_obj] - path_real.x[p_real]

            dist = ti.math.length(diff)
            pio_real_dist[p_real,p_obj] += dist
    

@ti.kernel
def compute_loss(obj: ti.template(), path_real: ti.template(), this_kf_step: ti.i32, real_point_index:ti.i32, pio_index: ti.i32):
    dist = ti.math.length(obj.pio_x[this_kf_step, pio_index] - path_real.x[real_point_index])
    loss[None] += dist

    
    
def record_data(iteration, r_loss,  r_v_input, r_grad,trial_number):
    r_loss[iteration] = loss[None]
    v_input_np = r_v_input.to_numpy()
    loss_np = r_loss.to_numpy()
    grad_np = r_grad.to_numpy()
    opt_para_np = opt_para.to_numpy()
    
    np.save(f'../data/trials/trial_{trial_number}/v_input.npy',v_input_np)
    np.save(f'../data/trials/trial_{trial_number}/loss.npy',loss_np)
    np.save(f'../data/trials/trial_{trial_number}/grad.npy',grad_np)
    np.save(f'../data/trials/trial_{trial_number}/opt_para.npy',opt_para_np)

@ti.kernel
def record_v_input(iteration: ti.i32, v_input:ti.template(), grad: ti.template(), r_v_input: ti.template(), r_grad: ti.template()):
    for p in v_input:
        r_v_input[iteration,p] = v_input[p]
    for p in grad:
        r_grad[iteration,p] = grad[p]

@ti.kernel
def update_is_boundary(t: ti.i32, obj: ti.template()):
    for p in range(obj.n):
        if obj.x[t,p][1] < 0.55 and obj.v[t,p][1] < 0.0:
            dist = 0.53 - obj.x[t,p][1]
            obj.is_boundary[t,p] = ti.pow(4,100 * dist) + 0.1
        else:
            obj.is_boundary[t,p] = 0.0

@ti.kernel
def reset_pio_real_dist():
    for i,j in pio_real_dist:
        pio_real_dist[i,j] = 0.0

def main():
    ti.root.lazy_grad()
   
    hanger = Hanger('../asset//Hanger',[0.5, 0.5, 0.5], (0.4, 0.4, 0.4))
    obj.initialize()
    target_num = 0
    
    exp_name = 'data_0613'
    exp_num = 0
    real_start_index = 26
    # aruco_0 exp_num 0: 60 exp_num 1: 36
    path_real = Path_Real(exp_name,exp_num,real_start_index)
    time_list = np.load(f'../data/real_path/{exp_name}/{exp_num}/time.npy')
            
    camera, scene, canvas, window, axisX, axisY, axisZ, colorX, colorY, colorZ = set_render()
    # fields for record
    # r_v_input = ti.Vector.field(3, dtype = ti.f32, shape = (max_iter,v_input_num))
    # r_loss = ti.field(dtype = ti.f32, shape = max_iter)
    # r_grad = ti.Vector.field(3, dtype = ti.f32, shape = (max_iter,v_input_num))
    # V_nda = ti.ndarray(dtype=ti.math.vec3, shape=(2, v_input_num)) # For Adam
    # S_nda = ti.ndarray(dtype=ti.math.vec3, shape=(2, v_input_num)) # For Adam
    
    # initialize opt_para
    opt_para[0] =  0.0001 # drag
    opt_para[1] = 5.8      # lift
    
    # compute key frame number
    n_kf = 0
    kf_step_list = []
    for t in range(max_step - 1):
        next_real_time = time_list[n_kf + real_start_index + 1] - time_list[real_start_index]
        if t / max_step > next_real_time:
            n_kf += 1
            kf_step_list.append(t)

    # these two are to draw the line
    pio_vis_np = np.ones((196,n_kf,3)).astype(np.float32)
    real_vis_point_np = np.ones((196,n_kf,3)).astype(np.float32)
    
    min_pio_real_dist_index = np.ones((obj.pio_n,2)).astype(np.int32)

    line = ti.Vector.field(3, float, (392,))
    while True:
        loss[None] = 0.0
        
        trial_number_np = np.load('../asset/trial_num.npy')
        trial_number = trial_number_np[0]
        trial_number_np[0] +=1
        np.save('../asset/trial_num.npy',trial_number_np)
        
        try:
            os.mkdir(f'../data/trials/trial_{trial_number}')
        except FileExistsError:
            pass
            print('fiel alrady exist!')
        
        # Random initialization
        drag_coe = np.random.uniform(1e-4,1e-3)
        lift_coe = np.random.uniform(2.0,10.0)
        print(drag_coe,lift_coe)
        opt_para[0] = drag_coe
        opt_para[1] = lift_coe
        # compute the index of the real point that has minimum distance with the point of interest in the simulation for each key frame

        kf_count = 0 # key frame count
        
        video_manager = ti.tools.VideoManager('../data/0A_suc_videos',video_filename=f'trial_{trial_number}',framerate = 2, automatic_build=False)
        
        for t in range(max_step - 1):
            substep(t, obj, hanger, opt_para)
            if t == 0:
                #render
                render(t,[obj], [], [path_real], camera, scene, canvas, window, axisX, axisY, axisZ, colorX, colorY, colorZ, line, pio_vis_np, real_vis_point_np, kf_count)
                window.show()

            
            next_real_time = time_list[kf_count + real_start_index + 1] - time_list[real_start_index]

            if t / max_step > next_real_time:
                
                
                #print(kf_count)
                path_real.update(exp_name,exp_num,real_start_index + kf_count + 1)
                
                # compute min_pio_real_dist_index
                reset_pio_real_dist()
                compute_dist(obj, path_real, t)
                pio_real_dist_np = pio_real_dist.to_numpy()
                min_index_np = np.where(pio_real_dist_np == np.amin(pio_real_dist_np, axis=0))
                

                for p in range(196):
                    min_pio_real_dist_index[p,0] = min_index_np[0][p]
                    min_pio_real_dist_index[p,1] = min_index_np[1][p] #

                
                # compute loss    
                this_kf_step = kf_step_list[kf_count]
                
                for n in range(196):
                    pio_index = int(min_pio_real_dist_index[n,1])
                    real_point_index = int(min_pio_real_dist_index[n,0])
                
                    #print(this_min_pio_real_dist_index)
                    compute_loss(obj, path_real, this_kf_step,real_point_index, pio_index)
      
                    diff = obj.pio_x[this_kf_step, pio_index] - path_real.x[real_point_index]
                    #diff_length = (diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2) ** 0.5
       
                    pio_vis_np[pio_index, kf_count, :] = [obj.pio_x[this_kf_step, pio_index][0],obj.pio_x[this_kf_step, pio_index][1],obj.pio_x[this_kf_step, pio_index][2]]
                    real_vis_point_np[pio_index, kf_count, :] =  [path_real.x[real_point_index][0],path_real.x[real_point_index][1],path_real.x[real_point_index][2]]
                
                #render
                render(t,[obj], [], [path_real], camera, scene, canvas, window, axisX, axisY, axisZ, colorX, colorY, colorZ, line, pio_vis_np, real_vis_point_np, kf_count)
                window.show()
                #img = window.get_image_buffer_as_numpy()
                #video_manager.write_frame(img)

                kf_count += 1
            
        #video_manager.make_video(gif=False, mp4=True)   
            
            
        
        
        
                    
        
    
        
        drag_coe_np = np.array(drag_coe)
        lift_coe_np = np.array(lift_coe)
        loss_list_np = np.array(loss[None])
        np.save(f'../data/trials/trial_{trial_number}/drag_coe_np.npy',drag_coe_np)
        np.save(f'../data/trials/trial_{trial_number}/lift_coe.npy',lift_coe)
        np.save(f'../data/trials/trial_{trial_number}/loss.npy',loss_list_np)
        print(trial_number,loss[None] / (obj.pio_n * n_kf) * 100)
        
    #%%


if __name__ == "__main__":
    main()

























