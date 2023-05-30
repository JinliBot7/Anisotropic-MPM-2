#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 21:47:35 2023

@author: luyin
"""

import taichi as ti
import numpy as np
import os
import time
from geometries import Obj, Hanger, Path
from math import pi
from compute_stress import compute_stress
from render_diff import set_render, render


#ti.init(debug=True)
ti.init(ti.cuda,device_memory_fraction = 0.9)
#ti.init(ti.cpu)
real = ti.f32
scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(3, dtype=real)

N_grid, dt = 128, 5e-5 # Grid number, substep time
dx, inv_dx = 1 / N_grid, N_grid
E, eta = 15e2, 0.3 # Young's modulus, Poisson's ratio
lam = E * eta / ((1 + eta) * (1 - 2 * eta)) # defined in https://encyclopediaofmath.org/wiki/Lam%C3%A9_constants
mu = E / (2 * (1 + eta))
damping = 0.0 # paper Eq. (2)

max_MPC_iter = 1 # maximum number of MPC update
max_step = 20000 # step per iteration
max_iter = 500 # max iteration in a MPC update

beta1 = 0.5
beta2 = 0.999

grid_v_in = vec()
grid_v_out = vec() 
grid_m = scalar()
block = ti.root.pointer(ti.ijk, (int(N_grid/4),int(N_grid/4),int(N_grid/4)))
pixel = block.dense(ti.ijk, (4,4,4))
pixel.place(grid_v_in,grid_v_out, grid_m)

loss, loss_standard, loss_position, loss_velocity, loss_all, pio_all = scalar(), scalar(), scalar(), scalar(), scalar(), scalar()
loss_field = scalar()
ti.root.place(loss, loss_standard,loss_position,loss_velocity, loss_all, pio_all)
ti.root.dense(ti.l, max_step).place(loss_field)

#Set gripper initial velocity
v_input_num = 100
v_input = vec()
ti.root.dense(ti.l, v_input_num).place(v_input)

# initialize object
center, dimension = ti.Vector([0.5, 0.8, 0.5]), ti.Vector([0.25, 0.25, 0.0005]) # dimension changes shape. if dimension[i] < e_radius, single layer
axis_angle = ti.Vector([-90 / 180 * pi, 1.0, 0.0, 0.0]) # angle and rotation axis
e_radius, total_mass, pho = 0.0019, dimension[0] * dimension[1] * dimension[2], 2.0 # e_radius is the radius of the element. The distance between two elements is designed as 2 * e_radius.
color = (0.5,0.5,0.8)
obj = Obj(center, dimension, axis_angle, e_radius, total_mass, pho, color, max_step)

dim = 3
neighbour = (3,) * dim

pio_target = vec()
ti.root.dense(ti.l, obj.pio_n).place(pio_target)

@ti.kernel
def Particle_To_Grid(t: ti.i32, obj:ti.template()):
    for p in range(obj.n):
        x_p = obj.x[t,p]
        v_p = obj.v[t,p]
        v_p += obj.grasp_flag[p] * (v_input[int(t / max_step * v_input_num)] - obj.v[t,p])
        C_p = obj.C[t,p]
        F_p = obj.F[t,p]

        #Deformation update
        F_p += dt * C_p @ F_p
        
        base = (x_p * inv_dx - 0.5).cast(int)  # x2[p]/dx transfer world coordinate to mesh coordinate
        fx = x_p * inv_dx - base.cast(float)  # fx is displacement vector
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2,]  # quadratic b spline
        
        # # Anisotropic
        # P, F_elastic, F_plastic = compute_stress(F_p,mu,lam)
        # obj.set_F(t + 1,F_elastic,p)
        # dF_dC = F_elastic.transpose() 
        # #dF_dC = F_plastic.inverse().transpose() @ F_p_ori.transpose() # identical to above equation
        
        # Neo-Hookean
        J = F_p.determinant()
        F_T = F_p.transpose()
        dF_dC = F_p.transpose()
        F_inv_T = F_T.inverse()
        P = mu * (F_p - F_inv_T) + lam * ti.log(J) * F_inv_T
        obj.set_F(t + 1,F_p,p)
        
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]

            grid_m[base + offset] += weight * obj.m  # mass conservation
            grid_v_in[base + offset] += weight * obj.m * v_p + weight * obj.m * C_p @ dpos # grid_v is acatually momentum, not velocity
            grid_v_in[base + offset] -= weight * 4 * dt * inv_dx * inv_dx * obj.vol * P @ dF_dC @ dpos
            grid_v_in[base + offset].z += weight * dt * obj.m * -9.8  

bound = 5
@ti.kernel
def Grid_Operations(boundary:ti.template()):
    for i, j, k in grid_v_in:
        v_out = 0.9997 * grid_v_in[i, j, k]
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
        
        # dist = ti.Vector([i * dx, j * dx, k * dx]) - Circle_Center[0]
        # if dist.x ** 2 + dist.y ** 2 + dist.z ** 2 < Circle_Radius * Circle_Radius :
        #     dist = dist.normalized()
        #     grid_v[i, j, k] -= dist * ti.min(0, grid_v[i, j, k].dot(dist))
        #     grid_v[i, j, k] *= 0.9  #friction
        
    for p in boundary.sdf:
        grid_index = boundary.sdf_index[p]
        if grid_m[grid_index] != 0:
            v_out = grid_v_in[grid_index]
            v_proj = v_out.dot(boundary.sdf_n[p])            
            if v_proj < 0:
                v_normal = v_proj * boundary.sdf_n[p]
                #v_tangent = v_out - v_normal # no friciton now
                v_out -= v_normal * boundary.sdf[p]
                #v_out -= v_tangent * 0.5
                
                grid_v_out[grid_index] = v_out

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
        obj.pio_v[t + 1,p] = obj.v[t + 1,p % obj.pio_nx * obj.pio_scale + p // obj.pio_nx * obj.nx * obj.pio_scale]
        
    
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
def substep(counter, obj_0, hanger):
    Reset()
    Particle_To_Grid(counter, obj_0)
    Grid_Operations(hanger)
    Grid_To_Particle(counter, obj_0)

@ti.ad.grad_for(substep)
def substep_grad(counter, obj_0, hanger):
    Reset()
    Particle_To_Grid(counter, obj_0)
    Grid_Operations(hanger)
    
    Grid_To_Particle.grad(counter, obj_0)
    Grid_Operations.grad(hanger)
    Particle_To_Grid.grad(counter, obj_0)

@ti.kernel
def ini_v_input():
    for p in range(v_input_num):
        v_input[p] = ti.Vector([0.0, -0.3,0.0])

@ti.kernel
def compute_this_loss(obj:ti.template(), target:ti.template(),step: ti.i32):
    for p in range(obj.pio_n):
        dist = ti.math.length(obj.pio_x[step, p] - target[p])
        loss_field[step] += dist
    #loss_field[step] = accu_loss
    
def compute_minimum_index(obj,target,minimum_loss, minimu_index):
    for step in range(max_step):
        compute_this_loss(obj, target, step)
    for step in range(max_step):
        minimum_loss = min(minimum_loss, loss_field[step])
    for step in range(max_step):
        if loss_field[step] == minimum_loss:
            minimum_index = int(step)
    return minimum_index


beta_loss = 0.05
v_loss_parameter = 0.1
v_threshold = 0.5
@ti.kernel
def compute_loss(obj:ti.template(), hanger: ti.template(), target:ti.template(),pio_hanger_vec_target:ti.template(), v_input: ti.template(), minimum_index: ti.i32):

    for p in range(obj.pio_n):
        dist = obj.pio_coefficient[p] * ti.math.length(obj.pio_x[minimum_index, p] - target[p])
        loss[None] += dist 
        loss_position[None] += dist
    
    for p in range(v_input_num):
        v_length = (v_input[p][0] ** 2 + v_input[p][1] ** 2 + v_input[p][2] ** 2) ** 0.5
        if v_length >= v_threshold:
            loss[None] += v_loss_parameter * v_length / v_input_num
            loss_velocity[None] += v_loss_parameter * v_length / v_input_num
    
        
        

@ti.kernel
def update_pio_coe(obj:ti.template(), target:ti.template(),loss_nda: ti.types.ndarray(), iter_num: ti.i32):
    for p in range(obj.pio_n):
        dist = obj.pio_coefficient[p] * ti.math.length(obj.pio_x[max_step-1, p] - target[p])
        loss_nda_this = beta_loss * loss_nda[0, p] + (1 - beta_loss) * dist
        loss_nda[1, p] = loss_nda_this / (1 - beta_loss ** iter_num)
        loss_nda[0, p] +=  loss_nda[1, p] - loss_nda[0, p]
        loss_all[None] += loss_nda[1, p]

    for p in range(obj.pio_n):
        obj.pio_coefficient[p] = loss_nda[1, p] / (loss_all[None] + 1e-10)
        pio_all[None] += obj.pio_coefficient[p]
    print('loss_all: ', loss_all[None])
    # for o in range(obj.pio_n):
    #     obj.pio_coefficient[o] = obj.pio_this_loss[o] / loss[None]
    #     obj.pio_this_loss[o] = 0.0
    
    
    

                    
            
            
    # for step in range(max_step-2):
    #     if (obj.pio_x[step, 0][1] - 0.6) ** 2 < 0.004:
    #         loss[None] += ((obj.pio_x[step, 0][1] - 0.6) ** 2) ** 0.5
    #         loss_velocity[None] +=  ((obj.pio_x[step, 0][1] - 0.6) ** 2) ** 0.5
    #         if obj.pio_x[step, 0][1] - 0.5 < 0:
    #             loss[None] += 5 * ((obj.pio_x[step, 0][1] - 0.5) ** 2) ** 0.5
    #             loss_velocity[None] += 5 *  ((obj.pio_x[step, 0][1] - 0.5) ** 2) ** 0.5
    
def record_data(iteration, r_loss, r_loss_standard, r_loss_position, r_loss_velocity,  r_v_input, r_grad,r_minimum_index):
    r_loss[iteration] = loss[None]
    r_loss_standard[iteration] = loss_standard[None]
    r_loss_position[iteration] = loss_position[None]
    r_loss_velocity[iteration] = loss_velocity[None]
    v_input_np = r_v_input.to_numpy()
    loss_np = r_loss.to_numpy()
    loss_standard_np = r_loss_standard.to_numpy()
    loss_position_np = r_loss_position.to_numpy()
    loss_velocity_np = r_loss_velocity.to_numpy()
    r_minimum_index_np = r_minimum_index.to_numpy()
    # grad_np = r_grad.to_numpy()
    
    np.save('./data/v_input.npy',v_input_np)
    np.save('./data/loss.npy',loss_np)
    np.save('./data/loss_standard.npy',loss_standard_np)
    np.save('./data/loss_position.npy',loss_position_np)
    np.save('./data/loss_velocity.npy',loss_velocity_np)
    np.save('./data/minimum_index_np.npy',r_minimum_index_np)
    # np.save('./data/grad.npy',grad_np)


@ti.kernel
def record_v_input(iteration: ti.i32, v_input:ti.template(), grad: ti.template(), r_v_input: ti.template(), r_grad: ti.template()):
    for p in v_input:
        r_v_input[iteration,p] = v_input[p]
    # for p in grad:
    #     r_grad[iteration,p] = grad[p]

@ti.kernel
def mpc_reset(V_nda: ti.types.ndarray(), S_nda: ti.types.ndarray()):
    for p in range(50):
        v_input[p] = v_input[50 + p]
        for j in range(3):
            V_nda[0, p][j] = V_nda[0, 50+p][j]
            S_nda[0, p][j] = S_nda[0, 50+p][j]
            
    for p in range(50):
        v_input[50 + p] = ti.Vector([0.0,0.0,0.0])
        for j in range(3):
            V_nda[0, 50+p][j] = 0.0
            S_nda[0, 50+p][j] = 0.0

@ti.kernel
def compute_pio_hanger_vec(obj: ti.template(), hanger: ti.template()):
    for h in range(hanger.n_sdf):
        target = dx * hanger.sdf_index[h]
        for o in range(obj.pio_n):
            obj.pio_hanger_vec[o,h] = target - obj.pio_x_t[o]

def main():
    ti.root.lazy_grad()
    target_num = 0
    pio_target_np = np.load(f'./target_path/target_{target_num}.npy')
    pio_target.from_numpy(pio_target_np)
    
    obj.initialize()
    
    hanger = Hanger('./Hanger',[0.5, 0.5, 0.5], (0.4, 0.4, 0.4))
    path = Path('./target_path', target_num, (0.5,0.8,0.5), obj.n, obj.pio_n, obj.nx)
    path.initialize()
    
    pio_hanger_ver_target_np = np.load(f'./target_path/pio_hanger_vec_{target_num}.npy')
    pio_hanger_vec_target = vec()
    ti.root.dense(ti.l, obj.pio_n).dense(ti.axes(4),2196).place(pio_hanger_vec_target)
    pio_hanger_vec_target.from_numpy(pio_hanger_ver_target_np)

    camera, scene, canvas, window, axisX, axisY, axisZ, colorX, colorY, colorZ = set_render()
    floor = ti.Vector.field(3,dtype = ti.f32,shape = 4)
    floor_index = ti.field(dtype = ti.i32,shape = 6)
    floor[0], floor[1], floor[2], floor[3] = [0.0, 0.0, -1e-3], [1.0, 0.0, -1e-3], [0.0, 1.0, -1e-3], [1.0, 1.0, -1e-3]
    floor_index[0], floor_index[1],floor_index[2], floor_index[3], floor_index[4], floor_index[5] = 0, 1, 2, 3, 2, 1

    
    # fields for record
    r_v_input = ti.Vector.field(3, dtype = ti.f32, shape = (max_iter * max_MPC_iter,v_input_num))
    r_loss = ti.field(dtype = ti.f32, shape = max_iter * max_MPC_iter)
    r_loss_standard = ti.field(dtype = ti.f32, shape = max_iter * max_MPC_iter)
    r_loss_position = ti.field(dtype = ti.f32, shape = max_iter * max_MPC_iter)
    r_loss_velocity = ti.field(dtype = ti.f32, shape = max_iter * max_MPC_iter)
    r_grad = ti.Vector.field(3, dtype = ti.f32, shape = (max_iter,v_input_num))
    r_minimum_index = ti.field(dtype = ti.i32, shape = max_iter * max_MPC_iter)
    V_nda = ti.ndarray(dtype=ti.math.vec3, shape=(2, v_input_num)) # For Adam
    S_nda = ti.ndarray(dtype=ti.math.vec3, shape=(2, v_input_num)) # For Adam
    #dv_nda = ti.ndarray(dtype=ti.math.vec3, shape=(max_iter, v_input_num)) # For Adam
    
    loss_nda = ti.ndarray(dtype=ti.f32, shape=(2, obj.pio_n))
    
    MPC_iter_num = 0
    #ti.profiler.clear_kernel_profiler_info()

    k_ori = 0.04
    decay_rate = 0.005
    

    ini_v_input()
    



    while MPC_iter_num < max_MPC_iter:
        if MPC_iter_num == 0:
            pass
        else:
            obj.mpc_initialize(max_step)
            mpc_reset(V_nda, S_nda)
        iter_num = 1
        
        
        while iter_num <= max_iter:
            k = k_ori / (1 + iter_num * decay_rate)
            
            for t in range(max_step - 1):
                substep(t, obj, hanger)
                
                if t % 100 == 0: 
                    render(t,[obj], [hanger], [path], camera, scene, canvas, window, axisX, axisY, axisZ, colorX, colorY, colorZ)
                    #scene.mesh(floor, indices = floor_index, color = floor_color)
                    window.show()
            
            loss[None] = 0.0
            loss_standard[None] = 0.0
            loss_position[None] = 0.0
            loss_velocity[None] = 0.0
            loss_all[None] = 0.0
            pio_all[None] = 0.0
            #Gradient part
            with ti.ad.Tape(loss=loss):
                for t in range(max_step - 1):
                    substep(t, obj, hanger)
                
                #minimum_index = compute_minimum_index(obj,pio_target,1000.0,0)
                #r_minimum_index[MPC_iter_num*(max_iter-1) + iter_num -1] = minimum_index
                #compute_pio_hanger_vec(obj,hanger)
                compute_loss(obj, hanger, pio_target, pio_hanger_vec_target, v_input,max_step-1)

            update_pio_coe(obj, pio_target, loss_nda, iter_num)



            grad = v_input.grad
            if iter_num == 0:
                pass
            else:
                for i in range(v_input_num):
                    for j in range(3):
                        V_nda[1, i][j] = (beta1 * V_nda[0, i][j] + (1 - beta1) * grad[i][j]) 
                        S_nda[1, i][j] = (beta2 * S_nda[0, i][j] + (1 - beta2) * grad[i][j] ** 2)
                        V_hat = V_nda[1, i][j] / (1 - beta1 ** iter_num)
                        S_hat = S_nda[1, i][j] / (1 - beta2 ** iter_num)
                        dv = k * V_hat / (S_hat**0.5 + 1e-8)
                        V_nda[0, i][j] =  V_nda[1, i][j]
                        S_nda[0, i][j] =  S_nda[1, i][j]
    
                        v_input[i][j] -= dv
    
    
                        
            
            #print('total_iter:', MPC_iter_num*(max_iter-1) + iter_num -1,'MPC_iter:',MPC_iter_num,'iter: ', iter_num,'loss:', loss[None], 'loss_pos:', loss_position[None], 'los_vel:', loss_velocity[None])
            print('iter: ', iter_num,'loss:', loss[None], 'loss_pos:', loss_position[None], 'pio_coe_all: ', pio_all[None])
            #print('end effector:', int(obj.pio_x[max_step-1,0][1] * 100)/100)

    
            
            
            record_v_input(MPC_iter_num*(max_iter-1) + iter_num - 1, v_input, grad, r_v_input, r_grad)
            record_data(MPC_iter_num*(max_iter-1) + iter_num -1
                        , r_loss,r_loss_standard, r_loss_position, r_loss_velocity, r_v_input, r_grad, r_minimum_index)
            
            iter_num += 1
        
        MPC_iter_num +=1
        
    #%%


if __name__ == "__main__":
    main()

























