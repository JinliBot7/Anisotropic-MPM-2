#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 21:47:35 2023

@author: luyin
"""

import taichi as ti
import numpy as np
import os
from geometries import Obj, Hanger, Path
from math import pi
from compute_stress import compute_stress
from render_diff import set_render, render


#ti.init(debug=True)
ti.init(ti.cuda,device_memory_fraction = 0.9)
real = ti.f32
scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(3, dtype=real)


N_grid, dt = 128, 1.5e-4 # Grid number, substep time
dx, inv_dx = 1 / N_grid, N_grid
E, eta = 15e2, 0.3 # Young's modulus, Poisson's ratio
lam = E * eta / ((1 + eta) * (1 - 2 * eta)) # defined in https://encyclopediaofmath.org/wiki/Lam%C3%A9_constants
mu = E / (2 * (1 + eta))
damping = 0.0 # paper Eq. (2)

max_step = 20000
max_iter = 200

beta1 = 0.7
beta2 = 0.999

grid_v_in = vec()
grid_v_out = vec()
grid_m = scalar()
block = ti.root.pointer(ti.ijk, (int(N_grid/4),int(N_grid/4),int(N_grid/4)))
pixel = block.dense(ti.ijk, (4,4,4))
pixel.place(grid_v_in,grid_v_out, grid_m)

loss, loss_standard, loss_position, loss_velocity = scalar(), scalar(), scalar(), scalar()
x_avg = vec()
v = vec()
ti.root.place(x_avg, loss, v, loss_standard,loss_position,loss_velocity)

#Set gripper initial velocity
v_input_num = 100
v_input = vec()
ti.root.dense(ti.l, v_input_num).place(v_input)
v_input_ini_max = 0.01

# v_input = vec()
# ti.root.place(v_input)


# initialize object

center, dimension = ti.Vector([0.5, 0.6, 0.5]), ti.Vector([0.27, 0.27, 0.0005]) # dimension changes shape. if dimension[i] < e_radius, single layer
axis_angle = ti.Vector([-90 / 180 * pi, 1.0, 0.0, 0.0]) # angle and rotation axis
e_radius, total_mass, pho = 0.0019, dimension[0] * dimension[1] * dimension[2], 2.0 # e_radius is the radius of the element. The distance between two elements is designed as 2 * e_radius.
color = (0.5,0.5,0.8)
obj = Obj(center, dimension, axis_angle, e_radius, total_mass, pho, color, max_step)

dim = 3
neighbour = (3,) * dim

poi_target = vec()
ti.root.dense(ti.l, obj.poi_n).place(poi_target)



@ti.kernel
def Particle_To_Grid(t: ti.i32, obj:ti.template()):
    for p in range(obj.n):
        x_p = obj.x[t,p]
        v_p = obj.v[t,p]
        if t < max_step:
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
                v_tangent = v_out - v_normal # no friciton now
                v_out -= v_normal * boundary.sdf[p]
                v_out -= v_tangent * 0.5
                
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
                g_v = grid_v_out[base + offset] / (grid_m[base + offset] + 1e-10)

            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx * inv_dx
        
        sym_C, skew_C = (new_C + new_C.transpose()) * 0.5, (new_C - new_C.transpose()) * 0.5
        next_C = skew_C + (1 - damping) * sym_C
        
        obj.set_C(t + 1,next_C,p)
        obj.set_v(t + 1,new_v,p)
        x_p += dt * obj.v[t,p]
        obj.set_x(t + 1,x_p,p)
    
    for p in range(obj.poi_n):
        obj.poi_x[t + 1,p] = obj.x[t + 1,p % obj.poi_nx * obj.poi_scale + p // obj.poi_nx * obj.nx * obj.poi_scale]
        obj.poi_x_t[p] = obj.x[t + 1,p % obj.poi_nx * obj.poi_scale + p // obj.poi_nx * obj.nx * obj.poi_scale]
    
@ti.kernel
def Reset():
    for i, j, k in grid_m:
        grid_v_in[i, j, k] = [0.0, 0.0, 0.0]
        grid_v_out[i, j, k] = [0.0, 0.0, 0.0]
        grid_m[i, j, k] = 0.0
        
        grid_v_in.grad[i, j, k] = [0.0, 0.0, 0.0]
        grid_m.grad[i, j, k] = 0.0
        grid_v_out.grad[i, j, k] = [0.0, 0.0, 0.0]

# @ti.ad.grad_replaced
# def substep(counter, obj, hanger):
#     Reset()
#     Particle_To_Grid(counter, obj)
#     Grid_Operations(hanger)
#     Grid_To_Particle(counter, obj)


@ti.kernel
def compute_loss(obj:ti.template(), target:ti.template(), v_input: ti.template()):
    for p in range(obj.poi_n):
        dist = ti.math.length(obj.poi_x[max_step - 1, p] - target[p])
        loss[None] += dist * obj.poi_coefficient[p]
        loss_standard[None] += dist
        loss_position[None] += dist
    for p in v_input:
        for i in range(3):
                vij_up = ti.pow(1* ti.abs(v_input[p][i]),5)
                loss[None] += ti.pow(2,vij_up) - 1
                loss_velocity[None] += ti.pow(2,vij_up) - 1

def forward(total_steps,obj,hanger):
    for counter in range(total_steps - 1):
        Reset()
        Particle_To_Grid(counter, obj)
        Grid_Operations(hanger)
        Grid_To_Particle(counter, obj)
    #compute_loss(obj,poi_target, v_input)
    assign_fianl_poi(obj)



def backward(total_steps,obj,hanger):
    clear_particle_grad(obj)
    #compute_loss.grad(obj,poi_target, v_input)
    assign_fianl_poi.grad(obj)
    for counter in reversed(range(total_steps - 1)):
        # Since we do not store the grid history (to save space), we redo p2g and grid op
        Reset()
        Particle_To_Grid(counter, obj)
        Grid_Operations(hanger)
        
        Grid_To_Particle.grad(counter, obj)
        Grid_Operations.grad(hanger)
        Particle_To_Grid.grad(counter, obj)

@ti.kernel
def clear_particle_grad(obj:ti.template()):
    # for all time steps and all particles
    for f, i in obj.x:
        obj.x.grad[f, i] = zero_vec()
        obj.v.grad[f, i] = zero_vec()
        obj.C.grad[f, i] = zero_matrix()
        obj.F.grad[f, i] = zero_matrix()

@ti.func
def zero_vec():
    return [0.0, 0.0, 0.0]


@ti.func
def zero_matrix():
    return [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]

@ti.kernel
def assign_fianl_poi(obj: ti.template()):
    for p in range(obj.poi_n):
        obj.poi_final[p] = obj.poi_x_t[p]
# @ti.ad.grad_for(substep)
# def substep_grad(counter, obj, hanger):
#     Reset()
#     Particle_To_Grid(counter, obj)
#     Grid_Operations(hanger)
    
#     Grid_To_Particle.grad(counter, obj)
#     Grid_Operations.grad(hanger)
#     Particle_To_Grid.grad(counter, obj)

# @ti.kernel
# def substep_ori(obj: ti.template()):
#     for p in range(obj.n):
#         obj.x[max_step - 1, p] += v[None]

@ti.kernel
def ini_v_input(obj:ti.template(), iter_num: ti.i32, v_input_ti: ti.template()):
    for i in range(100):
        v_input[i] = v_input_ti[iter_num,i]



@ti.kernel
def compute_collision_loss(gripper_dist: ti.f32):
    loss[None] += ti.pow(2, -500*gripper_dist)
    #print(ti.pow(2, -500*gripper_dist))

def record_data(iteration, r_loss, r_loss_standard, r_loss_position, r_loss_velocity,  r_v_input, r_grad):
    r_loss[iteration] = loss[None]
    r_loss_standard[iteration] = loss_standard[None]
    r_loss_position[iteration] = loss_position[None]
    r_loss_velocity[iteration] = loss_velocity[None]
    v_input_np = r_v_input.to_numpy()
    loss_np = r_loss.to_numpy()
    loss_standard_np = r_loss_standard.to_numpy()
    loss_position_np = r_loss_position.to_numpy()
    loss_velocity_np = r_loss_velocity.to_numpy()
    grad_np = r_grad.to_numpy()
    
    np.save('./data/v_input.npy',v_input_np)
    np.save('./data/loss.npy',loss_np)
    np.save('./data/loss_standard.npy',loss_standard_np)
    np.save('./data/loss_position.npy',loss_position_np)
    np.save('./data/loss_velocity.npy',loss_velocity_np)
    np.save('./data/grad.npy',grad_np)


@ti.kernel
def record_v_input(iteration: ti.i32, v_input:ti.template(), grad: ti.template(), r_v_input: ti.template(), r_grad: ti.template()):
    for p in v_input:
        r_v_input[iteration,p] = v_input[p]
    for p in grad:
        r_grad[iteration,p] = grad[p]
        
@ti.kernel
def compute_square(grad: ti.template()) -> ti.f32: 
    grad_length = 0.0
    for i in grad:
        grad_length += ti.math.length(grad[i])
    return grad_length

def threshold(value):
    scale = 1.0
    if value > 0:
        value = min(value, 0.1)
        if value > 0.1:
            scale = value / 0.1
    else:
        value = max(value, -0.1)
        if value < -0.1:
            scale = value / -0.1
    return scale, value

@ti.kernel
def grad_to_one(obj: ti.template(),p: ti.i32,j: ti.i32):
    if j == 0:
        obj.poi_final.grad[p] = [1.0, 0.0, 0.0]
    elif j == 1:
        obj.poi_final.grad[p] = [0.0, 1.0, 0.0]
    elif j == 2:
        obj.poi_final.grad[p] = [0.0, 0.0, 1.0]

@ti.kernel
def grad_to_zero(obj: ti.template()):
    for p in obj.poi_final:
        obj.poi_final.grad[p] = [0.0, 0.0, 0.0]
    
    
def main():
    ti.root.lazy_grad()
    target_num = 30000
    poi_target_np = np.load(f'./target_path/target_{target_num}.npy')
    poi_target.from_numpy(poi_target_np)
    
    obj.initialize()
    hanger = Hanger('./Hanger',[0.5, 0.5, 0.5], (0.4, 0.4, 0.4))
    path = Path('./target_path', target_num, (0.5,0.8,0.5), obj.n, obj.poi_n, obj.nx)
    path.initialize()
    
    Circle_Center = ti.Vector.field(3, dtype = float, shape = 1)
    Circle_Center[0] = [0.45, 0.5, 0.3]
    Circle_Radius = 0.02
    camera, scene, canvas, window, axisX, axisY, axisZ, colorX, colorY, colorZ = set_render()
    floor = ti.Vector.field(3,dtype = ti.f32,shape = 4)
    floor_index = ti.field(dtype = ti.i32,shape = 6)
    floor[0], floor[1], floor[2], floor[3] = [0.0, 0.0, -1e-3], [1.0, 0.0, -1e-3], [0.0, 1.0, -1e-3], [1.0, 1.0, -1e-3]
    floor_index[0], floor_index[1],floor_index[2], floor_index[3], floor_index[4], floor_index[5] = 0, 1, 2, 3, 2, 1
    floor_color = (0.4, 0.4, 0.4)
    
    # fields for record
    r_v_input = ti.Vector.field(3, dtype = ti.f32, shape = (max_iter,v_input_num))
    r_poi_grad = ti.Vector.field(3, dtype = ti.f32, shape = (max_iter,v_input_num))
    
    
    #Visualization
    v_input_np = np.load('./data/v_input.npy')
    v_input_ti = ti.Vector.field(3, ti.f32, shape=(200,100))
    v_input_ti.from_numpy(v_input_np)
    
    iter_num = 0
    #decay_rate = 0.1
    while iter_num < 20:
        ini_v_input(obj, iter_num, v_input_ti)
         
        #Gradient part
        ti.ad.clear_all_gradients()
        forward(max_step,obj,hanger)
        #loss.grad[None] = 1
        for p in range(obj.poi_n):
            for j in range(3):
                grad_to_one(obj,p,j)
                backward(max_step,obj,hanger)
                grad = v_input.grad
                print(iter_num,p,j)
                grad_to_zero(obj)
                grad_np = grad.to_numpy()
                np.save(f'./data/grad_{iter_num}_{p}_{j}',grad_np)
                ti.ad.clear_all_gradients()
                

        
        
        iter_num += 1
        
        
    #%%


if __name__ == "__main__":
    main()

























