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


N_grid, dt = 128, 10e-5 # Grid number, substep time
dx, inv_dx = 1 / N_grid, N_grid
E, eta = 15e2, 0.3 # Young's modulus, Poisson's ratio
lam = E * eta / ((1 + eta) * (1 - 2 * eta)) # defined in https://encyclopediaofmath.org/wiki/Lam%C3%A9_constants
mu = E / (2 * (1 + eta))
damping = 0.0 # paper Eq. (2)

max_step = 20000
max_iter = 200

beta1 = 0.9
beta2 = 0.999

grid_v_in = vec()
grid_v_out = vec()
grid_m = scalar()
block = ti.root.pointer(ti.ijk, (int(N_grid/4),int(N_grid/4),int(N_grid/4)))
pixel = block.dense(ti.ijk, (4,4,4))
pixel.place(grid_v_in,grid_v_out, grid_m)

loss, loss_standard = scalar(), scalar()
x_avg = vec()
v = vec()
ti.root.place(x_avg, loss, v, loss_standard)

#Set gripper initial velocity
v_input_num = 100
v_input = vec()
ti.root.dense(ti.l, v_input_num).place(v_input)
v_input_ini_max = 0.01

# v_input = vec()
# ti.root.place(v_input)


# initialize object

center, dimension = ti.Vector([0.5, 0.6, 0.5]), ti.Vector([0.35, 0.35, 0.0005]) # dimension changes shape. if dimension[i] < e_radius, single layer
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
def substep_ori(obj: ti.template()):
    for p in range(obj.n):
        obj.x[max_step - 1, p] += v[None]

@ti.kernel
def ini_v_input(obj:ti.template(), target: ti.template()):
    ini_v = ti.Vector([0.0, 0.0, 0.0])
    for p in range(obj.pio_n):
        ini_v += (target[p] - obj.pio_x[0, p]) / (2 * obj.pio_n)
    for p in v_input:
        v_input[p] = ini_v

@ti.kernel
def compute_loss(obj:ti.template(), target:ti.template()):
    for p in range(obj.pio_n):
        dist = ti.math.length(obj.pio_x[max_step - 1, p] - target[p])
        loss[None] += dist * obj.pio_coefficient[p]
        loss_standard[None] += dist
    

def record_data(iteration, r_loss, r_loss_standard, r_v_input, r_grad):
    r_loss[iteration] = loss[None]
    r_loss_standard[iteration] = loss_standard[None]
    v_input_np = r_v_input.to_numpy()
    loss_np = r_loss.to_numpy()
    loss_standard_np = r_loss_standard.to_numpy()
    grad_np = r_grad.to_numpy()
    
    np.save('./data/v_input.npy',v_input_np)
    np.save('./data/loss.npy',loss_np)
    np.save('./data/loss_standard.npy',loss_standard_np)
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

def main():
    ti.root.lazy_grad()
    
    pio_target_np = np.load('./target_path/target_17650.npy')
    pio_target.from_numpy(pio_target_np)
    
    obj.initialize()
    hanger = Hanger('./Hanger',[0.5, 0.5, 0.5], (0.4, 0.4, 0.4))
    path = Path('./target_path', 17650, (0.5,0.8,0.5), obj.n, obj.pio_n, obj.nx)
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
    r_loss = ti.field(dtype = ti.f32, shape = max_iter)
    r_loss_standard = ti.field(dtype = ti.f32, shape = max_iter)
    r_grad = ti.Vector.field(3, dtype = ti.f32, shape = (max_iter,v_input_num))
    V_nda = ti.ndarray(dtype=ti.math.vec3, shape=(max_iter, v_input_num)) # For Adam
    S_nda = ti.ndarray(dtype=ti.math.vec3, shape=(max_iter, v_input_num)) # For Adam
    #dv_nda = ti.ndarray(dtype=ti.math.vec3, shape=(max_iter, v_input_num)) # For Adam
    
    
    #Visualization
    ini_v_input(obj,pio_target)
    iter_num = 0
    ti.profiler.clear_kernel_profiler_info()
    
    k = 0.05
    
    while iter_num < max_iter:
        
        # visualization
        if iter_num == 0 or (iter_num + 1) % 5 == 0:
            try:
                os.mkdir(f'./vis/{iter_num}')
            except FileExistsError:
                pass
                print('fiel alrady exist!')
            video_manager = ti.tools.VideoManager(f'./vis/{iter_num}',framerate = 50, automatic_build=False)
            
        
        for t in range(max_step - 1):
            substep(t, obj, hanger)
            
            if t % 100 == 0: 
                render(t,[obj], [hanger], [path], camera, scene, canvas, window, axisX, axisY, axisZ, colorX, colorY, colorZ,Circle_Center,Circle_Radius)
                scene.mesh(floor, indices = floor_index, color = floor_color)
                window.show()
                
            if iter_num == 0 or (iter_num + 1) % 5 == 0:
                if t % 100 == 0:
                    img = window.get_image_buffer_as_numpy()
                    video_manager.write_frame(img)


        video_manager.make_video(gif=False, mp4=True)        
        loss[None] = 0.0
        loss_standard[None] = 0.0
        #Gradient part
        with ti.ad.Tape(loss=loss):
            for t in range(max_step - 1):
                substep(t, obj, hanger)
            compute_loss(obj,pio_target)
        
        grad = v_input.grad
        
        
        #grad_square = compute_square(grad)
        

        for i in range(v_input_num):
            for j in range(3):
                if iter_num == 0:
                    pass
                else:
                    V_nda[iter_num, i][j] = (beta1 * V_nda[iter_num - 1, i][j] + (1 - beta1) * grad[i][j]) 
                    S_nda[iter_num, i][j] = (beta2 * S_nda[iter_num - 1, i][j] + (1 - beta2) * grad[i][j] ** 2)
                    V_hat = V_nda[iter_num, i][j] / (1 - beta1 ** iter_num)
                    S_hat = S_nda[iter_num, i][j] / (1 - beta2 ** iter_num)
                    dv = k * V_hat / (S_hat**0.5 + 1e-8)
                    # threshold
                    scale, dv = threshold(dv)
                    V_nda[iter_num, i][j] /= scale
                    v_input[i][j] -= dv

                    
        
        print('iteration: ', iter_num,'loss:', loss[None])
        #print('dv: ', v_input[0])
        #print('V: ', dv_nda[iter_num,0], 'V_t-1:', beta1 * dv_nda[iter_num -1,0], 'grad:', (1 - beta1) * grad[0])

        
        
        record_v_input(iter_num, v_input, grad, r_v_input, r_grad)
        record_data(iter_num, r_loss,r_loss_standard, r_v_input, r_grad)
        
        iter_num += 1
        
        
    #%%


if __name__ == "__main__":
    main()

























