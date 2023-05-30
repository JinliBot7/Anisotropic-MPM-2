#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import taichi as ti
from geometries import Obj, Cylinder
from render import set_render, render
from compute_stress import compute_stress
from math import pi
import time
import os
import numpy as np

#ti.init(ti.cpu, cpu_max_num_threads=1) # for debug
ti.init(ti.cuda,device_memory_fraction = 0.8,fast_math=True)

N_grid, dt = 128, 10e-5 # Grid number, substep time
dx, inv_dx = 1 / N_grid, N_grid

E, eta = 8e2, 0.3 # Young's modulus, Poisson's ratio
lam = E * eta / ((1 + eta) * (1 - 2 * eta)) # defined in https://encyclopediaofmath.org/wiki/Lam%C3%A9_constants
mu = E / (2 * (1 + eta))

damping = 0.0 # paper Eq. (2)

grid_v_in = ti.Vector.field(3, dtype=ti.f32, shape=(N_grid, N_grid, N_grid),needs_grad=True)  # grid momentum
grid_v_out = ti.Vector.field(3, dtype=ti.f32, shape=(N_grid, N_grid, N_grid),needs_grad=True)  # grid momentum
grid_m = ti.field(dtype=ti.f32, shape=(N_grid, N_grid, N_grid),needs_grad=True)  # grid mass
grid_f = ti.Vector.field(3, dtype=ti.f32, shape=(N_grid, N_grid, N_grid))  # grid force

# Object
#center, dimension = ti.Vector([0.5, 0.5, 0.4]), ti.Vector([0.25, 0.25, 0.0005]) # dimension changes shape. if dimension[i] < e_radius, single layer

center, dimension = ti.Vector([0.49, 0.55, 0.38]), ti.Vector([0.1, 0.1, 0.001]) # dimension changes shape. if dimension[i] < e_radius, single layer


axis_angle = ti.Vector([-90 / 180 * pi, 1.0, 0.0, 0.0]) # angle and rotation axis
e_radius, total_mass, pho = 0.001, dimension[0] * dimension[1] * dimension[2], 2.0 # e_radius is the radius of the element. The distance between two elements is designed as 2 * e_radius.
color = (0.5,0.5,0.8)

target = [0.5, 0.5, 0.42]
target_for_render = ti.Vector.field(3, dtype = float, shape = 1)
target_for_render[0] = target

max_step = 2000
iteration = 100

v_input_num = 100
v_input = ti.Vector.field(3, dtype=ti.f32, shape=(v_input_num), needs_grad=True)
v_input_ini_max = 0.1
@ti.kernel
def ini_v_input():
    for p in v_input:
        v_input[p] = [v_input_ini_max, -v_input_ini_max, v_input_ini_max]
ini_v_input()
 

Circle_Center = ti.Vector.field(3, dtype = float, shape = 1)
Circle_Center[0] = [0.5, 0.5, 0.4]
Circle_Radius = 0.02

obj_0 = Obj(center, dimension, axis_angle, e_radius, total_mass, pho, color, max_step)

loss = ti.field(dtype = ti.f32, shape=(), needs_grad=True)
x_avg = ti.Vector.field(3, dtype = ti.f32, shape=(), needs_grad=True)
z_abs = ti.field(dtype = ti.f32, shape=(), needs_grad=True)

# # Static Boundary Cynlnder
# center, quaternion, radius, height, friciton, resolution, e_radius = ti.Vector([0.5, 0.5, 0.3]), ti.Vector([0.0, 1.0, 0.0, 0.0]), 0.02, 0.02, 0.3, 32, 0.001 # friction and speed are not used yet
# speed = ti.Vector([0.0, 0.0, 0.0])
# blc_0 = Cylinder(center, speed, quaternion, radius, height, friciton, resolution, e_radius, (0.5, 0.5, 0.5),dx)

obj_list = [obj_0]
boundary_list = []

# fields for record
r_v_input = ti.Vector.field(3, dtype = ti.f32, shape = (iteration,v_input_num))
r_loss = ti.field(dtype = ti.f32, shape = iteration)
r_grad = ti.Vector.field(3, dtype = ti.f32, shape = (iteration,v_input_num))
r_x_avg = ti.Vector.field(3, dtype = ti.f32, shape = iteration)


#%%
dim = 3
neighbour = (3,) * dim


@ti.kernel
def Particle_To_Grid(t: ti.i32, obj:ti.template()):
    for p in range(obj.n):
        x_p = obj.x[t,p]
        v_p = ti.Vector([0.0, 0.0, 0.0])
        if p < 51 * 3 and int(t/max_step * v_input_num) < v_input_num * 0.8:
            v_p = v_input[int(t/max_step * v_input_num)]
        elif p < 51 * 3 and int(t/max_step * v_input_num) >= v_input_num * 0.8:
            v_p = [0.0, 0.0, 0.0]
        else:
            v_p = obj.v[t,p]

        C_p = obj.C[t,p]
        F_p = obj.F[t,p]
        
        m = obj.m
        vol = obj.vol
        
        #Deformation update
        F_p += dt * C_p @ F_p
        
        base = (x_p * inv_dx - 0.5).cast(int)  # x2[p]/dx transfer world coordinate to mesh coordinate
        fx = x_p * inv_dx - base.cast(float)  # fx is displacement vector
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2,]  # quadratic b spline
        
        # Anisotropic
        P, F_elastic, F_plastic = compute_stress(F_p,mu,lam)
        obj.set_F(t + 1,F_elastic,p)
        dF_dC = F_elastic.transpose() 
        #dF_dC = F_plastic.inverse().transpose() @ F_p_ori.transpose() # identical to above equation
        
        # # Neo-Hookean
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

            grid_m[base + offset] += weight * m  # mass conservation
            grid_v_in[base + offset] += weight * m * v_p + weight * m * C_p @ dpos # grid_v is acatually momentum, not velocity
            grid_v_in[base + offset] -= weight * 4 * dt * inv_dx * inv_dx * vol * P @ dF_dC @ dpos
            grid_v_in[base + offset].z += weight * dt * m * -9.8  


bound = 5
@ti.kernel
def Grid_Operations():
    for i, j, k in grid_v_in:
        v_out = grid_v_in[i, j, k]
        if k < bound and v_out.z < 0:
            v_out.z *= 0
            v_out.x *= 0.1
            v_out.y *= 0.1
        
        dist = ti.Vector([i * dx, j * dx, k * dx]) - Circle_Center[0]
        if dist.x ** 2 + dist.y ** 2 + dist.z ** 2 < Circle_Radius * Circle_Radius :
            dist = dist.normalized()
            v_out -= dist * ti.min(0, grid_v_in[i, j, k].dot(dist))
            v_out *= 0.9  #friction
        grid_v_out[i, j, k] = v_out
# @ti.kernel
# def Grid_Operations_Manipulator(boundary:ti.template()):
#     for p in boundary.collision_box:
#         grid_index = int(boundary.collision_box[p] / dx)

#         #speed = boundary.speed[0]
#         if grid_m[grid_index] != 0:
#             v_proj = grid_v[grid_index].dot(boundary.cf_direction[p])            
#             #v_gripper_proj = grid_v[grid_index].dot(boundary.speed_normal[0])
#             if v_proj < 0:
#                 v_normal = v_proj * boundary.cf_direction[p]
#                 #v_tangent = grid_v[grid_index] - v_normal # no friciton now
#                 grid_v[grid_index] -= v_normal * boundary.cf_coefficient[p]
#             # if v_gripper_proj < 0:
#             #     # TO DO: change speed to speed * v_proj, special case when speed_normal[0] is [0,0,0] should be handled
#             #     v_normal_gripper = v_gripper_proj * boundary.cf_direction[p]
#             #     grid_v[grid_index] +=  v_normal_gripper * boundary.cf_coefficient[p]

                
d_threshold =  0.3 * dx        

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
                g_v = grid_v_out[base + offset] / (grid_m[base + offset] + 1e-12)

            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx * inv_dx
        
        sym_C, skew_C = (new_C + new_C.transpose()) * 0.5, (new_C - new_C.transpose()) * 0.5
        next_C = skew_C + (1 - damping) * sym_C
        obj.set_C(t + 1,next_C,p)
        obj.set_v(t + 1,new_v,p)
        x_p += dt * obj.v[t,p]
        obj.set_x(t + 1,x_p,p)
    
        # if ti.math.length(dt * obj.v[p]) < d_threshold:
        #     x_p += dt * obj.v[p]
        # else:
        #     x_p += d_threshold * ti.math.normalize(obj.v[p])
        # obj.set_x(x_p,p)

@ti.kernel
def Reset():
    for i, j, k in grid_m:
        grid_v_in[i, j, k] = [0.0, 0.0, 0.0]
        grid_v_out[i, j, k] = [0.0, 0.0, 0.0]
        grid_m[i, j, k] = 0.0
        grid_f[i, j, k] = [0.0, 0.0, 0.0]
        
        grid_v_in.grad[i, j, k] = [0.0, 0.0, 0.0]
        grid_m.grad[i, j, k] = 0.0
        grid_v_out.grad[i, j, k] = [0.0, 0.0, 0.0]

@ti.func
def to_quaternion(arg):
    sin_theta = ti.math.sin(arg[0]/2)
    return ti.Vector([ti.math.cos(arg[0]/2),sin_theta * arg[1], sin_theta * arg[2], sin_theta * arg[3]])

@ti.kernel
def rotate_z(step_counter:ti.i32):
    obj_0.rotate(to_quaternion([1e-2, 0.0, 0.0, 1.0]))

@ti.kernel
def compute_x_avg(n_particles: ti.i32):

    for i in range(n_particles):
        #x_avg[None].atomic_add((1 / n_particles) * obj_0.x[max_step - 1, i])
        x_avg[None] += (1 / n_particles) * obj_0.x[max_step - 1, i]
        z_abs[None] += (1 / n_particles) * (obj_0.x[max_step - 1, i][2] - target[2]) ** 2


@ti.kernel
def compute_loss():
    dist = (x_avg[None] - ti.Vector(target)) ** 2 
    
    loss[None] =  0.5 * (dist[0] + dist[1] + z_abs[None] * 3)
    print('xy loss component: ', 0.5 * 1e5 * (dist[0] + dist[1]))
    print('z_abs loss component: ',  0.5 * 1e5 * z_abs[None] * 3)
    
    

@ti.ad.grad_replaced
def substep(step_counter):
    Reset()
    
    Particle_To_Grid(step_counter,obj_0)

    Grid_Operations()

    Grid_To_Particle(step_counter,obj_0)


@ti.ad.grad_for(substep)
def substep_grad(step_counter):
    Reset()

    Particle_To_Grid(step_counter,obj_0)
    Grid_Operations()

    Grid_To_Particle.grad(step_counter, obj_0)
    Grid_Operations.grad()
    Particle_To_Grid.grad(step_counter, obj_0)


@ti.kernel
def record_v_input(iteration: ti.i32, v_input:ti.template(), grad: ti.template()):
    for p in v_input:
        r_v_input[iteration,p] = v_input[p]
    for p in grad:
        r_grad[iteration,p] = grad[p]

def record_data(iteration,loss,x_avg,learning_scale):
    r_loss[iteration] = loss
    r_x_avg[iteration] = x_avg
    
    v_input_np = r_v_input.to_numpy()
    loss_np = r_loss.to_numpy()
    grad_np = r_grad.to_numpy()
    x_avg_np = r_x_avg.to_numpy()
    
    np.save('./data/v_input_{learning_scale:d}.npy'.format(learning_scale = learning_scale),v_input_np)
    np.save('./data/loss_{learning_scale:d}.npy'.format(learning_scale = learning_scale),loss_np)
    np.save('./data/grad_{learning_scale:d}.npy'.format(learning_scale = learning_scale),grad_np)
    np.save('./data/x_avg_{learning_scale:d}.npy'.format(learning_scale = learning_scale),x_avg_np)
    

def main():
    camera, scene, canvas, window, axisX, axisY, axisZ, colorX, colorY, colorZ = set_render()
    floor = ti.Vector.field(3,dtype = ti.f32,shape = 4)
    floor_index = ti.field(dtype = ti.i32,shape = 6)
    floor[0], floor[1], floor[2], floor[3] = [0.0, 0.0, -1e-3], [1.0, 0.0, -1e-3], [0.0, 1.0, -1e-3], [1.0, 1.0, -1e-3]
    floor_index[0], floor_index[1],floor_index[2], floor_index[3], floor_index[4], floor_index[5] = 0, 1, 2, 3, 2, 1
    floor_color = (0.6, 0.6, 0.6)
    
    for learning_scale in range(3,10,1):
        for iter_counter in range(iteration):
            with ti.ad.Tape(loss=loss):
                for step_counter in range(max_step - 1):
                    substep(step_counter)
                    
                loss[None] = 0.0
                z_abs[None] = 0.0
                x_avg[None] = [0.0, 0.0, 0.0]
                compute_x_avg(obj_0.n)
                #print("x_avg: ",x_avg[None])
                compute_loss()
        
            l = loss[None]
            grad = v_input.grad
            print('learning_scale: ', learning_scale)
            print('iteration: ', iter_counter)
            print('loss=', l * 1e5)
            #print('grad=', grad[0])
            
            record_v_input(iter_counter, v_input, grad)
            record_data(iter_counter,loss[None],x_avg[None],learning_scale)
            
            learning_rate = 1e2 * learning_scale
            
            for i in range(v_input_num):
                for j in range(3):
                    if learning_rate * grad[i][j] >=0:
                        v_input[i][j] -= ti.min(learning_rate * grad[i][j], 0.1)
                    elif learning_rate * grad[i][j] <0:
                        v_input[i][j] -= ti.max(learning_rate * grad[i][j], -0.1)
                
                
            #print('v_input[0]=', v_input[0])
            
            # visualization
            if iter_counter == 0 or (iter_counter + 1) % 5 == 0:
                try:
                    os.mkdir('./vis/{learning_scale:d}/{iter_counter:d}'.format(learning_scale = learning_scale, iter_counter = iter_counter))
                except FileExistsError:
                    pass
                video_manager = ti.tools.VideoManager('./vis/{learning_scale:d}/{iter_counter:d}'.format(learning_scale = learning_scale, iter_counter = iter_counter),framerate=16,automatic_build=False)
    
            
            for step_counter in range(max_step):
                # if window.get_event(ti.ui.PRESS):
                #     if window.event.key == ti.GUI.ESCAPE:
                #         window.destroy()
                substep(step_counter)
                render(obj_list, boundary_list, camera, scene, canvas, window, axisX, axisY, axisZ, colorX, colorY, colorZ,Circle_Center,Circle_Radius, target_for_render)
                scene.mesh(floor, indices = floor_index, color = floor_color)
                
                if iter_counter == 0 or (iter_counter + 1) % 5 == 0:
                    if step_counter % 20 == 0:
                        img = window.get_image_buffer_as_numpy()
                        video_manager.write_frame(img)
                        print('iter: ', iter_counter, 'step: ', step_counter)
                window.show()
            
            if iter_counter == 0 or (iter_counter + 1) % 5 == 0:
                video_manager.make_video(gif=True, mp4=True)
        
        # loss[None] = 0.0
        # x_avg[None] = [0.0, 0.0, 0.0]
        # compute_loss()
        # compute_x_avg(obj_0.n)
        # print(loss[None])
        # print(x_avg[None])
            
        


if __name__ == "__main__":
    main()




























