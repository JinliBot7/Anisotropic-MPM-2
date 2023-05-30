#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:06:51 2022

@author: luyin
"""

import taichi as ti
from geometries import Obj, Cylinder
from render import set_render, render
from compute_stress_113 import compute_stress
from math import pi, cos, sin
#ti.init(ti.cpu, cpu_max_num_threads=1)
#arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(ti.cuda,device_memory_fraction=0.9,fast_math=False)

N_grid, dt = 256, 5e-5
dx, inv_dx = 1 / N_grid, N_grid

E, eta = 10e2, 0.3
lam = E * eta / ((1 + eta) * (1 - 2 * eta)) 
mu = E / (2 * (1 + eta))

damping = 0.0

grid_v = ti.Vector.field(3, dtype=ti.f32, shape=(N_grid, N_grid, N_grid))  # grid velocity
grid_m = ti.field(dtype=ti.f32, shape=(N_grid, N_grid, N_grid))  # grid mass
grid_f = ti.Vector.field(3, dtype=ti.f32, shape=(N_grid, N_grid, N_grid))  # grid force

# Object
center, dimension = ti.Vector([0.5, 0.5, 0.4]), ti.Vector([0.25, 0.25, 0.0005])
axis_angle = ti.Vector([0 / 180 * pi, 1.0, 0.0, 0.0])
e_radius, total_mass, pho = 0.001, dimension[0] * dimension[1] * dimension[2], 2.0
color = (0.5,0.5,0.8)
obj_0 = Obj(center, dimension, axis_angle, e_radius, total_mass, pho, color)

#print(obj_0.quaternion)

# # Static Boundary
# center, quaternion, radius, height, friciton, resolution, e_radius = ti.Vector([0.45, 0.5, 0.3]), ti.Vector([0.0, 1.0, 0.0, 0.0]), 0.02, 0.02, 0.3, 32, 0.001
# speed = ti.Vector([0.0, 0.0, 0.0])
# blc_0 = Cylinder(center, speed, quaternion, radius, height, friciton, resolution, e_radius, (0.5, 0.5, 0.5),dx)


obj_list = [obj_0]
boundary_list = []


#%%

dim = 3
neighbour = (3,) * dim

@ti.kernel
def Particle_To_Grid(obj:ti.template()):
    m = obj.m
    vol = obj.vol
    for p in obj.x:
        x_p = obj.x[p]
        v_p = obj.v[p]
        C_p = obj.C[p]
        F_p = obj.F[p]
        F_T = F_p.transpose()
        #Deformation update
        F_p += dt * C_p @ F_p
        obj.set_F(F_p,p)

        
        # Momentum and mass update
        base = (x_p * inv_dx - 0.5).cast(int)  # x2[p]/dx transfer world coordinate to mesh coordinate
        fx = x_p * inv_dx - base.cast(float)  # fx is displacement vector
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2,]  # quadratic b spline

        P, F_elastic, F_plastic = compute_stress(F_p,mu,lam)
        #P = P @ F_plastic.transpose().inverse()


               
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]

            grid_m[base + offset] += weight * m  # mass conservation
            grid_v[base + offset] += weight * m * v_p + weight * m * C_p @ dpos # grid_v is acatually momentum, not velocity
            grid_v[base + offset] -= weight * 4 * dt * inv_dx * inv_dx * vol * P @ F_T @ dpos
            grid_v[base + offset].z += weight * dt * m * -9.8  



Circle_Center = ti.Vector.field(3, dtype = float, shape = 1)
Circle_Center[0] = [0.45, 0.5, 0.3]
Circle_Radius = 0.02
bound = 5
@ti.kernel
def Grid_Operations():
    for i, j, k in grid_v:
        
        if k < bound and grid_v[i, j, k].z < 0:
            grid_v[i, j, k].z *= -0.1
            grid_v[i, j, k].x *= 0.1
            grid_v[i, j, k].y *= 0.1
        
        # dist = ti.Vector([i * dx, j * dx, k * dx]) - Circle_Center[0]
        # if dist.x**2 + dist.y**2 + dist.z**2< Circle_Radius* Circle_Radius :
        #     dist = dist.normalized()
        #     grid_v[i, j, k] -= dist * min(0, grid_v[i, j, k].dot(dist) )
        #     #grid_v[i, j, k] *= 0.9  #friction
        
@ti.kernel
def Grid_Operations_Manipulator(boundary:ti.template()):
    for p in boundary.collision_box:
        grid_index = int(boundary.collision_box[p] / dx)

        #speed = boundary.speed[0]
        if grid_m[grid_index] != 0:
            v_proj = grid_v[grid_index].dot(boundary.cf_direction[p])
            
            v_gripper_proj = grid_v[grid_index].dot(boundary.speed_normal[0])
            if v_proj < 0:
                v_normal = v_proj * boundary.cf_direction[p]
                v_tangent = grid_v[grid_index] - v_normal
                grid_v[grid_index] -= v_normal * boundary.cf_coefficient[p]
            if v_gripper_proj <= 0:
                # TO DO: change speed to speed * v_proj, special case when speed_normal[0] is [0,0,0] should be handled
                v_normal_gripper = v_gripper_proj * boundary.cf_direction[p]
                grid_v[grid_index] +=  v_normal_gripper * boundary.cf_coefficient[p]

                
d_threshold =  0.3 * dx        

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
        
        sym_C = (new_C + new_C.transpose()) * 0.5
        skew_C = (new_C - new_C.transpose()) * 0.5
        next_C = skew_C + (1 - damping) * sym_C
        obj.set_C(next_C,p)
        obj.set_v(new_v,p)
        x_p += dt * obj.v[p]
        obj.set_x(x_p,p)
        # if ti.math.length(dt * obj.v[p]) < d_threshold:
        #     x_p += dt * obj.v[p]
        # else:
        #     x_p += d_threshold * ti.math.normalize(obj.v[p])
        # obj.set_x(x_p,p)

@ti.kernel    
def Update_boundary(obj:ti.template()):
    dis = obj.speed[0] * dt
    obj.translate(dis)     

@ti.kernel 
def Update_gripper(obj:ti.template(),x:ti.f32, y:ti.f32, z:ti.f32):
    obj.update_speed(x,y,z)       
        


@ti.kernel
def Reset():
    for i, j, k in grid_m:
        grid_v[i, j, k] = [0.0, 0.0, 0.0]
        grid_m[i, j, k] = 0.0
        grid_f[i, j, k] = [0.0, 0.0, 0.0]

@ti.func
def to_quaternion(arg):
    sin_theta = ti.math.sin(arg[0]/2)
    return ti.Vector([ti.math.cos(arg[0]/2),sin_theta * arg[1], sin_theta * arg[2], sin_theta * arg[3]])

@ti.kernel
def rotate_z(counter:ti.i32):
    obj_0.rotate(to_quaternion([1e-2, 0.0, 0.0, 1.0]))

@ti.kernel
def rotate_x(counter:ti.i32):
    obj_0.rotate(to_quaternion([1e-2, 1.0, 0.0, 0.0]))


    #print(counter)
    
# def quat_to_axis_angle(quat):
    

def main():
    counter = 0
    camera, scene, canvas, window, axisX, axisY, axisZ, colorX, colorY, colorZ = set_render()
    floor = ti.Vector.field(3,dtype = ti.f32,shape = 4)
    floor_index = ti.field(dtype = ti.i32,shape = 6)
    floor[0], floor[1], floor[2], floor[3] = [0.0, 0.0, -1e-3], [1.0, 0.0, -1e-3], [0.0, 1.0, -1e-3], [1.0, 1.0, -1e-3]
    floor_index[0], floor_index[1],floor_index[2], floor_index[3], floor_index[4], floor_index[5] = 0, 1, 2, 3, 2, 1
    floor_color = (0.6, 0.6, 0.6)
    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.GUI.ESCAPE:
                window.destroy()
        Reset()
        
        for obj in obj_list:
            Particle_To_Grid(obj)
        
        for boundary in boundary_list:
            Grid_Operations_Manipulator(boundary)
            Update_boundary(boundary)
        
        Grid_Operations()
        
        for obj in obj_list:
            Grid_To_Particle(obj)
        
        # counter += 1
        # if counter % 300  == 0:
        #     print(obj_list[0].F[1000])
            
    
        
        #print(obj_0.quaternion[0])
        render(obj_list, boundary_list, camera, scene, canvas, window, axisX, axisY, axisZ, colorX, colorY, colorZ,Circle_Center,Circle_Radius)
        scene.mesh(floor, indices = floor_index, color = floor_color)
        
        window.show()



if __name__ == "__main__":
    main()




























