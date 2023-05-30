#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 15:12:36 2022

@author: luyin
"""

import numpy as np
import taichi as ti
from taichi.math import cos, sin, pi



@ti.data_oriented
class Obj:
    def __init__(self, center, dimension, axis_angle: ti.types.vector(4,ti.f32), e_radius, mass, pho, color):
        self.center = ti.Vector.field(3, dtype=ti.f32, shape=1)
        self.center[0] = center
        self.nx, self.ny, self.nz = int(dimension[0] / (2 * e_radius)) + 1, int(dimension[1] / (2 * e_radius)) + 1, int(dimension[2] / (2 * e_radius)) + 1
        self.e_radius = e_radius
        self.n = int(self.nx * self.ny * self.nz) # total particle number
        self.m = mass / self.n
        self.vol = self.m / pho
        self.color = color
        self.displacment = ti.Vector.field(3, dtype=ti.f32, shape = 1)
        self.displacment[0] = center
        self.quaternion = ti.Vector.field(4, dtype = ti.f32, shape = 1)
        self.quaternion[0] = self.to_quaternion([0.0, 0.0, 0.0, 1.0])
        self.axis_angle = axis_angle

        # Taichi field
        self.x = ti.Vector.field(3, dtype=ti.f32, shape=self.n) # position field
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=self.n) # velocity field
        self.C = ti.Matrix.field(3, 3, dtype=float, shape=self.n)  # affine velocity field
        self.F = ti.Matrix.field(3, 3, dtype=float, shape=self.n)  # deformation gradient
        self.colors = ti.Vector.field(3, dtype=float, shape=self.n)
        
        triangles = (self.nx - 1) * (self.ny - 1) * 2
        self.rectangles = int(triangles / 2)
        indices = triangles * 3
        self.mesh_indices = ti.field(dtype = ti.i32, shape = indices)
        
        self.initialize()


    
    @ti.kernel
    def initialize(self): # initialize particle positions
        start_x = self.center[0][0] - (self.nx - 1) * 2 * self.e_radius / 2
        start_y = self.center[0][1] - (self.ny - 1) * 2 * self.e_radius / 2
        start_z = self.center[0][2] - (self.nz - 1) * 2 * self.e_radius / 2
        
        for p in self.x:
            layer_z = int((p) / (self.nx * self.ny))
            colum_y = int((p - layer_z * self.nx * self.ny) / (self.nx))
            row_x = (p  - layer_z * self.nx * self.ny - colum_y * self.nx)
            self.x[p] = [start_x + row_x * 2 * self.e_radius, start_y + colum_y * 2 * self.e_radius, start_z + layer_z * 2 * self.e_radius]
            self.F[p] = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        self.rotate(self.to_quaternion_func(self.axis_angle))

        
        for p in range(self.rectangles):
            N_rect = p
            N_colum = int(N_rect / (self.nx -1))
            N_row = N_rect - N_colum * (self.nx - 1)
            self.mesh_indices[p * 6]     = N_row + N_colum * (self.nx) 
            self.mesh_indices[p * 6 + 1] = N_row + 1 + N_colum * (self.nx ) 
            self.mesh_indices[p * 6 + 2] = N_row + (N_colum + 1) * (self.nx ) 
            self.mesh_indices[p * 6 + 3] = N_row + 1 + (N_colum + 1) * (self.nx ) 
            self.mesh_indices[p * 6 + 4] = N_row + (N_colum + 1) * (self.nx )
            self.mesh_indices[p * 6 + 5] = N_row + 1 + N_colum * (self.nx ) 
    # coordinate in particle to index in position filed. e.g., row 2 , col 3, layer 1 correspond to p'th partcicle in p
    # def coordinate_2_index(self,cor):
    #     p = int(cor[2] * self.nx * self.ny + cor[1] * self.nx + cor[0])
    #     return p
    
    @ti.func
    def translate(self, dis: ti.types.vector(3,ti.f32)):
        for p in self.x:
            self.x[p] += dis
        self.displacment[0] = dis
    
    def to_quaternion(self,arg):
        sin_theta = sin(arg[0]/2)
        return ti.Vector([cos(arg[0]/2),sin_theta * arg[1], sin_theta * arg[2], sin_theta * arg[3]])
    
    @ti.func
    def to_quaternion_func(self,arg):
        sin_theta = sin(arg[0]/2)
        return ti.Vector([cos(arg[0]/2),sin_theta * arg[1], sin_theta * arg[2], sin_theta * arg[3]])
    
    
        
    @ti.func
    def quat_mul(self, q1, q2):# quaternion multiplication from wiki
        a1, b1, c1, d1 = q1[0], q1[1], q1[2], q1[3]
        a2, b2, c2, d2 = q2[0], q2[1], q2[2], q2[3]
        t1 = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
        t2 = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
        t3 = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
        t4 = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2
        result = [t1, t2, t3, t4]
        return result

    @ti.func
    def rotate(self, quaternion: ti.types.vector(4,ti.f32)): # quaternion rotation from wiki
        #new_quaternion = self.quat_mul(quaternion, self.quaternion[0])
        new_quaternion = self.quat_mul(self.quaternion[0],quaternion)
        self.quaternion[0] = new_quaternion
        # theta = quaternion[0]
        # axis = [quaternion[1], quaternion[2], quaternion[3]]
        for p in self.x:
            complex_p = [0.0, self.x[p][0] - self.displacment[0][0], self.x[p][1] - self.displacment[0][1], self.x[p][2] - self.displacment[0][2]]
            #complex_q = [ti.math.cos(theta/2), axis[0] * ti.math.sin(theta/2), axis[1] * ti.math.sin(theta/2), axis[2] * ti.math.sin(theta/2)]
            #complex_q_inv = [ti.math.cos(theta/2), -axis[0] * ti.math.sin(theta/2), -axis[1] * ti.math.sin(theta/2), -axis[2] * ti.math.sin(theta/2)]
            complex_q = quaternion
            complex_q_inv = [quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]]
            p_prime = self.quat_mul(self.quat_mul(complex_q, complex_p),complex_q_inv)
            self.x[p][0], self.x[p][1], self.x[p][2] = p_prime[1] + self.displacment[0][0], p_prime[2] + self.displacment[0][1], p_prime[3] + self.displacment[0][2]
    
    @ti.func
    def set_x(self, tar: ti.types.vector(3,ti.f32),p):
        for I in ti.static(range(3)):
            self.x[p][I] = tar[I]
    
    @ti.func
    def set_v(self, tar: ti.types.vector(3,ti.f32),p):
        for I in ti.static(range(3)):
            self.v[p][I] = tar[I]
    
    @ti.func
    def set_C(self, tar: ti.types.matrix(3,3,ti.f32),p):
        for I in ti.static(ti.grouped(ti.ndrange(3, 3))):
            self.C[p][I] = tar[I]
    
    @ti.func
    def set_F(self, tar: ti.types.matrix(3,3,ti.f32),p):
        for I in ti.static(ti.grouped(ti.ndrange(3, 3))):
            self.F[p][I] = tar[I]
    
    # def save_array(self):
    #     array = self.mesh_indices.to_numpy()
    #     np.save('mesh_indices.npy', array)
    #     array = self.x_record.to_numpy()
    #     np.save('x_record.npy', array)

@ti.data_oriented
class Cylinder():
    def __init__(self, center, speed, quaternion, radius, height, friction, resolution, e_radius, color, dx):
        self.center = ti.Vector.field(3, dtype=ti.f32, shape=1)
        self.center[0] = center
        
        self.radius = radius
        self.height = height
        self.friction = friction
        self.resolution = resolution
        self.e_radius = e_radius
        self.color = color
        self.dx = dx
        
        self.x = ti.Vector.field(3, dtype=ti.f32, shape= self.resolution * 2 + 2) # position field
        self.triangles = resolution * 4
        self.indices = ti.field(int, shape = self.triangles * 3)
        self.vertices_color = ti.Vector.field(3, dtype=ti.f32, shape= self.resolution * 2 + 2)
        
        self.displacment = ti.Vector.field(3, dtype=ti.f32, shape = 1)
        self.displacment[0] = center
        self.speed = ti.Vector.field(3, dtype=ti.f32, shape = 1)
        self.speed[0] = speed
        self.speed_normal = ti.Vector.field(3, dtype=ti.f32, shape = 1)
        self.quaternion = ti.Vector.field(4, dtype = ti.f32, shape = 1)
        
        # collision box 
        scale = 1.5
        self.box_height = int(ti.math.round(scale * self.height / self.dx )) + 1
        self.box_length = int(ti.math.round(scale * self.radius * 2 / self.dx )) + 1
        max_length = max(self.box_length, self.box_height)
        self.box_height = max_length
        self.box_length = max_length
        self.collision_box = ti.Vector.field(3, dtype = ti.f32, shape = self.box_length ** 2 * self.box_height)
        
        # only for test
        self.offset = ti.Vector.field(3, dtype = ti.f32, shape = self.box_length ** 2 * self.box_height)
        self.sdf_filed = ti.field(dtype = ti.f32, shape = self.box_length ** 2 * self.box_height)
        
        # collision force coefficient and dirction
        self.cf_coefficient = ti.field(dtype = ti.f32, shape = self.box_length ** 2 * self.box_height)
        self.cf_direction = ti.Vector.field(3, dtype = ti.f32, shape = self.box_length ** 2 * self.box_height)
        
        # for visualization
        self.vis_vertices = ti.Vector.field(3, dtype = ti.f32, shape = self.box_length ** 2 * self.box_height * 2)
        self.vis_vertices_indices = ti.field(dtype = ti.i32, shape = self.box_length ** 2 * self.box_height * 2)
        
        self.initializie()
        
    @ti.kernel
    def initializie(self):
        # ini_x() and ini_indices() are only for visualization
        self.ini_x()
        self.ini_indices()
        self.sdf_init()
        self.collision_box_update()
        
    
    @ti.func
    def ini_x(self):
        for p in self.x:
            angle = p/self.resolution * 2 * pi
            self.x[p] = [cos(angle) * self.radius + self.center[0][0], sin(angle) * self.radius + self.center[0][1], self.center[0][2] + int(p / self.resolution) * self.height - 0.5 * self.height]
            self.vertices_color[p] = self.color
        self.x[self.resolution * 2] = [self.center[0][0], self.center[0][1], self.center[0][2] - 0.5 * self.height]
        self.x[self.resolution * 2 + 1] = [self.center[0][0], self.center[0][1], self.center[0][2]+ 0.5 * self.height]
        
        normal_speed = ti.math.normalize(self.speed[0])
        if ti.math.length(self.speed[0]) < 1e-6:
            normal_speed[0], normal_speed[1], normal_speed[2] = 0.0, 0.0, 0.0
        self.speed_normal[0][0], self.speed_normal[0][1], self.speed_normal[0][2] = normal_speed[0],normal_speed[1],normal_speed[2]
    
    @ti.func
    def ini_indices(self):
        for i in range(self.resolution ):
            # lower_surface
            self.indices[i * 3] = i
            self.indices[i * 3 + 1] = i + 1
            self.indices[i * 3 + 2] = self.resolution * 2
            # upper surface
            self.indices[self.resolution * 3 + i * 3] = self.resolution + i 
            self.indices[self.resolution * 3 + i * 3 + 1] = self.resolution + i + 1
            self.indices[self.resolution * 3 + i * 3 + 2] = self.resolution * 2 + 1
            # side
            self.indices[self.resolution * 6 + i * 6] = i
            self.indices[self.resolution * 6 + i * 6 + 1] = self.resolution + i 
            self.indices[self.resolution * 6 + i * 6 + 2] = self.resolution + i + 1
            self.indices[self.resolution * 6 + i * 6 + 3] = i
            self.indices[self.resolution * 6 + i * 6 + 4] = i + 1
            self.indices[self.resolution * 6 + i * 6 + 5] = self.resolution + i + 1
      
        self.indices[(self.resolution - 1) * 3 + 1] = 0
        self.indices[self.resolution * 3 + (self.resolution - 1) * 3 + 1] = self.resolution 
        
        self.indices[self.resolution * 6 + (self.resolution - 1) * 6 + 2] = self.resolution
        self.indices[self.resolution * 6 + (self.resolution - 1) * 6 + 4] = 0
        self.indices[self.resolution * 6 + (self.resolution - 1) * 6 + 5] = self.resolution
    
    # initialize collision box and collision force direciton
    @ti.func
    def sdf_init(self):
        box_length = self.box_length 
        for i in range(box_length ** 2 * self.box_height):
            layer_z = int(i / box_length ** 2)
            col_y = int((i - layer_z * box_length ** 2) / box_length)
            row_x = int(i - layer_z * box_length ** 2 - col_y * box_length)
            offset = ti.Vector([int(row_x - 0.5 * (self.box_length - 1)) , int(col_y - 0.5 * (self.box_length - 1)), int(layer_z - 0.5 * (self.box_height - 1))])
            self.collision_box[i] = (offset) * self.dx
            self.offset[i] = offset
        
        # collision coefficient and force dirction
        for p in self.collision_box:
            # take the absolute value here since cylinder sdf is symmetric
            sdf = self.get_sdf(abs(self.collision_box[p]))
            # 50 is emperical
            self.cf_coefficient[p] = ti.math.exp(-50 * sdf)
            self.cf_direction[p] = self.cf_d(self.collision_box[p], p)
            self.sdf_filed[p] = sdf
            
            offset = self.offset[p]
            dis = (offset[0] ** 2 + offset[1] ** 2 + offset[2] ** 2) ** 0.5
            if dis > self.box_length / 2:
                self.sdf_filed[p] = 1

            
            
    # distance function
    @ti.func
    def get_sdf(self, grid_p):
        x, y, z = grid_p[0], grid_p[1], grid_p[2]
        
        alpha, beta = 1.0, 0.0

        if x ** 2 + y ** 2 < self.radius ** 2 and z > 0.5 * self.height:
            alpha, beta = 0.0, 1.0

        if x ** 2 + y ** 2 > self.radius ** 2 and z > 0.5 * self.height:
            alpha, beta = 100.0, 100.0
            
        if x ** 2 + y ** 2 > self.radius ** 2 and z < 0.5 * self.height:
            alpha, beta = 1.0, 0.0

        sdf = (alpha * (((x ** 2 + y ** 2) ** 0.5 - self.radius)) **2 + beta * (z - 0.5 * self.height) ** 2 ) ** 0.5
        
        if x ** 2 + y ** 2 < self.radius ** 2 and z < 0.5 * self.height:
            sdf = 1e-10
        

        
        return sdf
    
    # Collision force normal dirction
    @ti.func
    def cf_d(self, grid_p,p):

        x, y, z = grid_p[0], grid_p[1], grid_p[2]
        
        angle = ti.math.atan2(y,x)
        xy_length = abs(((x ** 2 + y ** 2) ** 0.5 - self.radius))
        x_d = ti.math.cos(angle) * xy_length
        y_d = ti.math.sin(angle) * xy_length

        cf_direction = ti.math.normalize(ti.Vector([x_d , y_d, 0]))
        
        if x ** 2 + y ** 2 < self.radius ** 2 and z > 0.5 * self.height:
            cf_direction = ti.Vector([0.0, 0.0, 1.0])
        if x ** 2 + y ** 2 < self.radius ** 2 and z < -0.5 * self.height:
            cf_direction = ti.Vector([0.0, 0.0, -1.0])
        if x ** 2 + y ** 2 > self.radius ** 2 and z > 0.5 * self.height:
            z_d = z - 0.5 * self.height
            cf_direction = ti.math.normalize(ti.Vector([x_d, y_d, z_d]))
        if x ** 2 + y ** 2 > self.radius ** 2 and z < -0.5 * self.height:
            z_d = z + 0.5 * self.height
            cf_direction = ti.math.normalize(ti.Vector([x_d , y_d, z_d]))

        return cf_direction
    
    
    @ti.func
    def collision_box_update(self):
        for p in self.collision_box:
            self.collision_box[p] += self.displacment[0]            
            

    @ti.func
    def translate(self, dis):
        
        if self.center[0][2] < self.height / 2 * 2 + 2 * self.dx and dis[2]<0:
            dis[2] = 0.0


        
        for p in self.x:
            self.x[p] += dis
        self.displacment[0] = dis
        self.center[0] += dis
        self.collision_box_update()
    
    @ti.func
    def update_speed(self,x,y,z):
        self.speed[0] = [x,y,z]
        normal_speed = ti.math.normalize(self.speed[0])
        if ti.math.length(self.speed[0]) < 1e-6:
            normal_speed[0], normal_speed[1], normal_speed[2] = 0.0, 0.0, 0.0
        self.speed_normal[0][0], self.speed_normal[0][1], self.speed_normal[0][2] = normal_speed[0],normal_speed[1],normal_speed[2]

   
   
    
    @ti.func
    def rotate(self,quaternion): # quaternion rotation from wiki
        new_quaternion = self.quat_mul(quaternion, self.quaternion)
        self.quaternion = new_quaternion
        theta = self.new_quaternion[0][0]
        axis = [self.new_quaternion[0][1], self.quaternion[0][2], self.quaternion[0][3]]
        for p in self.x:
            complex_p = [0.0, self.x[p][0] - self.displacment[0][0], self.x[p][1] - self.displacment[0][1], self.x[p][2] - self.displacment[0][2]]
            complex_q = [ti.math.cos(theta/2), axis[0] * ti.math.sin(theta/2), axis[1] * ti.math.sin(theta/2), axis[2] * ti.math.sin(theta/2)]
            complex_q_inv = [ti.math.cos(theta/2), -axis[0] * ti.math.sin(theta/2), -axis[1] * ti.math.sin(theta/2), -axis[2] * ti.math.sin(theta/2)]
            p_prime = self.quat_mul(self.quat_mul(complex_q, complex_p),complex_q_inv)
            self.x[p][0], self.x[p][1], self.x[p][2] = p_prime[1] + self.displacment[0][0], p_prime[2] + self.displacment[0][1], p_prime[3] + self.displacment[0][2]
        
    def save_array(self):
        array = self.collision_box.to_numpy()
        np.save('collision_box.npy', array)
        array = self.offset.to_numpy()
        np.save('offset.npy', array)
        array = self.cf_coefficient.to_numpy()
        np.save('cf_coefficient.npy', array)
        array = self.cf_direction.to_numpy()
        np.save('cf_direction.npy', array)
        array = self.sdf_filed.to_numpy()
        np.save('sdf_filed.npy', array)


        


# Load external files
#@ti.data_oriented
# class Obj:
#     def __init__(self, file_name, center, e_radius, mass, pho, color):
#         self.ini_mesh(file_name)
#         self.center[0] = center
#         self.e_radius = e_radius
#         self.n = self.mesh.shape[0]
#         self.m = mass / self.n
#         self.vol = self.m / pho
#         self.color = color
        
#         # Taichi field
#         self.x = ti.Vector.field(3, dtype=ti.f32, shape=self.n) # position field
#         self.x.from_numpy(self.mesh * 0.1)
#         face_array = np.array(self.faces).astype(np.int32)
#         face_array = face_array.reshape(face_array.shape[0] * 3)
#         self.indices = ti.field(dtype=ti.i32, shape=(face_array.shape)) # indices field
#         self.indices.from_numpy(face_array)
#         self.v = ti.Vector.field(3, dtype=ti.f32, shape=self.n) # velocity field
#         self.C = ti.Matrix.field(3, 3, dtype=float, shape=self.n)  # affine velocity field
#         self.F = ti.Matrix.field(3, 3, dtype=float, shape=self.n)  # deformation gradient
#         self.F_j = ti.field(dtype = float, shape=self.n)
#         self.colors = ti.Vector.field(3, dtype=float, shape=self.n)
        
#         self.displacment = ti.Vector.field(3, dtype=ti.f32, shape = 1)
#         self.displacment[0] = center
        
#         self.initialize()

    
#     def ini_mesh(self,file_name):
#         mesh = Mesh(file_name)
#         self.mesh, self.faces = mesh.vertices(), mesh.faces()
    
#     @ti.kernel
#     def initialize(self): # initialize particle positions
#         for p in self.x:
#             self.x[p] += self.center[0]
#             self.F[p] = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
#             self.F_j[p] = 1.0

#     # coordinate in particle to index in position filed. e.g., row 2 , col 3, layer 1 correspond to p'th partcicle in p
#     def coordinate_2_index(self,cor):
#         p = int(cor[2] * self.nx * self.ny + cor[1] * self.nx + cor[0])
#         return p
    
#     @ti.func
#     def translate(self, dis: ti.template()):
#         for p in self.x:
#             self.x[p] += dis
#         self.displacment[0] = dis
    
#     @ti.func
#     def quat_mul(self, q1:ti.template(), q2:ti.template()): # quaternion multiplication from wiki
#         a1, b1, c1, d1 = q1[0], q1[1], q1[2], q1[3]
#         a2, b2, c2, d2 = q2[0], q2[1], q2[2], q2[3]
#         t1 = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
#         t2 = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
#         t3 = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
#         t4 = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2
#         result = [t1, t2, t3, t4]
#         return result

    
#     @ti.func
#     def rotate(self, axis: ti.template(), theta: ti.template()): # # quaternion rotation from wiki
#         for p in self.x:
#             complex_p = [0.0, self.x[p][0] - self.displacment[0][0], self.x[p][1] - self.displacment[0][1], self.x[p][2] - self.displacment[0][2]]
#             complex_q = [ti.math.cos(theta/2), axis[0] * ti.math.sin(theta/2), axis[1] * ti.math.sin(theta/2), axis[2] * ti.math.sin(theta/2)]
#             complex_q_inv = [ti.math.cos(theta/2), -axis[0] * ti.math.sin(theta/2), -axis[1] * ti.math.sin(theta/2), -axis[2] * ti.math.sin(theta/2)]
#             p_prime = self.quat_mul(self.quat_mul(complex_q, complex_p),complex_q_inv)
#             self.x[p][0], self.x[p][1], self.x[p][2] = p_prime[1] + self.displacment[0][0], p_prime[2] + self.displacment[0][1], p_prime[3] + self.displacment[0][2]
    
    
#     @ti.func
#     def set_x(self, tar: ti.template(),p):
#         for I in ti.static(range(3)):
#             self.x[p][I] = tar[I]
    
#     @ti.func
#     def set_v(self, tar: ti.template(),p):
#         for I in ti.static(range(3)):
#             self.v[p][I] = tar[I]
    
#     @ti.func
#     def set_C(self, tar: ti.template(),p):
#         for I in ti.static(ti.grouped(ti.ndrange(3, 3))):
#             self.C[p][I] = tar[I]
    
#     @ti.func
#     def set_F(self, tar: ti.template(),p):
#         for I in ti.static(ti.grouped(ti.ndrange(3, 3))):
#             self.F[p][I] = tar[I]
            
#     @ti.func
#     def set_F_j(self, tar: ti.template(),p):
#             self.F_j[p] = tar

            
            


            





