#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:57:43 2023

@author: luyin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import taichi as ti
from geometries import Hanger, Path, Obj
from render import set_render, render
import numpy as np
from math import pi
import time


#ti.init(ti.cpu, cpu_max_num_threads=1) # for debug
ti.init(ti.cuda,device_memory_fraction = 0.9, offline_cache = False, kernel_profiler = False)

center, dimension = ti.Vector([0.5, 0.5, 0.37]), ti.Vector([0.35, 0.35, 0.0005]) # dimension changes shape. if dimension[i] < e_radius, single layer
axis_angle = ti.Vector([-180 / 180 * pi, 1.0, 0.0, 0.0]) # angle and rotation axis

e_radius, total_mass, pho = 0.0019, dimension[0] * dimension[1] * dimension[2], 2.0 # e_radius is the radius of the element. The distance between two elements is designed as 2 * e_radius.
color = (0.5,0.5,0.8)
obj_0 = Obj(center, dimension, axis_angle, e_radius, total_mass, pho, color)

# Hanger
hanger = Hanger('./Hanger',[0.5, 0.5, 0.5], (0.4, 0.4, 0.4))

path_list = []
# Path

    
obj_list = [obj_0]
boundary_list = [hanger]

Circle_Center = ti.Vector.field(3, dtype = float, shape = 1)
Circle_Center[0] = [0.45, 0.5, 0.3]
Circle_Radius = 0.02

def main():
    counter = 0
    camera, scene, canvas, window, axisX, axisY, axisZ, colorX, colorY, colorZ = set_render()
    floor = ti.Vector.field(3,dtype = ti.f32,shape = 4)
    floor_index = ti.field(dtype = ti.i32,shape = 6)
    floor[0], floor[1], floor[2], floor[3] = [0.0, 0.0, -1e-3], [1.0, 0.0, -1e-3], [0.0, 1.0, -1e-3], [1.0, 1.0, -1e-3]
    floor_index[0], floor_index[1],floor_index[2], floor_index[3], floor_index[4], floor_index[5] = 0, 1, 2, 3, 2, 1
    floor_color = (0.4, 0.4, 0.4)
    #ti.profiler.clear_kernel_profiler_info()  # Clears all records
    video_manager = ti.tools.VideoManager('./video',framerate=60,automatic_build=False)
    
    while counter < 142:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.GUI.ESCAPE:
                window.destroy()
        
        path = Path('./target_path', 17650 - counter * 50, (0.5,0.5,0.8), obj_0.n, obj_0.pio_n, obj_0.nx)
        
        path.initialize()
        path_list = [path]
        obj_list = [obj_0]
        boundary_list = [hanger]
        #time.sleep(0.01)
        #update_dt()

        render(obj_list, boundary_list, path_list ,camera, scene, canvas, window, axisX, axisY, axisZ, colorX, colorY, colorZ,Circle_Center,Circle_Radius)
        scene.mesh(floor, indices = floor_index, color = floor_color)
        window.show()
        
        img = window.get_image_buffer_as_numpy()
        video_manager.write_frame(img)
        
        
        counter += 1
        print(counter)
    video_manager.make_video(gif=True, mp4=True)
        
        
        

        

if __name__ == "__main__":
    main()
    # ti.sync()
    # ti.profiler.print_kernel_profiler_info()  # The default mode: 'count'


