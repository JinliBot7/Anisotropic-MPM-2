#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import taichi as ti
from geometries import Obj, Boundary
from render import set_render, render

#ti.init(ti.cpu, cpu_max_num_threads=1) # for debug
ti.init(ti.cuda,device_memory_fraction = 0.8,fast_math=False)

hanger = Boundary('./Hanger',[0.0, 0.0, 0.0], (0.5, 0.5, 0.7))
obj_list = []
boundary_list = [hanger]

    
def main():
    counter = 0
    camera, scene, canvas, window, axisX, axisY, axisZ, colorX, colorY, colorZ = set_render()
    floor = ti.Vector.field(3,dtype = ti.f32,shape = 4)
    floor_index = ti.field(dtype = ti.i32,shape = 6)
    floor[0], floor[1], floor[2], floor[3] = [0.0, 0.0, -1e-3], [1.0, 0.0, -1e-3], [0.0, 1.0, -1e-3], [1.0, 1.0, -1e-3]
    floor_index[0], floor_index[1],floor_index[2], floor_index[3], floor_index[4], floor_index[5] = 0, 1, 2, 3, 2, 1
    floor_color = (0.4, 0.4, 0.4)
    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.GUI.ESCAPE:
                window.destroy()
        
            
        render(obj_list, boundary_list, camera, scene, canvas, window, axisX, axisY, axisZ, colorX, colorY, colorZ)
        scene.mesh(floor, indices = floor_index, color = floor_color)
        
        window.show()
        
        counter += 1

if __name__ == "__main__":
    main()




























