#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 17:07:13 2022

@author: luyin
"""
import taichi as ti

def set_render():
    res = (int(1080 * 0.5), int(720 * 0.5))
    window = ti.ui.Window("MPM 3D", res, vsync=True,show_window = True)
    canvas = window.get_canvas()
    canvas.set_background_color((1,1,1))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    
    scale = 1.0
    camera.position(1.2*scale, 0.1*scale, 0.5*scale)
    camera.lookat(0.5, 0.6, 0.45)
    
    # scale = 1.2
    # camera.position(0.9*scale, 0.9*scale, 0.75*scale)
    # camera.lookat(0, 0, 0)

    camera.up(0, 0, 1)

    # Set axis
    axisLength = 0.5
    axisX = ti.Vector.field(3, float, (2,))
    axisX[1][0], axisX[1][1], axisX[1][2] = axisLength, 0, 0
    colorX = (0.5, 0, 0)
    axisY = ti.Vector.field(3, float, (2,))
    axisY[1][0], axisY[1][1], axisY[1][2] = 0, axisLength, 0
    colorY = (0, 0.5, 0)
    axisZ = ti.Vector.field(3, float, (2,))
    axisZ[1][0], axisZ[1][1], axisZ[1][2] = 0, 0, axisLength
    colorZ = (0, 0, 0.5)
    
    render_list = [camera, scene, canvas, window, axisX, axisY, axisZ, colorX, colorY, colorZ]
    return render_list

def render(counter, obj_list, manipulator_list, path_list, camera, scene, canvas, window, axisX, axisY, axisZ, colorX, colorY, colorZ):
    camera.track_user_inputs(window, movement_speed=0.01, hold_key=ti.ui.RMB)

    scene.point_light(pos=(0.0, 0.8, 1.2), color=(1.0, 1.0, 1.0))
    #scene.point_light(pos=(0.0, -0.8, 1.2), color=(1.0, 1.0, 1.0))   


    scene.ambient_light([1.0, 1.0, 1.0])
    scene.set_camera(camera)
    for i in range(len(obj_list)):
        scene.mesh(obj_list[i].x_t, indices = obj_list[i].mesh_indices, per_vertex_color = obj_list[i].colors)
        
        #scene.particles(obj_list[i].x_t, radius = obj_list[i].e_radius * 0.5, per_vertex_color = obj_list[i].colors)
        #print(i,obj_list[i].colors[0],obj_list[i].colors[1000])
        scene.particles(obj_list[i].pio_x_t, radius = obj_list[i].e_radius * 0.8  , color = (0.0, 0.5, 0.0))
        #pass

    for i in range(len(manipulator_list)):
        scene.mesh(manipulator_list[i].x, indices = manipulator_list[i].mesh_indices, color = manipulator_list[i].color)   
        #scene.particles(manipulator_list[i].x, radius = manipulator_list[i].e_radius * 1, color = manipulator_list[i].color)
        #scene.particles(manipulator_list[i].collision_box, radius = manipulator_list[i].e_radius * 0.3, color = (0.8, 0.2, 0.2))
    for i in range(len(path_list)):
        scene.particles(path_list[i].x, radius = obj_list[0].e_radius * 0.25, per_vertex_color = path_list[i].colors)
        scene.particles(path_list[i].pio_x, radius = obj_list[0].e_radius * 0.8, color = (0.0, 0.5, 0.0))
    # #scene.particles(ball_center, radius = ball_radius * 0.9, color = (0.3, 0.3, 0.3))

    # scene.lines(axisX, 2, color=colorX)
    # scene.lines(axisY, 2, color=colorY)
    # scene.lines(axisZ, 2, color=colorZ)
    canvas.scene(scene)

    



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
