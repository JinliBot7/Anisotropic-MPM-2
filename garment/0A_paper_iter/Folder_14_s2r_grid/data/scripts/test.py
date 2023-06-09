#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 23:29:46 2023

@author: luyin
"""
import numpy as np

drag_list = []
lift_list = []
loss_list = []

for trial_num in range(1718):

    A_drag = np.load(f'../trials/trial_{trial_num}/drag_coe_np.npy')
    A_lift = np.load(f'../trials/trial_{trial_num}/lift_coe.npy')
    A_loss = np.load(f'../trials/trial_{trial_num}/loss.npy') / (196 * 24) * 100
    drag_list.append(A_drag)
    lift_list.append(A_lift)
    loss_list.append(A_loss)

loss_np = np.array(loss_list)
# v_input_np = np.load(f'../trials/trial_{trial_num}/v_input.npy')
# print(v_input_np[1])
# v_input_ti = ti.Vector.field(3, ti.f32, shape=(200,10))
# v_input_ti.from_numpy(v_input_np)