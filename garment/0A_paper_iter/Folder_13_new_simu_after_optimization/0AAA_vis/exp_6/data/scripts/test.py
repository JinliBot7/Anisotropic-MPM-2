#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 23:29:46 2023

@author: luyin
"""
import numpy as np

for i in range(50):
    trial_num = i
    if i == 14:
        continue
    loss = np.load(f'../trials/trial_{trial_num}/loss.npy')
    target_iter_num = np.where(loss == np.amin(loss))[0]
    print(trial_num,target_iter_num, np.amin(loss))
    if np.amin(loss) <= 3:
        print(target_iter_num, np.amin(loss))

# v_input_np = np.load(f'../trials/trial_{trial_num}/v_input.npy')
# print(v_input_np[1])
# v_input_ti = ti.Vector.field(3, ti.f32, shape=(200,10))
# v_input_ti.from_numpy(v_input_np)