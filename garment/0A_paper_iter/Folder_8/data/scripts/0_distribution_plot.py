#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 23:29:46 2023

@author: luyin
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as pat

loss_min_list = []
obj_center_list = []
target_center_list = []
target_rot_list = []

suc_target_center_list = []
suc_target_rot_list = []
suc_obj_center_list = []
suc_trial_num_list = []

terminate_iter_list = []

total_trial = 60



suc_count = 0
for i in range(total_trial):
    trial_num = i
    loss = np.load(f'../trials/trial_{trial_num}/loss.npy')

    #plt.plot(loss)
    loss_min = np.amin(loss)
    target_iter_num = np.where(loss == np.amin(loss))[0]
    if loss_min == 0.0:
        print(trial_num)
    
    obj_center_np = np.load(f'../trials/trial_{trial_num}/obj_center.npy')
    target_num = np.load(f'../trials/trial_{trial_num}/target_num.npy')[0]
    target_center = np.load(f'../../asset/target_path/center_{target_num}.npy')
    target_rot = np.load(f'../../asset/target_path/rot_{target_num}.npy')[0] / math.pi * 180 + 180
    
    obj_center_list.append([obj_center_np[1],obj_center_np[2]])
    target_center_list.append(target_center[2])
    target_rot_list.append(target_rot)
    
    if (np.amin(loss) < 2):
        suc_count +=1
        suc_obj_center_list.append([obj_center_np[1],obj_center_np[2]])
        suc_target_center_list.append(target_center[2])
        suc_target_rot_list.append(target_rot)
        #print(i, loss_min)

all_sample_center_list = []
all_sample_rot_list = []

for i in range(10000):
    center_z = np.load(f'../../asset/target_path/center_{i}.npy')[2]
    all_sample_center_list.append(center_z)
    rot = np.load(f'../../asset/target_path/rot_{i}.npy')[0] / math.pi * 180 + 180
    all_sample_rot_list.append(rot)

# sub_plot_count = 1
# row = int(suc_count / 6) + 1
# for i in range(total_trial):
#     trial_num = i
#     loss = np.load(f'../trials/trial_{trial_num}/loss.npy')
#     if (np.amin(loss) < 3):
#         plt.subplot(row, 6, sub_plot_count)
#         plt.plot(loss)
#         plt.ylim(0,50)
#         sub_plot_count +=1

#%%
fig, ax = plt.subplots()

rect = pat.Rectangle((0.35,0.0),0.1,80,alpha = 0.05,fill=True,label = "sampling region",color = 'green',zorder = -1)
ax.add_patch(rect)
color = ['gray']
plt.scatter(np.array(all_sample_center_list),np.array(all_sample_rot_list),c=color,marker = 'o', s = 0.1, label = 'collision free samples',)    
color = ['black']
plt.scatter(np.array(target_center_list),np.array(target_rot_list),c=color,marker = 'o', s = 5, label = 'failed trials')
color = ['red']
plt.scatter(np.array(suc_target_center_list),np.array(suc_target_rot_list),c=color,marker = 'o', s = 5, label = 'successful trials')
plt.ylim(-5, 100)
plt.xlabel(r'$p_{t_{z}}(m)$')
plt.ylabel(r'$\alpha_{t_x}(^o)$')
plt.legend(loc='upper left')
plt.title('Target Initial Configuration Distribution')

# obj_center_list_np = np.array(obj_center_list)
# suc_obj_center_list_np = np.array(suc_obj_center_list)
# rect = pat.Rectangle((0.6,0.3),0.3,0.5,alpha = 0.05,fill=True,label = "sampling region",color = 'green',zorder = -1)
# ax.add_patch(rect)
# plt.scatter(obj_center_list_np[:,0],obj_center_list_np[:,1],c='black',label='failed trials', s=15)
# plt.scatter(suc_obj_center_list_np[:,0],suc_obj_center_list_np[:,1],c='red',label = 'successful trials', s=15)
# plt.title('Fabric Initial Positon Distribution')
# plt.ylim(0.25, 1)
# plt.xlabel(r'$p_{o_{y}}(m)$')
# plt.ylabel(r'$p_{o_{z}}(m)$')
# plt.legend(loc='upper left')


plt.savefig('target_ini_position.png', dpi=300)
#plt.savefig('obj_ini_position.png', dpi=300)
plt.show()

#pat.Rectangle((0.6,0.3),30,60)
#plt.show()
    
    
