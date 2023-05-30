#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 11:05:35 2023

@author: luyin
"""

import os
for i in range(0,50,2):
    mydir = f'./vis/{i}/frames'
    for f in os.listdir(mydir):
        if not f.endswith(".png"):
            continue
            print(f)
        os.remove(os.path.join(mydir, f))

# mydir = './vis/0/frames'
# for f in os.listdir(mydir):
#     if not f.endswith(".png"):
#         continue
#         print(f)
#     os.remove(os.path.join(mydir, f))
    

# for i in range(4,100,5):
#     mydir = f'./vis/{i}'
#     for f in os.listdir(mydir):
#         if not f.endswith(".mp4"):
#             continue
#             print(f)
#         os.remove(os.path.join(mydir, f))

# mydir = './vis/0'
# for f in os.listdir(mydir):
#     if not f.endswith(".mp4"):
#         continue
#         print(f)
#     os.remove(os.path.join(mydir, f))