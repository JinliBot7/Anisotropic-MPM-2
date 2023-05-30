#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 10:45:53 2023

@author: luyin
"""

import shutil
import os

for i in range(4,500,5): 
    print(i)
    #shutil.rmtree('./{i:d}/frames'.format(i = i))
    os.remove('./{i:d}/video.gif'.format(i = i))