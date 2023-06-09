#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 02:42:09 2023

@author: luyin
"""
import taichi as ti
import numpy as np

ti.init(ti.cuda,debug=True)


@ti.data_oriented
class TestClass():
    def __init__(self):  
        self.x = ti.Vector.field(3,ti.f32,shape=2)
    
    def update(self):
        A_np = np.ones((2,3)).astype(np.float32)
        #self.x = ti.Vector.field(3,ti.f32,shape=2)
        self.x.from_numpy(A_np)
        print('In the update fucntion print self.x[0]: \n',self.x[0])


@ti.kernel
def test_class(A:ti.template()):
    print(A.x[0])
    
    
ti.sync()
A_obj = TestClass()
print('before update call kernel function: ')
test_class(A_obj)

A_obj.update()

print('after update call kernel function:')
test_class(A_obj) # Why all zeros here?

print('after update print A.x[0] value:',A_obj.x[0])


