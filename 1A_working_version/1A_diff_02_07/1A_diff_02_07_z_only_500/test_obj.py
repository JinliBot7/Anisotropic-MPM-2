#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:25:59 2022

@author: luyin
"""

import taichi as ti


@ti.data_oriented
class Test_class():
    def __init__(self):
        self.center_filed = ti.Vector.field(3, dtype = ti.f32, shape = 1)
        self.center_filed[0] = ti.Vector([0.0, 0.0, 1.0])
        print(self.center_filed,type(self.center_filed))
        print(self.center_filed[0],type(self.center_filed[0]))
    
    @ti.kernel
    def test_kernel(self):
        self.test_function(self.center_filed[0])

    @ti.func
    def test_function(arg:ti.types.vector(3,ti.f32)):
        print('func')
    



# # Test = Test_class()
# # Test.check_kernel(Test.center_filed[0])

# arg_field = ti.Vector.field(3, dtype = ti.f32, shape = 1)

# @ti.kernel
# def test_kernel():
#     print('kernel')
#     arg_field[0] = ti.Vector([0.0, 1.0, 2.0])
#     test_function(arg_field[0])

# @ti.func
# def test_function(arg:ti.types.vector(3,ti.f32)):
#     print('func')


# test_kernel()