#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 17:59:16 2023

@author: luyin
"""

import pymeshlab
ms = pymeshlab.MeshSet()
ms.load_new_mesh('hanger.ply')
pymeshlab.print_filter_list()
pymeshlab.print_filter_parameter_list('generate_surface_reconstruction_screened_poisson')
pymeshlab.search('poisson')
