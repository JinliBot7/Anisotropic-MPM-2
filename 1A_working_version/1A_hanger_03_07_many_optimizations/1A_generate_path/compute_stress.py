#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:58:33 2023

@author: luyin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 17:34:05 2022
@author: luyin
"""
import taichi as ti

@ti.func
def compute_stress(F:ti.template(), mu, lam):
    
    k = 5e2
    gamma = 500
    
    #Q, R = QR_given_rotation(F)
    Q, R = QR3(F)
    Q_T = Q.transpose()
    R_T = R.transpose()
    Q_T_inv = Q_T.inverse()
    R_T_inv = R_T.inverse()
    r11 = R[0,0]
    r12 = R[0,1]
    r13 = R[0,2]
    r22 = R[1,1] 
    r23 = R[1,2]
    r33 = R[2,2]
    
    energy = 0.0
    
    if r33 >= 1:
        r13 = R[0,2] = 0.0
        r23 = R[1,2] = 0.0
        r33 = R[2,2] = 1.0
        
    
    if r33 < 1:
        f_n = k / (r11 * r22) * (r33 - 1) ** 2
        f_f = gamma / (r11 * r12) * (r13 ** 2 + r23 ** 2) ** 0.5
        cf = 0.0
        if f_f > cf * f_n:
            r13 = R[0,2] = cf * f_n / f_f * r13
            r23 = R[1,2] = cf * f_n / f_f * r23
    
    # fixed corotated potential
    F_2d = ti.Matrix([[r11, r12], [0.0, r22]])
    R_2d, S_2d = ti.polar_decompose(F_2d) # R rotation, S strech
    J_2d = F_2d.determinant()
    #F_2d_T = F_2d.transpose()
    #F_2d_inv_T = F_2d_T.inverse()
    #P1 = 2 * mu * (F_2d - R_2d) + lam * (J_2d - 1) * J_2d * F_2d_inv_T
    d_F_2d_d_F = ti.Matrix([[r22, 0.0], [0.0, r11]])
    P1 = 2 * mu * (F_2d - R_2d) + lam * (J_2d - 1) * d_F_2d_d_F
    
    # shearing of normal to surface
    P2 = [gamma * r13, gamma * r23]
    
    # compression
    P3 = 0.0

    if r33 <= 1 :
        P3 = - k * (1 - r33) ** 2
    
    dphi_dR = ti.Matrix([
        [P1[0,0], P1[0,1], P2[0]],
        [P1[1,0], P1[1,1], P2[1]],
        [0.0, 0.0, P3]])

    upper = dphi_dR @ R_T
    
    QT_P_RT = ti.Matrix([
        [upper[0,0], upper[0,1], upper[0,2]],
        [upper[0,1], upper[1,1], upper[1,2]],
        [upper[0,2], upper[1,2], upper[2,2]]
        ])
    
    P = (Q_T_inv @ (QT_P_RT @ R_T_inv))         
    
    F_elastic = Q @ R
    F_plastic = F_elastic.inverse() @ F
        
    return P, F_elastic, F_plastic

# # QR decomposition adapted from https://github.com/danbar/qr_decomposition
# @ti.func
# def QR_given_rotation(A):
#     Q = ti.Matrix.identity(ti.f32,3)
#     R = A
#     Q, R = QR_given_rotation_submodule_1(Q,R)
#     Q, R = QR_given_rotation_submodule_2(Q,R)
#     Q, R = QR_given_rotation_submodule_3(Q,R)
#     return Q, R

# @ti.func
# def QR_given_rotation_submodule_1(Q,R):
#     if R[1, 0] != 0:
#         c, s = _givens_rotation_matrix_entries(R[0, 0], R[1, 0])
#         G = ti.Matrix.identity(ti.f32,3)
#         G[0,0] = c
#         G[1,1] = c
#         G[1, 0] = s
#         G[0, 1] = -s
#         R = G @ R
#         Q = Q @ G.transpose()
#     return Q,R

# @ti.func
# def QR_given_rotation_submodule_2(Q,R):
#     if R[2, 0] != 0:
#         c, s = _givens_rotation_matrix_entries(R[0, 0], R[2, 0])
#         G = ti.Matrix.identity(ti.f32,3)
#         G[0,0] = c
#         G[2,2] = c
#         G[2, 0] = s
#         G[0, 2] = -s
#         R = G @ R
#         Q = Q @ G.transpose()
#     return Q,R

# @ti.func
# def QR_given_rotation_submodule_3(Q,R):
#     if R[2, 1] != 0:
#         c, s = _givens_rotation_matrix_entries(R[1, 1], R[2, 1])
#         G = ti.Matrix.identity(ti.f32,3)
#         G[1,1] = c
#         G[2,2] = c
#         G[2, 1] = s
#         G[1, 2] = -s
#         R = G @ R
#         Q = Q @ G.transpose()
#     return Q,R

# @ti.func
# def _givens_rotation_matrix_entries(a, b):
#     r = (a ** 2 + b ** 2) ** 0.5
#     c = a/r
#     s = -b/r
#     return c, s

@ti.func
  #3x3 mat, Gram–Schmidt Orthogonalization
def QR3(Mat:ti.template()) :
    a1 = ti.Vector([Mat[0,0],Mat[1,0],Mat[2,0]])
    a2 = ti.Vector([Mat[0,1],Mat[1,1],Mat[2,1]])
    a3 = ti.Vector([Mat[0,2],Mat[1,2],Mat[2,2]])
    u1 = a1
    e1 = u1/u1.norm()
    u2 = a2 - projection(u1,a2)
    e2 = u2/u2.norm()
    u3 = a3 - projection(u1,a3)-projection(u2,a3)
    e3 = u3/u3.norm()
    #r11, r12, r13, r22, r23, r33 = e1.transpose()@a1, e1.transpose()@a2, e1.transpose()@a3, e2.transpose()@a2, e2.transpose()@a3, e3.transpose()@a3
    r11, r12, r13, r22, r23, r33 = e1.dot(a1), e1.dot(a2), e1.dot(a3), e2.dot(a2), e2.dot(a3), e3.dot(a3)
    R = ti.Matrix([[r11,r12,r13],[0.0,r22,r23],[0.0,0.0,r33]])
    Q = ti.Matrix.cols([e1,e2,e3])
    return Q,R

@ti.func
def projection(u,a):
    proj = ti.Vector([0.0,0.0,0.0])
    # upper = u.transpose()@a
    # lower = u.transpose()@u
    upper = u.dot(a)
    lower = u.dot(u)
    compute = upper/lower*u
    proj[0],proj[1],proj[2] = compute[0],compute[1],compute[2]
    return proj