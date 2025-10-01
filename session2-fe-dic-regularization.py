# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 17:37:38 2025

@author: passieux
"""

import numpy as np
import matplotlib.pyplot as plt
import pyxel as px
import scipy.sparse as sps

# %% Images (STUDENTS)
filename = 'oht_cfrp_%02d.tiff'
I0 = px.Image(filename % 0).Load()
# I0.Plot()
It = px.Image(filename % 10).Load()
# It.Plot()

sigf = 1.   # estimated before

# %% MESH
box = np.array([[0, 0], [90, 30]])
r = 5
cpos = np.array([45.1, 15.3])
lc = 1
lf = 0.5
m = px.OpenHolePlateUnstructured(box, r, cpos, lc, lf)
m.Plot()

# %% PRECALIBRATED CAMERA MODEL
cam = px.Camera(2)
p = np.array([-0.002176, 2.237317, 33.687119, 9.51087e-02])
cam.set_p(p)
px.PlotMeshImage(I0, m, cam)

# %% DIC RESOLUTION WITHOUT REGULARIZATION
m.Connectivity()
m.DICIntegration(cam)

U0 = px.MultiscaleInit(I0, It, m, cam, scales=[3, 2, 1])
U_noreg, res_noreg = px.Correlate(I0, It, m, cam, U0=U0)

# Visualization: displacement fields
m.PlotContourDispl(U_noreg, s=30)

# Visualization: strain fields
m.PlotContourStrain(U_noreg, clim=1, cmap='RdBu')


# %% DIC WITH LAPLACIAN REGULARIZATION
# play with the regularization length below from 0 (no regularisation) to infty
l0 = 2
L = m.Laplacian()
U_reg1, res = px.Correlate(I0, It, m, cam, U0=U0, L=L, l0=l0)

# Visualization: superimpose warped meshes
m.Plot(alpha=0.2, edgecolor='r')         # in light red, the reference config
m.Plot(U_noreg, 30, alpha=0.4)           # in light black, no regularization
m.Plot(U_reg1, 30)                       # solid black, regularization

# %% DIC WITH LAPLACIAN REGULARIZATION
L = m.Laplacian()
l0_all = np.array([1, 2, 5, 10, 15, 20, 30, 50])
U = U_noreg.copy()
ULU_G = [np.sqrt(U_noreg.T @ L @ U_noreg)]
rr_G = [np.sqrt(res_noreg.T @ res_noreg)]
for l0 in l0_all:
    U, res = px.Correlate(I0, It, m, cam, U0=U, L=L, l0=l0)
    ULU_G += [np.sqrt(U.T @ L @ U)]
    rr_G += [np.sqrt(res.T @ res)]

# Plot the L-Curve
plt.loglog(rr_G, ULU_G, 'ko-')
for i in range(1, len(l0_all)):
    plt.text(rr_G[i], ULU_G[i], str(l0_all[i]))
    
# Identify the optimal regularization length and run again:
l0_G_opt = 5#TODO
U_G_opt, res_G_opt = px.Correlate(I0, It, m, cam, U0=U0, L=L, l0=l0_G_opt)
# Visualization: displacement fields
m.PlotContourDispl(U_G_opt, s=30)

# Visualization: strain fields
m.PlotContourStrain(U_G_opt, clim=1, cmap='RdBu')

#
m.Plot(alpha=0.2, edgecolor='r')         # in light red, the reference config
m.Plot(U_noreg, 30, alpha=0.4)           # in light black, no regularization
m.Plot(U_G_opt, 30)                       # solid black, regularization


# %% DIC WITH MECHANICAL REGULARIZATION
El = 20.3e3
Et = 15.4e3
nult = 0.14
Glt = 0.93e3
hooke = px.Hooke([El, Et, nult, Glt], 'orthotropic_2D')
# hooke = px.Hooke([20e3, 0.2])
K = m.Stiffness(hooke)
nodes_l = m.SelectEndLine('left')
nodes_r = m.SelectEndLine('right')
nodes_no_regul = np.append(nodes_l, nodes_r)
dof_no_regul = m.conn[nodes_no_regul].ravel()
D = np.ones(m.ndof)
D[dof_no_regul] = 0
# m.PlotContourDispl(D, s=0, stype='mag')
Ddiag = sps.diags(D)
KDK = K @ Ddiag @ K

U, res = px.Correlate(I0, It, m, cam, U0=U_noreg, L=KDK, l0=1)


# %% L-curve

U, res = px.Correlate(I0, It, m, cam, U0=U0)
l0_all = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 5, 10, 15, 20, 30, 50, 75, 100])
l0_all = np.array([0.1, 0.3, 0.5, 1, 5, 10, 20, 30, 50, 75])
ULU_M = [np.sqrt(U_noreg.T @ KDK @ U_noreg)]
rr_M = [np.sqrt(res_noreg.T @ res_noreg)]
U = U_noreg.copy()
for l0 in l0_all:
    U, res = px.Correlate(I0, It, m, cam, U0=U0, L=KDK, l0=l0)
    ULU_M += [np.sqrt(U.T @ KDK @ U)]
    rr_M += [np.sqrt(res.T @ res)]

plt.loglog(rr_G, np.array(ULU_G)*1000, 'ko-', alpha=0.5)
plt.loglog(rr_M, ULU_M, 'ko-')
for i in range(1, len(l0_all)):
    plt.text(rr_M[i], ULU_M[i], str(l0_all[i]))

l0_M_opt = 10 #TODO
U_M_opt, res_M_opt = px.Correlate(I0, It, m, cam, U0=U, L=L, l0=l0_M_opt)
# Visualization: displacement fields
m.PlotContourDispl(U_M_opt, s=30)

# Visualization: strain fields
m.PlotContourStrain(U_M_opt, clim=1, cmap='RdBu')

m.Plot(U_noreg, 30, alpha=0.4)           # in light black, no regularization
m.Plot(U_G_opt, 30)                       # solid black, regularization
m.Plot(U_M_opt, 30)                       # solid black, regularization

# %% RESIDUALS

# Initialization
emp = px.ExportPixMap(I0, m, cam)

# Get Residual on the pixel map
Rmap = emp.PlotResidual(I0, It, U_G_opt)
Rmap = emp.PlotResidual(I0, It, U_M_opt)

# %% 
box2 = np.array([[0, 0], [30, 30]])
m2 = px.StructuredMeshQ4(box2, 0.5)
m2.Connectivity()
m2.DICIntegration(cam)
U20 = px.MultiscaleInit(I0, It, m2, cam, scales=[3, 2, 1])
U2, res2 = px.Correlate(I0, It, m2, cam, U0=U20)

m2.Plot(alpha=0.2)
m2.Plot(U2, 30)

tx, ty, rz = m2.RBM()
R = np.c_[tx, ty, rz]

U0proj = R @ np.linalg.solve(R.T@R, R.T@U20)
m.Plot(U0proj, 30)

U_rbm, res_rbm = px.Correlate(I0, It, m2, cam, U0=U0proj, Basis=R)
m2.Plot(alpha=0.2)
m2.Plot(U_rbm, 30)

# %%
an = np.where(m2.conn[:, 0] > -1)[0]
tx, ty, rz = m2.RBM()
xn = m2.n[an, 0] - np.mean(m2.n[an, 0])
yn = m2.n[an, 1] - np.mean(m2.n[an, 1])
xn/=np.max(xn)
yn/=np.max(yn)

ux = np.zeros(m2.ndof)
ux[m2.conn[an, 0]] = xn
uy = np.zeros(m2.ndof)
uy[m2.conn[an, 0]] = yn
vx = np.zeros(m2.ndof)
vx[m2.conn[an, 1]] = xn
vy = np.zeros(m2.ndof)
vy[m2.conn[an, 1]] = yn

m2.Plot(edgecolor='r', alpha=0.4)
# m2.Plot(vy**2*vx)

R = np.c_[tx, ty, rz, ux, uy, vx, vy, ux*uy, vx*vy, ux**2, uy**2, vx**2, vy**2,
          ux**2*uy, uy**2*ux, vx**2*vy, vy**2*vx]

U0proj = R @ np.linalg.solve(R.T@R, R.T@U20)

U_poly, res_poly = px.Correlate(I0, It, m2, cam, U0=U0proj, Basis=R)
m2.Plot(U2, 30, edgecolor='r')
m2.Plot(U_poly, 30)


# %% Experimental-BC Driven DIC

rep = np.setdiff1d(np.arange(m.ndof), dof_no_regul)
U = 0*U_noreg
U[dof_no_regul] = U_noreg[dof_no_regul].copy()
repk = np.ix_(rep, rep)
f = -K@U
U[rep] = sps.linalg.spsolve(K[repk], f[rep])
m.Plot(U0, 30)
m.Plot(U_noreg, 30, edgecolor='b')

