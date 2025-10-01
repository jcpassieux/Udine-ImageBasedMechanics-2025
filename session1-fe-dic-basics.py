# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 17:37:38 2025

@author: passieux
"""

import numpy as np
import matplotlib.pyplot as plt
import pyxel as px

# %% Images (STUDENTS)
filename = 'oht_cfrp_%02d.tiff'
I0 = px.Image(filename % 0).Load()
# I0.Plot()
It = px.Image(filename % 10).Load()
# It.Plot()

# plot initial residual
plt.imshow(I0.pix-It.pix, cmap='RdBu')
plt.colorbar()

# %% EVALUATION OF IMAGE NOISE (ME)
# I0.SelectROI()
roi = np.array([[476, 162], [525, 222]])
froi = I0.pix[roi[0, 1]:roi[1, 1], roi[0, 0]:roi[1, 0]]

fall = np.zeros((froi.shape[0], froi.shape[1], 11))
fall[:, :, 0] = froi
for i in range(1, 11):
    It = px.Image(filename % i).Load()
    fall[:, :, i] = It.pix[roi[0, 1]:roi[1, 1], roi[0, 0]:roi[1, 0]]

plt.imshow(np.std(fall, axis=2))
plt.colorbar()

sigf = np.mean(np.std(fall, axis=2))

print(sigf)

# %% MESH
box = np.array([[0, 0], [90, 30]])
r = 5
cpos = np.array([45.1, 15.3])
lc = 3
lf = 3
m = px.OpenHolePlateUnstructured(box, r, cpos, lc, lf)
m.Plot()

# %% CALIBRATION OF THE CAMERA MODEL (ME)
ls = px.LSCalibrator(I0, m)
ls.NewCircle()
ls.NewLine()
ls.NewLine()
ls.FineTuning()
ls.Init3Pts()
cam = ls.Calibration()

# Plot the levelset functions
ls.Plot()

# Plot the iso-zero of the levelset functions
I0.Plot()
for i in range(3):
    plt.contour(ls.lvl[i].pix, [0.5], colors=["r"])

# and plot the FE mesh
n = m.n.copy()
u, v = cam.P(n[:, 0], n[:, 1])
m.Plot(n=np.c_[u, v], edgecolor="y", alpha=0.6)

# %% PRECALIBRATED CAMERA MODEL
cam = px.Camera(2)
p = np.array([-0.002176, 2.237317, 33.687119, 9.51087e-02])
cam.set_p(p)    
px.PlotMeshImage(I0, m, cam)

# %% EVALUATION OF THE A PRIORI UNCERTAINTY
npix = 300  # approximately
mm = 30
pix2m = mm / npix

pix2m * 1e-2

m.Plot()

# %% DIC RESOLUTION WITHOUT INITIALIZATION
m.Connectivity()
m.DICIntegration(cam)

U, res = px.Correlate(I0, It, m, cam)

# %% DIC RESOLUTION WITH INITIALIZATION

U0, UV0 = px.DISFlowInit(I0, It, m, cam)

U0 = px.MultiscaleInit(I0, It, m, cam, scales=[3, 2, 1])
U, res = px.Correlate(I0, It, m, cam, U0=U0)

# %%  Post-processing
# Visualization: Scaled deformation of the mesh
m.Plot(edgecolor='#CCCCCC')
m.Plot(U, 30)

# Visualization: displacement fields
m.PlotContourDispl(U, s=30)

# Visualization: strain fields
m.PlotContourStrain(U, clim=1, cmap='RdBu')

# Plot deformed Mesh on deformed state image
px.PlotMeshImage(It, m, cam, U)

px.PlotMeshImage(It, m, cam, U, plot='displ')

px.PlotMeshImage(It, m, cam, U, plot='strain')

   
# %% TIME RESOLUTION USING LINEAR PREDICTION

UU = np.zeros((m.ndof, 11))
for i in range(1, 11):
    It = px.Image(filename % i).Load()
    if i>1 : 
        U0 = 2*UU[:, i-1] - UU[:, i-2]
    else:
        U0 = UU[:, i-1]
    U, res = px.Correlate(I0, It, m, cam, U0=U0)
    UU[:, i] = U

m.AnimatedPlot(UU, s=30 )

# %% A PRIORI UNCERTAINTY QUANTIFICATION

dic = px.DICEngine()
H = dic.ComputeLHS(I0, m, cam)

Cov = np.linalg.inv(H.toarray())

diag = 2 * sigf * np.sqrt(np.diag(Cov))

m.PlotContourDispl(diag, s=0)

# %% PLAY WITH THE MESH SIZE



# %% GRAYLEVEL RESIDUAL OBSERVATION

# Initialization
emp = px.ExportPixMap(I0, m, cam)

# Get Residual on the pixel map
Rmap = emp.PlotResidual(I0, It, U)

# Get displacement on the pixel map
Umap, Vmap = emp.PlotDispl(U)