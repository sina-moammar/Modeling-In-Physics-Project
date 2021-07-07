#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import genfromtxt


def loading():
    px_6 = genfromtxt('fig6_a_px.csv', delimiter=',')
    px_6 = np.delete(px_6, 0, 1)
    px_6 = np.delete(px_6, 0, 0)

    py_6 = genfromtxt('fig6_b_py.csv', delimiter=',')
    py_6 = np.delete(py_6, 0, 1)
    py_6 = np.delete(py_6, 0, 0)

    pz_6 = genfromtxt('fig6_c_pz.csv', delimiter=',')
    pz_6 = np.delete(pz_6, 0, 1)
    pz_6 = np.delete(pz_6, 0, 0)

    px_7 = genfromtxt('fig7_a_px.csv', delimiter=',')
    px_7 = np.delete(px_7, 0, 1)
    px_7 = np.delete(px_7, 0, 0)

    py_7 = genfromtxt('fig7_b_py.csv', delimiter=',')
    py_7 = np.delete(py_7, 0, 1)
    py_7 = np.delete(py_7, 0, 0)

    pz_7 = genfromtxt('fig7_c_pz.csv', delimiter=',')
    pz_7 = np.delete(pz_7, 0, 1)
    pz_7 = np.delete(pz_7, 0, 0)
    return px_6, py_6, pz_6, px_7, py_7, pz_7


px_6, py_6, pz_6, px_7, py_7, pz_7 = loading()

#############################################################################
# plotting fig6
#############################################################################
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.subplots_adjust(wspace=0.5)
fig.suptitle("CP Mode")

im1 = ax1.imshow(px_6.T, cmap='jet', origin='lower', extent=[0, 1, 0, 1])
ax1_divider = make_axes_locatable(ax1)
cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
cb1 = fig.colorbar(im1, cax=cax1)
ax1.set_title('Ignorants')
ax1.set_xlabel(r'$\lambda$')
ax1.set_ylabel(r'$\beta$')

im2 = ax2.imshow(py_6.T, cmap='jet', origin='lower', extent=[0, 1, 0, 1])
ax2_divider = make_axes_locatable(ax2)
cax2 = ax2_divider.append_axes("right", size="7%", pad="2%")
cb2 = fig.colorbar(im2, cax=cax2)
ax2.set_title('Spreaders')
ax2.set_xlabel(r'$\lambda$')
ax2.set_ylabel(r'$\beta$')

im3 = ax3.imshow(pz_6.T, cmap='jet', origin='lower', extent=[0, 1, 0, 1])
ax3_divider = make_axes_locatable(ax3)
cax3 = ax3_divider.append_axes("right", size="7%", pad="2%")
cb3 = fig.colorbar(im3, cax=cax3)
ax3.set_title('Stiflers')
ax3.set_xlabel(r'$\lambda$')
ax3.set_ylabel(r'$\beta$')

#############################################################################
# plotting fig7
#############################################################################
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.subplots_adjust(wspace=0.5)
fig.suptitle("RP Mode")

im1 = ax1.imshow(px_7.T, cmap='jet', extent=[0, 1, 0, 1])
ax1_divider = make_axes_locatable(ax1)
cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
cb1 = fig.colorbar(im1, cax=cax1)
ax1.set_title('Ignorants')
ax1.set_xlabel(r'$\lambda$')
ax1.set_ylabel(r'$\beta$')

im2 = ax2.imshow(py_7.T, cmap='jet', origin='lower', extent=[0, 1, 0, 1])
ax2_divider = make_axes_locatable(ax2)
cax2 = ax2_divider.append_axes("right", size="7%", pad="2%")
cb2 = fig.colorbar(im2, cax=cax2)
ax2.set_title('Spreaders')
ax2.set_xlabel(r'$\lambda$')
ax2.set_ylabel(r'$\beta$')

im3 = ax3.imshow(pz_7.T, cmap='jet', origin='lower', extent=[0, 1, 0, 1])
ax3_divider = make_axes_locatable(ax3)
cax3 = ax3_divider.append_axes("right", size="7%", pad="2%")
cb3 = fig.colorbar(im3, cax=cax3)
ax3.set_title('Stiflers')
ax3.set_xlabel(r'$\lambda$')
ax3.set_ylabel(r'$\beta$')

plt.show()
