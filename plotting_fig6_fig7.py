#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def loading():
    px_6 = np.load('data/fig6_a_px.npy')
    py_6 = np.load('data/fig6_b_py.npy')
    pz_6 = np.load('data/fig6_c_pz.npy')
    px_7 = np.load('data/fig7_a_px.npy')
    py_7 = np.load('data/fig7_b_py.npy')
    pz_7 = np.load('data/fig7_c_pz.npy')
    return px_6, py_6, pz_6, px_7, py_7, pz_7


px_6, py_6, pz_6, px_7, py_7, pz_7 = loading()
Lambda = np.linspace(0, 1, len(px_6))
Beta = np.linspace(0, 1, len(px_6[0]))
critical_val6 = (0.2 - 0.008 * Lambda) / (8 * Lambda - 0.9)
Lambda_c = Lambda[critical_val6 > 0]
critical_val6 = critical_val6[critical_val6 > 0]

#############################################################################
# plotting fig6
#############################################################################
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4), dpi=200)
fig.tight_layout(pad=3, w_pad=3)

im1 = ax1.imshow(px_6.T, cmap='jet', origin='lower', extent=[0, 1, 0, 1])
ax1.plot(Lambda_c, critical_val6, color='black')
ax1.set_ylim(0, 1)
ax1_divider = make_axes_locatable(ax1)
cax1 = ax1_divider.append_axes("right", size="7%", pad="3%")
cb1 = fig.colorbar(im1, cax=cax1)
ax1.set_title('Ignorants')
ax1.set_xlabel(r'$\lambda$')
ax1.set_ylabel(r'$\beta$')

im2 = ax2.imshow(py_6.T, cmap='jet', origin='lower', extent=[0, 1, 0, 1])
ax2.plot(Lambda_c, critical_val6, color='black')
ax2.set_ylim(0, 1)
ax2_divider = make_axes_locatable(ax2)
cax2 = ax2_divider.append_axes("right", size="7%", pad="3%")
cb2 = fig.colorbar(im2, cax=cax2)
ax2.set_title('Spreaders')
ax2.set_xlabel(r'$\lambda$')
ax2.set_ylabel(r'$\beta$')

im3 = ax3.imshow(pz_6.T, cmap='jet', origin='lower', extent=[0, 1, 0, 1])
ax3.plot(Lambda_c, critical_val6, color='black')
ax3.set_ylim(0, 1)
ax3_divider = make_axes_locatable(ax3)
cax3 = ax3_divider.append_axes("right", size="7%", pad="3%")
cb3 = fig.colorbar(im3, cax=cax3)
ax3.set_title('Stiflers')
ax3.set_xlabel(r'$\lambda$')
ax3.set_ylabel(r'$\beta$')
plt.savefig('images/Fig6.jpg')
plt.show()

#############################################################################
# plotting fig7
#############################################################################
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4), dpi=200)
fig.tight_layout(pad=3, w_pad=3)

im1 = ax1.imshow(px_7.T, cmap='jet', origin='lower', extent=[0, 1, 0, 1])
ax1_divider = make_axes_locatable(ax1)
cax1 = ax1_divider.append_axes("right", size="7%", pad="3%")
cb1 = fig.colorbar(im1, cax=cax1)
ax1.set_title('Ignorants')
ax1.set_xlabel(r'$\lambda$')
ax1.set_ylabel(r'$\beta$')

im2 = ax2.imshow(py_7.T, cmap='jet', origin='lower', extent=[0, 1, 0, 1])
ax2_divider = make_axes_locatable(ax2)
cax2 = ax2_divider.append_axes("right", size="7%", pad="3%")
cb2 = fig.colorbar(im2, cax=cax2)
ax2.set_title('Spreaders')
ax2.set_xlabel(r'$\lambda$')
ax2.set_ylabel(r'$\beta$')

im3 = ax3.imshow(pz_7.T, cmap='jet', origin='lower', extent=[0, 1, 0, 1])
ax3_divider = make_axes_locatable(ax3)
cax3 = ax3_divider.append_axes("right", size="7%", pad="3%")
cb3 = fig.colorbar(im3, cax=cax3)
ax3.set_title('Stiflers')
ax3.set_xlabel(r'$\lambda$')
ax3.set_ylabel(r'$\beta$')
plt.savefig('images/Fig7.jpg')
plt.show()
