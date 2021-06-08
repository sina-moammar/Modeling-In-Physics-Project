#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def loading():
    result_fig3 = pd.read_json('Fig3_data.json')
    result_fig4_a = pd.read_json('Fig4_a_data.json')
    result_fig4_b = pd.read_json('Fig4_b_data.json')
    result_fig5 = pd.read_json('Fig5_data.json')
    return result_fig3, result_fig4_a, result_fig4_b, result_fig5


def plotting_fig3(result_fig3):
    Lambda = np.linspace(0, 0.09, 10)
    ax1.plot(Lambda, result_fig3['py_e0.5_a0'][0], label=r'Spreader $\eta$=0.5, $\alpha$=0')
    ax1.plot(Lambda, result_fig3['pz_e0.5_a0'][0], label=r'Stifler $\eta$=0.5, $\alpha$=0')
    ax1.plot(Lambda, result_fig3['py_e1_a0'][0], label=r'Spreader $\eta$=1, $\alpha$=0')
    ax1.plot(Lambda, result_fig3['pz_e1_a0'][0], label=r'Stifler $\eta$=1, $\alpha$=0')
    ax1.plot(Lambda, result_fig3['py_e0.5_a1'][0], label=r'Spreader $\eta$=0.5, $\alpha$=1')
    ax1.plot(Lambda, result_fig3['pz_e0.5_a1'][0], label=r'Stifler $\eta$=0.5, $\alpha$=1')
    ax1.plot(Lambda, result_fig3['py_e1_a1'][0], label=r'Spreader $\eta$=1, $\alpha$=1')
    ax1.plot(Lambda, result_fig3['pz_e1_a1'][0], label=r'Stifler $\eta$=1, $\alpha$=1')
    ax1.legend()
    ax1.set_xlabel(r'$\lambda$')
    ax1.set_ylabel(r'$\rho$')
    ax1.set_title('Configuration')


def plotting_fig4_a(result_fig4_a):
    Lambda = np.linspace(0, 1, 20)
    ax2.plot(Lambda, result_fig4_a['py_beta0'][0], label=r'Spreader $\beta$=0')
    ax2.plot(Lambda, result_fig4_a['pz_beta0'][0], label=r'Stifler $\beta$=0')
    ax2.plot(Lambda, result_fig4_a['py_beta1'][0], label=r'Spreader $\beta$=1')
    ax2.plot(Lambda, result_fig4_a['pz_beta1'][0], label=r'Stifler $\beta$=1')
    ax2.legend()
    ax2.set_xlabel(r'$\lambda$')
    ax2.set_ylabel(r'$\rho$')
    ax2.set_title('Configuration Model')


def plotting_fig4_b(result_fig4_b):
    Lambda = np.linspace(0, 0.5, 30)
    ax3.plot(Lambda, result_fig4_b['py_b0_a0'][0], label=r'Spreader $\beta$=0, $\alpha$=0')
    ax3.plot(Lambda, result_fig4_b['pz_b0_a0'][0], label=r'Stifler $\beta$=0, $\alpha$=0')
    ax3.plot(Lambda, result_fig4_b['py_b1_a0'][0], label=r'Spreader $\beta$=1, $\alpha$=0')
    ax3.plot(Lambda, result_fig4_b['pz_b1_a0'][0], label=r'Stifler $\beta$=1, $\alpha$=0')
    ax3.plot(Lambda, result_fig4_b['py_b0_a1'][0], label=r'Spreader $\beta$=0, $\alpha$=1')
    ax3.plot(Lambda, result_fig4_b['pz_b0_a1'][0], label=r'Stifler $\beta$=0, $\alpha$=1')
    ax3.plot(Lambda, result_fig4_b['py_b1_a1'][0], label=r'Spreader $\beta$=1, $\alpha$=1')
    ax3.plot(Lambda, result_fig4_b['pz_b1_a1'][0], label=r'Stifler $\beta$=1, $\alpha$=1')
    ax3.legend()
    ax3.set_xlabel(r'$\lambda$')
    ax3.set_ylabel(r'$\rho$')
    ax3.set_title('Configuration')


def plotting_fig5(result_fig5):
    Beta = np.linspace(0, 0.5, 20)
    ax4.plot(Beta, result_fig5['py_e0.01_a0'][0], label=r'Spreader $\eta$=0.01, $\alpha$=0')
    ax4.plot(Beta, result_fig5['pz_e0.01_a0'][0], label=r'Stifler $\eta$=0.01, $\alpha$=0')
    ax4.plot(Beta, result_fig5['py_e0.5_a0'][0], label=r'Spreader $\eta$=0.5, $\alpha$=0')
    ax4.plot(Beta, result_fig5['pz_e0.5_a0'][0], label=r'Stifler $\eta$=0.5, $\alpha$=0')
    ax4.plot(Beta, result_fig5['py_e0.01_a1'][0], label=r'Spreader $\eta$=0.01, $\alpha$=1')
    ax4.plot(Beta, result_fig5['pz_e0.01_a1'][0], label=r'Stifler $\eta$=0.01, $\alpha$=1')
    ax4.plot(Beta, result_fig5['py_e0.5_a1'][0], label=r'Spreader $\eta$=0.5, $\alpha$=1')
    ax4.plot(Beta, result_fig5['pz_e0.5_a1'][0], label=r'Stifler $\eta$=0.5, $\alpha$=1')
    ax4.legend()
    ax4.set_xlabel(r'$\beta$')
    ax4.set_ylabel(r'$\rho$')
    ax4.set_title('Configuration Model')


result_fig3, result_fig4_a, result_fig4_b, result_fig5 = loading()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
plotting_fig3(result_fig3)
plotting_fig4_a(result_fig4_a)
plotting_fig4_b(result_fig4_b)
plotting_fig5(result_fig5)
