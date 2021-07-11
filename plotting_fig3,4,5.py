#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def loading():
    result_fig3 = np.load('data/Fig3_data.npy', allow_pickle=True).tolist()
    result_fig4_a = np.load('data/Fig4_a_data.npy', allow_pickle=True).tolist()
    result_fig4_b = np.load('data/Fig4_b_data.npy', allow_pickle=True).tolist()
    result_fig5 = np.load('data/Fig5_data.npy', allow_pickle=True).tolist()
    return result_fig3, result_fig4_a, result_fig4_b, result_fig5


def plotting_fig3(result_fig3):
    Lambda = result_fig3['lambda']
    plt.figure(figsize=(9, 6.5), dpi=200)
    plt.scatter(Lambda, result_fig3['py_e0.5_a0'], label=r'Spreader $\eta$=0.5, $\alpha$=0', marker='o', color=(0.1, 0.7, 0.9), edgecolor='black', linewidth=1, s=60)
    plt.scatter(Lambda, result_fig3['pz_e0.5_a0'], label=r'Stifler $\eta$=0.5, $\alpha$=0', marker='o', color='indigo', edgecolor='black', linewidth=0.5, s=60)
    plt.scatter(Lambda, result_fig3['py_e1_a0'], label=r'Spreader $\eta$=1, $\alpha$=0', marker='s', color=(0.1, 0.7, 0.9), edgecolor='black', linewidth=0.5, s=60)
    plt.scatter(Lambda, result_fig3['pz_e1_a0'], label=r'Stifler $\eta$=1, $\alpha$=0', marker='s', color='indigo', edgecolor='black', linewidth=0.5, s=60)
    plt.scatter(Lambda, result_fig3['py_e0.5_a1'], label=r'Spreader $\eta$=0.5, $\alpha$=1', marker='d', color=(0.1, 0.7, 0.9), edgecolor='black', linewidth=0.5, s=60)
    plt.scatter(Lambda, result_fig3['pz_e0.5_a1'], label=r'Stifler $\eta$=0.5, $\alpha$=1', marker='d', color='indigo', edgecolor='black', linewidth=0.5, s=60)
    plt.scatter(Lambda, result_fig3['py_e1_a1'], label=r'Spreader $\eta$=1, $\alpha$=1', marker='v', color=(0.1, 0.7, 0.9), edgecolor='black', linewidth=0.5, s=60)
    plt.scatter(Lambda, result_fig3['pz_e1_a1'], label=r'Stifler $\eta$=1, $\alpha$=1', marker='v', color='indigo', edgecolor='black', linewidth=0.5, s=60)
    plt.vlines(result_fig3['critical_values'], 0, 0.03, linestyles='--', colors='gray')
    plt.ylim(0, 0.03)
    plt.xlim(0, 0.09)
    plt.legend(prop={'size': 12})
    plt.gca().set_xlabel(r'$\lambda$')
    plt.gca().set_ylabel(r'$\rho$')
    plt.savefig('images/Fig3.jpg')
    plt.show()


def plotting_fig4_a(result_fig4_a):
    Lambda = result_fig4_a['lambda']
    plt.figure(figsize=(9, 6.5), dpi=200)
    plt.scatter(Lambda, result_fig4_a['py_beta0'], label=r'Spreader $\beta$=0', marker='o', color=(0.1, 0.7, 0.9), edgecolor='black', linewidth=1, s=60)
    plt.scatter(Lambda, result_fig4_a['pz_beta0'], label=r'Stifler $\beta$=0', marker='o', color='indigo', edgecolor='black', linewidth=1, s=60)
    plt.scatter(Lambda, result_fig4_a['py_beta1'], label=r'Spreader $\beta$=1', marker='s', color=(0.1, 0.7, 0.9), edgecolor='black', linewidth=1, s=60)
    plt.scatter(Lambda, result_fig4_a['pz_beta1'], label=r'Stifler $\beta$=1', marker='s', color='indigo', edgecolor='black', linewidth=1, s=60)
    plt.vlines(result_fig4_a['critical_values'], 0, 0.25, linestyles='--', colors='gray')
    plt.ylim(0, 0.25)
    plt.xlim(0, 1)
    plt.legend(prop={'size': 15})
    plt.gca().set_xlabel(r'$\lambda$')
    plt.gca().set_ylabel(r'$\rho$')
    plt.savefig('images/Fig4_a.jpg')
    plt.show()


def plotting_fig4_b(result_fig4_b):
    Lambda = result_fig4_b['lambda']
    plt.figure(figsize=(9, 6.5), dpi=200)
    plt.scatter(Lambda, result_fig4_b['py_b0_a0'], label=r'Spreader $\beta$=0, $\alpha$=0', marker='o', color=(0.1, 0.7, 0.9), edgecolor='black', linewidth=1, s=60)
    plt.scatter(Lambda, result_fig4_b['pz_b0_a0'], label=r'Stifler $\beta$=0, $\alpha$=0', marker='o', color='indigo', edgecolor='black', linewidth=1, s=60)
    plt.scatter(Lambda, result_fig4_b['py_b1_a0'], label=r'Spreader $\beta$=1, $\alpha$=0', marker='s', color=(0.1, 0.7, 0.9), edgecolor='black', linewidth=1, s=60)
    plt.scatter(Lambda, result_fig4_b['pz_b1_a0'], label=r'Stifler $\beta$=1, $\alpha$=0', marker='s', color='indigo', edgecolor='black', linewidth=1, s=60)
    plt.scatter(Lambda, result_fig4_b['py_b0_a1'], label=r'Spreader $\beta$=0, $\alpha$=1', marker='*', color=(0.1, 0.7, 0.9), edgecolor='black', linewidth=1, s=60)
    plt.scatter(Lambda, result_fig4_b['pz_b0_a1'], label=r'Stifler $\beta$=0, $\alpha$=1', marker='*', color='indigo', edgecolor='black', linewidth=1, s=60)
    plt.scatter(Lambda, result_fig4_b['py_b1_a1'], label=r'Spreader $\beta$=1, $\alpha$=1', marker='v', color=(0.1, 0.7, 0.9), edgecolor='black', linewidth=1, s=60)
    plt.scatter(Lambda, result_fig4_b['pz_b1_a1'], label=r'Stifler $\beta$=1, $\alpha$=1', marker='v', color='indigo', edgecolor='black', linewidth=1, s=60)
    plt.vlines(result_fig4_b['critical_values'], 0, 0.22, linestyles='--', colors='gray')
    plt.ylim(0, 0.22)
    plt.xlim(0, 0.5)
    plt.legend(prop={'size': 15})
    plt.gca().set_xlabel(r'$\lambda$')
    plt.gca().set_ylabel(r'$\rho$')
    plt.savefig('images/Fig4_b.jpg')
    plt.show()


def plotting_fig5(result_fig5):
    Beta = result_fig5['beta']
    plt.figure(figsize=(9, 5), dpi=200)
    plt.scatter(Beta, result_fig5['py_e0.01_a0'], label=r'Spreader $\eta$=0.01, $\alpha$=0', marker='o', color=(0.1, 0.7, 0.9), edgecolor='black', linewidth=1, s=60)
    plt.scatter(Beta, result_fig5['pz_e0.01_a0'], label=r'Stifler $\eta$=0.01, $\alpha$=0', marker='o', color='indigo', edgecolor='black', linewidth=1, s=60)
    plt.scatter(Beta, result_fig5['py_e0.5_a0'], label=r'Spreader $\eta$=0.5, $\alpha$=0', marker='s', color=(0.1, 0.7, 0.9), edgecolor='black', linewidth=1, s=60)
    plt.scatter(Beta, result_fig5['pz_e0.5_a0'], label=r'Stifler $\eta$=0.5, $\alpha$=0', marker='s', color='indigo', edgecolor='black', linewidth=1, s=60)
    plt.scatter(Beta, result_fig5['py_e0.01_a1'], label=r'Spreader $\eta$=0.01, $\alpha$=1', marker='*', color=(0.1, 0.7, 0.9), edgecolor='black', linewidth=1, s=60)
    plt.scatter(Beta, result_fig5['pz_e0.01_a1'], label=r'Stifler $\eta$=0.01, $\alpha$=1', marker='*', color='indigo', edgecolor='black', linewidth=1, s=60)
    plt.scatter(Beta, result_fig5['py_e0.5_a1'], label=r'Spreader $\eta$=0.5, $\alpha$=1', marker='v', color=(0.1, 0.7, 0.9), edgecolor='black', linewidth=1, s=60)
    plt.scatter(Beta, result_fig5['pz_e0.5_a1'], label=r'Stifler $\eta$=0.5, $\alpha$=1', marker='v', color='indigo', edgecolor='black', linewidth=1, s=60)
    plt.vlines(result_fig5['critical_values'], 0, 0.13, linestyles='--', colors='gray')
    plt.ylim(0, 0.13)
    plt.xlim(0, 0.5)
    plt.legend(prop={'size': 10})
    plt.gca().set_xlabel(r'$\beta$')
    plt.gca().set_ylabel(r'$\rho$')
    plt.savefig('images/Fig5.jpg')
    plt.show()


result_fig3, result_fig4_a, result_fig4_b, result_fig5 = loading()

plotting_fig3(result_fig3)
plotting_fig4_a(result_fig4_a)
plotting_fig4_b(result_fig4_b)
plotting_fig5(result_fig5)
