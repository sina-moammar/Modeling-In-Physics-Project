#!/usr/bin/env python3
import numpy as np
from MF_Model import MF_Model
import networkx as nx


def load_graphs():
    graph6 = nx.read_gpickle('networks/configuration_model_2.7_1000_100.gpickle')
    graph7 = nx.read_gpickle('networks/configuration_model_2.7_1000_10.gpickle')
    return graph6, graph7


def params_func(beta, lambda_, eta, gamma, alpha):
    params = {
        'delta_1': gamma,
        'delta_2': gamma,
        'lambda_': lambda_,
        'eta': eta,
        'alpha': alpha,
        'gamma': gamma,
        'beta': beta
    }
    return params


def modeling_fig7(params):
    model = MF_Model(graph7, MF_Model.MODE_RP, **params)
    model.set_initial_values(p_x_0, p_y_0)
    model.go_to_steady_state(threshold=5e-5, check_period=5)
    return np.mean(model.Px_s), np.mean(model.Py_s), np.mean(model.Pz_s)


def modeling_fig6(params):
    model = MF_Model(graph6, MF_Model.MODE_CP, **params)
    model.set_initial_values(p_x_0, p_y_0)
    model.go_to_steady_state(threshold=5e-5, check_period=5)
    return np.mean(model.Px_s), np.mean(model.Py_s), np.mean(model.Pz_s)


def saving(mean_px_6, mean_py_6, mean_pz_6,
           mean_px_7, mean_py_7, mean_pz_7):
    np.save('data/fig6_a_px.npy', mean_px_6)
    np.save('data/fig6_b_py.npy', mean_py_6)
    np.save('data/fig6_c_pz.npy', mean_pz_6)

    np.save('data/fig7_a_px.npy', mean_px_7)
    np.save('data/fig7_b_py.npy', mean_py_7)
    np.save('data/fig7_c_pz.npy', mean_pz_7)


####initialization####
Lambda = np.linspace(0, 1, 101)
Beta = np.linspace(0, 1, 101)
p_x_0 = 0.99
p_y_0 = 1 - p_x_0
graph6, graph7 = load_graphs()

mean_px_6 = np.zeros((len(Lambda), len(Beta)))
mean_py_6 = np.zeros((len(Lambda), len(Beta)))
mean_pz_6 = np.zeros((len(Lambda), len(Beta)))

mean_px_7 = np.zeros((len(Lambda), len(Beta)))
mean_py_7 = np.zeros((len(Lambda), len(Beta)))
mean_pz_7 = np.zeros((len(Lambda), len(Beta)))
######################


#####main#####
for index, beta in enumerate(Beta):
    for index_prime, lambda_ in enumerate(Lambda):
        print(f'\r{index + 1}\t{index_prime}', end='')
        params6 = params_func(beta, lambda_, 0.01, 0.1, 1)
        params7 = params_func(beta, lambda_, 0.01, 0.25, 1)

        mean_px_6[index_prime][index], mean_py_6[index_prime][index], mean_pz_6[index_prime][index] = modeling_fig6(
            params6)
        mean_px_7[index_prime][index], mean_py_7[index_prime][index], mean_pz_7[index_prime][index] = modeling_fig7(
            params7)
print('')
    ##############

#####saving#####
saving(mean_px_6, mean_py_6, mean_pz_6,
       mean_px_7, mean_py_7, mean_pz_7)
################
