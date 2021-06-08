import networkx as nx
import numpy as np
from MF_Model import MF_Model
import pandas as pd


def lambda_func(min_lambda, max_lambda, number):
    return np.linspace(min_lambda, max_lambda, number)


def load_graph():
    graph = nx.read_gpickle('2.7_10000_10_new.gpickle')
    return graph


def params_func(alpha, eta, lambda_):
    params = {
        'delta_1': 0.25,
        'delta_2': 0.2,
        'lambda_': lambda_,
        'eta': eta,
        'alpha': alpha,
        'gamma': 0.25,
        'beta': 0
    }
    return params


def eta_alpha_func(eta1, eta2, alpha1, alpha2):
    eta_alpha_matrix = np.zeros((4, 2))
    eta_alpha_matrix[:, 0] = [eta1, eta2] * 2
    eta_alpha_matrix[:, 1] = [alpha1] * 2 + [alpha2] * 2
    return eta_alpha_matrix


def modeling(params):
    model = MF_Model(graph, MF_Model.MODE_RP, **params)
    model.set_initial_values(p_x_0, p_y_0)
    model.go_to_steady_state()
    stable_time = model.time
    # print(np.mean(model.Py_s),np.mean(model.Pz_s))
    return np.mean(model.Py_s), np.mean(model.Pz_s)


def saving(mean_py_s, mean_pz_s):
    df = pd.DataFrame(columns=['py_e0.5_a0', 'pz_e0.5_a0', 'py_e1_a0', 'pz_e1_a0',
                               'py_e0.5_a1', 'pz_e0.5_a1', 'py_e1_a1', 'pz_e1_a1'])
    # making a data frame from pandas
    data = {
        'py_e0.5_a0': mean_py_s[:, 0],
        'pz_e0.5_a0': mean_pz_s[:, 0],
        'py_e1_a0': mean_py_s[:, 1],
        'pz_e1_a0': mean_pz_s[:, 1],
        'py_e0.5_a1': mean_py_s[:, 2],
        'pz_e0.5_a1': mean_pz_s[:, 2],
        'py_e1_a1': mean_py_s[:, 3],
        'pz_e1_a1': mean_pz_s[:, 3]
    }
    final_df = df.append(data, ignore_index=True)  # append data
    final_df.to_json('Fig3_data.json')  # saving


####initialization####
Lambda = lambda_func(0, 0.09, 10)
eta_alpha_matrix = eta_alpha_func(0.5, 1, 0, 1)
p_x_0 = 0.99
p_y_0 = 1 - p_x_0
graph = load_graph()
mean_py_s = np.zeros((len(Lambda), len(eta_alpha_matrix)))
mean_pz_s = np.zeros((len(Lambda), len(eta_alpha_matrix)))
######################

#####main#####
for index, temp in enumerate(eta_alpha_matrix):
    for index_prime, lambda_ in enumerate(Lambda):
        params = params_func(temp[1], temp[0], lambda_)
        mean_py_s[index_prime][index], mean_pz_s[index_prime][index] = modeling(params)
    ##############

#####saving#####
saving(mean_py_s, mean_pz_s)
################
