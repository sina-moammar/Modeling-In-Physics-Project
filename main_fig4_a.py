import networkx as nx
import numpy as np
from MF_Model import MF_Model
import pandas as pd


def lambda_func(min_lambda, max_lambda, number):
    return np.linspace(min_lambda, max_lambda, number)


def load_graph():
    graph = nx.read_gpickle('2.7_10000_100_new.gpickle')
    return graph


def params_func(beta, lambda_):
    params = {
        'delta_1': 0.15,
        'delta_2': 0.1,
        'lambda_': lambda_,
        'eta': 0.5,
        'alpha': 1,
        'gamma': 0.15,
        'beta': beta
    }
    return params


def beta_func(beta1, beta2):
    beta_matrix = np.array([beta1, beta2])
    return beta_matrix


def modeling(params):
    model = MF_Model(graph, MF_Model.MODE_CP, **params)
    model.set_initial_values(p_x_0, p_y_0)
    model.go_to_steady_state()
    stable_time = model.time
    # print(np.mean(model.Py_s),np.mean(model.Pz_s))
    return np.mean(model.Py_s), np.mean(model.Pz_s)


def saving(mean_py_s, mean_pz_s):
    df = pd.DataFrame(columns=['py_beta0', 'pz_beta0', 'py_beta1', 'pz_beta1'])  # making a data frame from pandas
    data = {
        'py_beta0': mean_py_s[:, 0],
        'pz_beta0': mean_pz_s[:, 0],
        'py_beta1': mean_py_s[:, 1],
        'pz_beta1': mean_pz_s[:, 1]
    }
    final_df = df.append(data, ignore_index=True)  # append data
    final_df.to_json('Fig4_a_data.json')  # saving


####initialization####
Lambda = lambda_func(0, 1, 20)
beta_matrix = beta_func(0, 1)
p_x_0 = 0.99
p_y_0 = 1 - p_x_0
graph = load_graph()
mean_py_s = np.zeros((len(Lambda), len(beta_matrix)))
mean_pz_s = np.zeros((len(Lambda), len(beta_matrix)))
######################

#####main#####
for index, beta in enumerate(beta_matrix):
    for index_prime, lambda_ in enumerate(Lambda):
        params = params_func(beta, lambda_)
        mean_py_s[index_prime][index], mean_pz_s[index_prime][index] = modeling(params)
    ##############

#####saving#####
saving(mean_py_s, mean_pz_s)
################
