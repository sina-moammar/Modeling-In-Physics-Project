import networkx as nx
import numpy as np
from MF_Model import MF_Model


def lambda_func(min_lambda, max_lambda, number):
    return np.linspace(min_lambda, max_lambda, number)


def load_graph():
    graph = nx.read_gpickle('networks/configuration_model_2.7_1000_100.gpickle')
    return graph


def params_func(alpha, beta, lambda_):
    params = {
        'delta_1': 0.15,
        'delta_2': 0.1,
        'lambda_': lambda_,
        'eta': 1,
        'alpha': alpha,
        'gamma': 0.15,
        'beta': beta
    }
    return params


def beta_alpha_func(beta1, beta2, alpha1, alpha2):
    beta_alpha_matrix = np.zeros((4, 2))
    beta_alpha_matrix[:, 0] = [beta1, beta2] * 2
    beta_alpha_matrix[:, 1] = [alpha1] * 2 + [alpha2] * 2
    return beta_alpha_matrix


def modeling(params):
    model = MF_Model(graph, MF_Model.MODE_CP, **params)
    model.set_initial_values(p_x_0, p_y_0)
    model.go_to_steady_state(threshold=5e-5, check_period=5)
    return np.mean(model.Py_s), np.mean(model.Pz_s)


def saving(mean_py_s, mean_pz_s, critical_values):
    data = {
        'py_b0_a0': mean_py_s[:, 0],
        'pz_b0_a0': mean_pz_s[:, 0],
        'py_b1_a0': mean_py_s[:, 1],
        'pz_b1_a0': mean_pz_s[:, 1],
        'py_b0_a1': mean_py_s[:, 2],
        'pz_b0_a1': mean_pz_s[:, 2],
        'py_b1_a1': mean_py_s[:, 3],
        'pz_b1_a1': mean_pz_s[:, 3],
        'lambda': Lambda,
        'critical_values': critical_values
    }
    np.save('data/Fig4_b_data.npy', data)  # saving


####initialization####
Lambda = lambda_func(0, 0.5, 101)
beta_alpha_matrix = beta_alpha_func(0, 1, 0, 1)
p_x_0 = 0.99
p_y_0 = 1 - p_x_0
graph = load_graph()
adj_matrix = nx.to_numpy_matrix(graph)
max_eig = np.sort(np.linalg.eigvals(adj_matrix))[-1]
beta_matrix = np.unique(beta_alpha_matrix[:, 0])
critical_values = (0.0375 + 0.1275 * beta_matrix) / (0.6375 * beta_matrix + 0.05625)
mean_py_s = np.zeros((len(Lambda), len(beta_alpha_matrix)))
mean_pz_s = np.zeros((len(Lambda), len(beta_alpha_matrix)))
######################

#####main#####
for index, temp in enumerate(beta_alpha_matrix):
    for index_prime, lambda_ in enumerate(Lambda):
        print(f'\r{index + 1}\t{index_prime}', end='')
        params = params_func(temp[1], temp[0], lambda_)
        mean_py_s[index_prime][index], mean_pz_s[index_prime][index] = modeling(params)
print('')
    ##############

#####saving#####
saving(mean_py_s, mean_pz_s, critical_values)
################
