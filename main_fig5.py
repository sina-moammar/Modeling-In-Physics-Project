import numpy as np
from MF_Model import MF_Model
import networkx as nx


def beta_func(min_beta, max_beta, number):
    return np.linspace(min_beta, max_beta, number)


def load_graph():
    graph = nx.read_gpickle('networks/configuration_model_2.7_1000_100.gpickle')
    return graph


def params_func(alpha, eta, beta):
    params = {
        'delta_1': 0.2,
        'delta_2': 0.2,
        'lambda_': 1,
        'eta': eta,
        'alpha': alpha,
        'gamma': 0.2,
        'beta': beta
    }
    return params


def eta_alpha_func(eta1, eta2, alpha1, alpha2):
    eta_alpha_matrix = np.zeros((4, 2))
    eta_alpha_matrix[:, 0] = [eta1, eta2] * 2
    eta_alpha_matrix[:, 1] = [alpha1] * 2 + [alpha2] * 2
    return eta_alpha_matrix


def modeling(params):
    model = MF_Model(graph, MF_Model.MODE_CP, **params)
    model.set_initial_values(p_x_0, p_y_0)
    model.go_to_steady_state(threshold=5e-5, check_period=5)
    return np.mean(model.Py_s), np.mean(model.Pz_s)


def saving(mean_py_s, mean_pz_s, critical_values):
    data = {
        'py_e0.01_a0': mean_py_s[:, 0],
        'pz_e0.01_a0': mean_pz_s[:, 0],
        'py_e0.5_a0': mean_py_s[:, 1],
        'pz_e0.5_a0': mean_pz_s[:, 1],
        'py_e0.01_a1': mean_py_s[:, 2],
        'pz_e0.01_a1': mean_pz_s[:, 2],
        'py_e0.5_a1': mean_py_s[:, 3],
        'pz_e0.5_a1': mean_pz_s[:, 3],
        'beta': Beta,
        'critical_values': critical_values
    }
    np.save('data/Fig5_data.npy', data)  # saving


####initialization####
Beta = beta_func(0, 0.5, 100)
eta_alpha_matrix = eta_alpha_func(0.01, 0.5, 0, 1)
p_x_0 = 0.99
p_y_0 = 1 - p_x_0
graph = load_graph()
adj_matrix = nx.to_numpy_matrix(graph)
max_eig = np.sort(np.linalg.eigvals(adj_matrix))[-1]
eta_matrix = np.unique(eta_alpha_matrix[:, 0])
critical_values = (2 - 3 * eta_matrix) / 8
mean_py_s = np.zeros((len(Beta), len(eta_alpha_matrix)))
mean_pz_s = np.zeros((len(Beta), len(eta_alpha_matrix)))
######################

#####main#####
for index, temp in enumerate(eta_alpha_matrix):
    for index_prime, beta in enumerate(Beta):
        print(f'\r{index + 1}\t{index_prime}', end='')
        params = params_func(temp[1], temp[0], beta)
        mean_py_s[index_prime][index], mean_pz_s[index_prime][index] = modeling(params)
print('')
    ##############

#####saving#####
saving(mean_py_s, mean_pz_s, critical_values)
################
