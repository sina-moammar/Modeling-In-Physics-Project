
import numpy as np
from MF_Model import MF_Model
import networkx as nx
import pandas as pd

def beta_func(min_beta, max_beta, number):
    return np.linspace(min_beta, max_beta, number)

def load_graph():
    graph = nx.read_gpickle('2.7_10000_100_new.gpickle')
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
    eta_alpha_matrix = np.zeros((4,2))
    eta_alpha_matrix[:,0] = [eta1,eta2]*2
    eta_alpha_matrix[:,1] = [alpha1]*2 + [alpha2]*2
    return eta_alpha_matrix

def modeling(params):
    model = MF_Model.MF_Model(graph, MF_Model.MF_Model.MODE_CP, **params)
    model.set_initial_px_s(p_x_0)
    model.set_initial_py_s(p_y_0)
    model.set_initial_pz_s(p_z_0)
    model.go_to_steady_state()
    stable_time = model.time
    #print(np.mean(model.Py_s), np.mean(model.Pz_s))
    return np.mean(model.Py_s), np.mean(model.Pz_s)

def saving(mean_py_s, mean_pz_s):
    df = pd.DataFrame(columns = ['py_e0.01_a0', 'pz_e0.01_a0', 'py_e0.5_a0', 'pz_e0.5_a0',
                               'py_e0.01_a1', 'pz_e0.01_a1', 'py_e0.5_a1', 'pz_e0.5_a1']) 
                                # making a data frame from pandas
    data={
      'py_e0.01_a0': mean_py_s[:,0],
      'pz_e0.01_a0': mean_pz_s[:,0],
      'py_e0.5_a0': mean_py_s[:,1],
      'pz_e0.5_a0': mean_pz_s[:,1],
      'py_e0.01_a1': mean_py_s[:,2],
      'pz_e0.01_a1': mean_pz_s[:,2],
      'py_e0.5_a1': mean_py_s[:,3],
      'pz_e0.5_a1': mean_pz_s[:,3]
      }   
    final_df = df.append(data, ignore_index=True) # append data
    final_df.to_json('Fig5_data.json')  #saving




####initialization####
Beta = beta_func(0, 0.5, 20)
eta_alpha_matrix = eta_alpha_func(0.01, 0.5, 0, 1)
p_x_0 = 0.99  
p_z_0 = 0
p_y_0 = 1 - p_x_0
graph = load_graph()
mean_py_s = np.zeros((len(Beta), len(eta_alpha_matrix)))
mean_pz_s = np.zeros((len(Beta), len(eta_alpha_matrix)))
######################

#####main#####
for index, temp in enumerate(eta_alpha_matrix):
    for index_prime, beta in enumerate(Beta) :
        params = params_func(temp[1], temp[0], beta)   
        mean_py_s[index_prime][index], mean_pz_s[index_prime][index] = modeling(params) 
##############

#####saving#####
saving(mean_py_s, mean_pz_s) 
################
