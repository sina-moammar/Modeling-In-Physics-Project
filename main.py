import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from MF_Model import MF_Model
from MarkovModel import MarkovModel


size = 10**3
params = {
    'delta_1': 0.1,
    'delta_2': 0.1,
    'lambda_': 1,
    'eta': 1,
    'alpha': 0,
    'gamma': 0.1,
    'beta': 0.5
}
p_x_0 = 0.99
p_y_0 = 1 - p_x_0
graph = nx.erdos_renyi_graph(size, 0.3)

###### Example 1 ######
model = MF_Model(graph, MF_Model.MODE_CP, **params)
model.set_initial_px_s(p_x_0)
model.set_initial_py_s(p_y_0)
model.go_to_steady_state()
stable_time = model.time

###### Example 2 ######
model = MF_Model(graph, MF_Model.MODE_CP, **params)
model.set_initial_px_s(p_x_0)
model.set_initial_py_s(p_y_0)
mean_px_s = []
for time in model.go_to_steady_state_iter():
    mean_px_s.append(np.mean(model.Px_s))

plt.plot(mean_px_s)
plt.xlabel('time')
plt.ylabel(r'$\rho_x$')
plt.show()

##### Example 3 ######
model = MarkovModel(graph, MF_Model.MODE_RP, **params)
model.set_initial_values(p_x_0, p_y_0)
model.go_to_steady_state()
stable_time = model.time

##### Example 4 ######
model = MarkovModel(graph, MF_Model.MODE_CP, **params)
model.set_initial_values(p_x_0, p_y_0)
mean_px_s = []
for time in model.go_to_steady_state_iter():
    mean_px_s.append(np.mean(model.X_s))

plt.plot(mean_px_s)
plt.xlabel('time')
plt.ylabel(r'$\rho_x$')
plt.show()
