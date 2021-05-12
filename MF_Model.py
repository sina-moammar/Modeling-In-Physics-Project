import numpy as np
import networkx as nx


class MF_Model:
    MODE_RP = 1
    MODE_CP = 2

    def __init__(self, graph, mode, delta_1, delta_2, lambda_, eta, alpha, gamma, beta):
        if graph is str:
            graph = nx.write_gpickle(graph)
        self.graph = graph
        self.degrees = np.array(nx.degree(graph))[:, 1]
        self.adj_matrix = np.array(nx.convert_matrix.to_numpy_matrix(self.graph))
        self.activity_matrix = self.adj_matrix if mode == MF_Model.MODE_RP else self.adj_matrix / self.degrees
        self.size = nx.number_of_nodes(self.graph)
        self.params = {
            'delta_1': delta_1,
            'delta_2': delta_2,
            'lambda': lambda_,
            'eta': eta,
            'alpha': alpha,
            'gamma': gamma,
            'beta': beta
        }
        self.Px_s = np.ones(self.size)
        self.Py_s = np.zeros(self.size)
        self.Pz_s = np.zeros(self.size)
        self.time = 0

    def a(self):
        return np.prod(1 - (self.params['lambda'] * (1 - self.params['delta_1'] - self.params['delta_2'])) *
                       (self.activity_matrix * self.Py_s), axis=0)

    def b(self):
        return np.prod(1 - self.params['alpha'] * self.activity_matrix * (self.Py_s + self.Pz_s), axis=0)

    def c(self):
        return np.prod(1 - self.params['beta'] * self.activity_matrix * self.Px_s, axis=0)

    def next_px_s(self):
        return self.Px_s * self.a() + self.params['delta_1'] * self.Py_s + self.params['gamma'] * self.Pz_s

    def next_py_s(self):
        return self.params['eta'] * (1 - self.a()) * self.Px_s + (1 - self.params['delta_1'] - self.params['delta_2'])\
               * self.b() * self.Py_s + (1 - self.params['gamma']) * (1 - self.c()) * self.Pz_s

    def next_pz_s(self):
        return (1 - self.params['eta']) * (1 - self.a()) * self.Px_s + \
               ((1 - self.params['delta_1'] - self.params['delta_2']) * (1 - self.b()) + self.params['delta_2'])\
               * self.Py_s + (1 - self.params['gamma']) * self.c() * self.Pz_s

    def _time_step(self):
        self.Px_s, self.Py_s, self.Pz_s = self.next_px_s(), self.next_py_s(), self.next_pz_s()

    def render(self, steps):
        for step in range(steps):
            self._time_step()
            self.time += 1

    def _render_iter(self, steps):
        for step in range(steps):
            self._time_step()
            self.time += 1
            yield self.time

    def set_initial_px_s(self, values):
        self.Px_s = np.ones(self.size) * values

    def set_initial_py_s(self, values):
        self.Py_s = np.ones(self.size) * values

    def set_initial_pz_s(self, values):
        self.Pz_s = np.ones(self.size) * values

    @staticmethod
    def _is_stable(px_s_pre, py_s_pre, pz_s_pre, px_s_new, py_s_new, pz_s_new, threshold):
        px_diff = np.abs(px_s_new - px_s_pre)
        py_diff = np.abs(py_s_new - py_s_pre)
        pz_diff = np.abs(pz_s_new - pz_s_pre)
        return np.alltrue(np.logical_and(np.logical_and(px_diff < threshold, py_diff < threshold), pz_diff < threshold))

    def go_to_steady_state(self, check_period=1, threshold=1e-3, sample_fraction=1):
        random_indexes = np.random.randint(0, self.size, int(sample_fraction * self.size))
        is_stable = False

        while not is_stable:
            px_s_pre = self.Px_s[random_indexes]
            py_s_pre = self.Py_s[random_indexes]
            pz_s_pre = self.Pz_s[random_indexes]

            self.render(check_period)

            px_s_new = self.Px_s[random_indexes]
            py_s_new = self.Py_s[random_indexes]
            pz_s_new = self.Pz_s[random_indexes]
            is_stable = self._is_stable(px_s_pre, py_s_pre, pz_s_pre, px_s_new, py_s_new, pz_s_new, threshold)

    def go_to_steady_state_iter(self, check_period=1, threshold=1e-3, sample_fraction=1):
        random_indexes = np.random.randint(0, self.size, int(sample_fraction * self.size))
        is_stable = False
        yield self.time

        while not is_stable:
            px_s_pre = self.Px_s[random_indexes]
            py_s_pre = self.Py_s[random_indexes]
            pz_s_pre = self.Pz_s[random_indexes]

            yield from self._render_iter(check_period)

            px_s_new = self.Px_s[random_indexes]
            py_s_new = self.Py_s[random_indexes]
            pz_s_new = self.Pz_s[random_indexes]
            is_stable = self._is_stable(px_s_pre, py_s_pre, pz_s_pre, px_s_new, py_s_new, pz_s_new, threshold)
