import numpy as np
import networkx as nx


class MarkovModel:
    MODE_RP = 1
    MODE_CP = 2

    def __init__(self, graph, mode, delta_1, delta_2, lambda_, eta, alpha, gamma, beta):
        if graph is str:
            graph = nx.read_gpickle(graph)
        self.graph = graph
        self.degrees = np.array(nx.degree(graph))[:, 1]
        self.adj_matrix = np.array(nx.convert_matrix.to_numpy_matrix(self.graph))
        self.mode = mode
        self.size = nx.number_of_nodes(self.graph)
        self.neighbors = self._get_neighbors()
        self.params = {
            'delta_1': delta_1,
            'delta_2': delta_2,
            'lambda': lambda_,
            'eta': eta,
            'alpha': alpha,
            'gamma': gamma,
            'beta': beta
        }
        self.X_s = np.ones(self.size, dtype=int)
        self.Y_s = np.zeros(self.size, dtype=int)
        self.Z_s = np.zeros(self.size, dtype=int)
        self.time = 0
        self._size_range = np.arange(self.size)

    def _get_neighbors(self):
        max_degree = max(self.degrees)
        neighbors = -1 * np.ones((self.size, max_degree), dtype=int)
        for node in range(self.size):
            neighbors[node][:self.degrees[node]] = list(nx.neighbors(self.graph, node))

        return neighbors

    def U_nb(self):
        nb_indexes = (np.random.rand(self.size) * self.degrees).astype(int)
        return self.neighbors[self._size_range, nb_indexes]

    def A(self, U, U_1, U_2, U_3, U_nb, I_gamma, I_lambda, I_alpha, I_beta, I_eta):
        if self.mode == MarkovModel.MODE_RP:
            return np.prod(1 - (I_lambda * self.adj_matrix * self.Y_s * U_3), axis=1)
        else:
            U_nb_matrix = (np.tile(U_nb, (self.size, 1)) == self._size_range[:, np.newaxis])
            return np.prod(1 - (I_lambda * U_nb_matrix * self.Y_s * U_3), axis=1)

    def B(self, U, U_1, U_2, U_3, U_nb, I_gamma, I_lambda, I_alpha, I_beta, I_eta):
        if self.mode == MarkovModel.MODE_RP:
            return np.prod(1 - I_alpha[:, np.newaxis] * self.adj_matrix * (self.Y_s + self.Z_s), axis=1)
        else:
            return 1 - I_alpha * (self.Y_s[U_nb] + self.Z_s[U_nb])

    def C(self, U, U_1, U_2, U_3, U_nb, I_gamma, I_lambda, I_alpha, I_beta, I_eta):
        if self.mode == MarkovModel.MODE_RP:
            return np.prod(1 - I_beta[:, np.newaxis] * self.adj_matrix * self.X_s, axis=1)
        else:
            return 1 - I_beta * self.X_s[U_nb]

    def next_px_s(self, A, B, C, U, U_1, U_2, U_3, U_nb, I_gamma, I_lambda, I_alpha, I_beta, I_eta):
        return self.X_s * A + self.Y_s * U_1 + self.Z_s * I_gamma

    def next_py_s(self, A, B, C, U, U_1, U_2, U_3, U_nb, I_gamma, I_lambda, I_alpha, I_beta, I_eta):
        return self.X_s * (1 - A) * I_eta + self.Y_s * U_3 * B + self.Z_s * (1 - I_gamma) * (1 - C)

    def next_pz_s(self, A, B, C, U, U_1, U_2, U_3, U_nb, I_gamma, I_lambda, I_alpha, I_beta, I_eta):
        return self.X_s * (1 - A) * (1 - I_eta) + self.Y_s * (U_3 * (1 - B) + U_2) + self.Z_s * (1 - I_gamma) * C

    def _time_step(self):
        U = np.random.rand(self.size)
        U_1 = U < self.params['delta_1']
        U_2 = np.logical_and(self.params['delta_1'] <= U, U < (self.params['delta_1'] + self.params['delta_2']))
        U_3 = U >= (self.params['delta_1'] + self.params['delta_2'])
        U_nb = self.U_nb() if self.mode == MarkovModel.MODE_CP else None
        I_gamma = np.random.binomial(1, self.params['gamma'], self.size)
        I_lambda = np.random.binomial(1, self.params['lambda'], (self.size, self.size))
        I_alpha = np.random.binomial(1, self.params['alpha'], self.size)
        I_beta = np.random.binomial(1, self.params['beta'], self.size)
        I_eta = np.random.binomial(1, self.params['eta'], self.size)
        randoms = {
            'U': U,
            'U_1': U_1,
            'U_2': U_2,
            'U_3': U_3,
            'U_nb': U_nb,
            'I_gamma': I_gamma,
            'I_lambda': I_lambda,
            'I_alpha': I_alpha,
            'I_beta': I_beta,
            'I_eta': I_eta
        }
        A, B, C = self.A(**randoms), self.B(**randoms), self.C(**randoms)
        self.X_s, self.Y_s, self.Z_s = self.next_px_s(A, B, C, **randoms), self.next_py_s(A, B, C, **randoms), self.next_pz_s(A, B, C, **randoms)

    def render(self, steps):
        for step in range(steps):
            self._time_step()
            self.time += 1

    def _render_iter(self, steps):
        for step in range(steps):
            self._time_step()
            self.time += 1
            yield self.time

    def set_initial_values(self, p_x, p_y):
        x_lim = int(self.size * p_x)
        y_lim = int(self.size * (p_x + p_y))
        indexes = np.arange(self.size)
        np.random.shuffle(indexes)

        self.X_s = np.zeros(self.size, dtype=int)
        self.X_s[indexes[:x_lim]] = 1
        self.Y_s = np.zeros(self.size, dtype=int)
        self.Y_s[indexes[x_lim:y_lim]] = 1
        self.Z_s = np.zeros(self.size, dtype=int)
        self.Z_s[indexes[y_lim:]] = 1

    @staticmethod
    def _is_stable(px_avg_s, py_avg_s, pz_avg_s, threshold):
        if len(px_avg_s) > 3:
            px_std = np.std(px_avg_s[int(0.3 * len(px_avg_s)):])
            py_std = np.std(px_avg_s[int(0.3 * len(py_avg_s)):])
            pz_std = np.std(px_avg_s[int(0.3 * len(pz_avg_s)):])
            return np.alltrue(np.logical_and(np.logical_and(px_std < threshold, py_std < threshold), pz_std < threshold))

        return False

    def go_to_steady_state(self, check_period=10, threshold=1e-2):
        is_stable = False
        px_avg_s = []
        py_avg_s = []
        pz_avg_s = []
        px_avg_s.append(np.mean(self.X_s))
        py_avg_s.append(np.mean(self.Y_s))
        pz_avg_s.append(np.mean(self.Z_s))

        while not is_stable:
            px_avg = py_avg = pz_avg = 0
            for step in range(check_period):
                self.render(check_period)
                px_avg += np.mean(self.X_s)
                py_avg += np.mean(self.Y_s)
                pz_avg += np.mean(self.Z_s)

            px_avg_s.append(px_avg / check_period)
            py_avg_s.append(py_avg / check_period)
            pz_avg_s.append(pz_avg / check_period)
            is_stable = self._is_stable(px_avg_s, py_avg_s, pz_avg_s, threshold)

    def go_to_steady_state_iter(self, check_period=10, threshold=1e-2):
        is_stable = False
        px_avg_s = []
        py_avg_s = []
        pz_avg_s = []
        px_avg_s.append(np.mean(self.X_s))
        py_avg_s.append(np.mean(self.Y_s))
        pz_avg_s.append(np.mean(self.Z_s))
        yield self.time

        while not is_stable:
            px_avg = py_avg = pz_avg = 0
            for step in range(check_period):
                yield from self._render_iter(1)
                px_avg += np.mean(self.X_s)
                py_avg += np.mean(self.Y_s)
                pz_avg += np.mean(self.Z_s)

            px_avg_s.append(px_avg / check_period)
            py_avg_s.append(py_avg / check_period)
            pz_avg_s.append(pz_avg / check_period)
            is_stable = self._is_stable(px_avg_s, py_avg_s, pz_avg_s, threshold)
