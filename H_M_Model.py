import numpy as np


class H_M_Model:
    MODE_RP = 1
    MODE_CP = 2

    def __init__(self, mode, delta_1, delta_2, lambda_, eta, alpha, gamma, beta):
        self.mode = mode
        self.params = {
            'delta_1': delta_1,
            'delta_2': delta_2,
            'lambda': lambda_,
            'eta': eta,
            'alpha': alpha,
            'gamma': gamma,
            'beta': beta
        }
        self.Px = 1
        self.Py = 0
        self.Pz = 0
        self.time = 0

    def a(self):
        if self.mode == H_M_Model.MODE_RP:
            return 0
        else:
            return np.exp(-self.params['lambda'] * (1 - self.params['delta_1'] - self.params['delta_2']) * self.Py)

    def b(self):
        if self.mode == H_M_Model.MODE_RP:
            return 0
        else:
            return 1 - self.params['alpha'] * (self.Py + self.Pz)

    def c(self):
        if self.mode == H_M_Model.MODE_RP:
            return 0
        else:
            return 1 - self.params['beta'] * self.Px

    def next_px_s(self, a, b, c):
        return self.Px * a + self.params['delta_1'] * self.Py + self.params['gamma'] * self.Pz

    def next_py_s(self, a, b, c):
        return self.params['eta'] * (1 - a) * self.Px + (1 - self.params['delta_1'] - self.params['delta_2']) * b \
               * self.Py + (1 - self.params['gamma']) * (1 - c) * self.Pz

    def next_pz_s(self, a, b, c):
        return (1 - self.params['eta']) * (1 - a) * self.Px + \
               ((1 - self.params['delta_1'] - self.params['delta_2']) * (1 - b) + self.params['delta_2']) * self.Py \
               + (1 - self.params['gamma']) * c * self.Pz

    def _time_step(self):
        a, b, c = self.a(), self.b(), self.c()
        self.Px, self.Py, self.Pz = self.next_px_s(a, b, c), self.next_py_s(a, b, c), self.next_pz_s(a, b, c)

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
        self.Px = p_x
        self.Py = p_y
        self.Pz = (1 - p_x - p_y)

    @staticmethod
    def _is_stable(px_s_pre, py_s_pre, pz_s_pre, px_s_new, py_s_new, pz_s_new, threshold):
        px_diff = np.abs(px_s_new - px_s_pre)
        py_diff = np.abs(py_s_new - py_s_pre)
        pz_diff = np.abs(pz_s_new - pz_s_pre)
        return np.alltrue(np.logical_and(np.logical_and(px_diff < threshold, py_diff < threshold), pz_diff < threshold))

    @staticmethod
    def _is_mean_stable(px_pre, py_pre, pz_pre, px_new, py_new, pz_new, threshold):
        px_diff = np.abs(np.mean(px_new) - np.mean(px_pre))
        py_diff = np.abs(np.mean(py_new) - np.mean(py_pre))
        pz_diff = np.abs(np.mean(pz_new) - np.mean(pz_pre))
        return np.alltrue(np.logical_and(np.logical_and(px_diff < threshold, py_diff < threshold), pz_diff < threshold))

    def go_to_steady_state(self, check_period=1, threshold=1e-3):
        is_stable = False

        while not is_stable:
            px_pre = self.Px
            py_pre = self.Py
            pz_pre = self.Pz

            self.render(check_period)

            px_new = self.Px
            py_new = self.Py
            pz_new = self.Pz
            is_stable = self._is_mean_stable(px_pre, py_pre, pz_pre, px_new, py_new, pz_new, threshold)
