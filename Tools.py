import numpy as np
import networkx as nx


def power_law_configuration_model(N, gamma, k_mean, k_shift=0):
    k_range = range(N - 1, 0, -1)
    p_s = np.power(k_range, -gamma)
    p_cum_s = np.cumsum(p_s)[::-1]
    p_cum_s_norm = p_cum_s / p_cum_s[0]
    k_mean_s = np.cumsum(p_s * k_range)[::-1] / p_cum_s
    k_mean_index = np.argmin(np.abs(k_mean_s - (k_mean + k_shift)))

    k_s = np.zeros(N)
    r_s = np.random.rand(N) * p_cum_s_norm[k_mean_index]
    for i, r in enumerate(r_s):
        k_s[i] = np.where(r > p_cum_s_norm)[0][0]

    if np.sum(k_s) % 2 != 0:
        k_s[np.random.randint(0, len(k_s))] += 1
    graph = nx.configuration_model(k_s.astype(int))
    graph = nx.Graph(graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    print(f'k_mean = {np.mean(np.array(nx.degree(graph))[:, 1])}')
    nx.write_gpickle(graph, f'networks/configuration_model_{gamma}_{N}_{k_mean}.gpickle')


def power_law_HP_model(N, gamma, k_mean):
    alpha = 1 / (gamma - 1)
    eta_s = np.power(range(1, N + 1), -alpha)
    c = k_mean / np.mean(eta_s)
    eta_s *= c

    adj = np.zeros((N, N), dtype=bool)
    eta_tiled = np.tile(eta_s, (N, 1))
    adj_prob = eta_tiled * np.transpose(eta_tiled) / k_mean / N
    adj_prob = (adj_prob + np.transpose(adj_prob)) / 4
    np.fill_diagonal(adj_prob, 0)
    rands = np.random.rand(N, N)
    adj[rands < adj_prob] = 1

    graph = nx.convert_matrix.from_numpy_matrix(adj)
    print(f'k_mean = {np.mean(np.array(nx.degree(graph))[:, 1])}')
    nx.write_gpickle(graph, f'networks/HP_model_{gamma}_{N}_{k_mean}.gpickle')
