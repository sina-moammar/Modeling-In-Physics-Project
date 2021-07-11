import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from MF_Model import MF_Model
from MarkovModel import MarkovModel


def a_MF_data(graph, alpha_s):
    MF_px = np.zeros(len(alpha_s))
    MF_py = np.zeros(len(alpha_s))
    MF_pz = np.zeros(len(alpha_s))

    for i_alpha, alpha in enumerate(alpha_s):
        print(f'\r{alpha}', end='')
        params = {
            'delta_1': 0.1,
            'delta_2': 0.1,
            'lambda_': 1,
            'eta': 1,
            'alpha': alpha,
            'gamma': 0.1,
            'beta': 0.5
        }
        model = MF_Model(graph, MF_Model.MODE_CP, **params)
        model.set_initial_values(0.99, 0.01)
        model.go_to_steady_state(threshold=5e-5, check_period=5)
        MF_px[i_alpha] = np.mean(model.Px_s)
        MF_py[i_alpha] = np.mean(model.Py_s)
        MF_pz[i_alpha] = np.mean(model.Pz_s)
    print('')
    data = {
        'alpha_s': alpha_s,
        'px_s': MF_px,
        'py_s': MF_py,
        'pz_s': MF_pz,
    }
    np.save('data/Fig8_a_MF_data.npy', data)


def a_Markov_data(graph, alpha_s, samples):
    markov_px_ensemble = np.zeros((samples, len(alpha_s)))
    markov_py_ensemble = np.zeros((samples, len(alpha_s)))
    markov_pz_ensemble = np.zeros((samples, len(alpha_s)))

    for sample in range(samples):
        for i_alpha, alpha in enumerate(alpha_s):
            print(f'\r{sample + 1}\t{alpha}', end='')
            params = {
                'delta_1': 0.1,
                'delta_2': 0.1,
                'lambda_': 1,
                'eta': 1,
                'alpha': alpha,
                'gamma': 0.1,
                'beta': 0.5
            }
            model = MarkovModel(graph, MF_Model.MODE_CP, **params)
            model.set_initial_values(0.99, 0.01)
            model.render(100)
            markov_px_ensemble[sample, i_alpha] = np.mean(model.X_s)
            markov_py_ensemble[sample, i_alpha] = np.mean(model.Y_s)
            markov_pz_ensemble[sample, i_alpha] = np.mean(model.Z_s)
    print('')
    data = {
        'alpha_s': alpha_s,
        'samples': samples,
        'px_s': markov_px_ensemble,
        'py_s': markov_py_ensemble,
        'pz_s': markov_pz_ensemble,
    }
    np.save('data/Fig8_a_Markov_data.npy', data)


def b_MF_data(graph, alpha_s):
    MF_px = np.zeros(len(alpha_s))
    MF_py = np.zeros(len(alpha_s))
    MF_pz = np.zeros(len(alpha_s))

    for i_alpha, alpha in enumerate(alpha_s):
        print(f'\r{alpha}', end='')
        params = {
            'delta_1': 0.1,
            'delta_2': 0.1,
            'lambda_': 1,
            'eta': 1,
            'alpha': alpha,
            'gamma': 0.1,
            'beta': 0.5
        }
        model = MF_Model(graph, MF_Model.MODE_RP, **params)
        model.set_initial_values(0.99, 0.01)
        model.go_to_steady_state(threshold=5e-5, check_period=5)
        MF_px[i_alpha] = np.mean(model.Px_s)
        MF_py[i_alpha] = np.mean(model.Py_s)
        MF_pz[i_alpha] = np.mean(model.Pz_s)
    print('')
    data = {
        'alpha_s': alpha_s,
        'px_s': MF_px,
        'py_s': MF_py,
        'pz_s': MF_pz,
    }
    np.save('data/Fig8_b_MF_data.npy', data)


def b_Markov_data(graph, alpha_s, samples):
    markov_px_ensemble = np.zeros((samples, len(alpha_s)))
    markov_py_ensemble = np.zeros((samples, len(alpha_s)))
    markov_pz_ensemble = np.zeros((samples, len(alpha_s)))

    for sample in range(samples):
        for i_alpha, alpha in enumerate(alpha_s):
            print(f'\r{sample + 1}\t{alpha}', end='')
            params = {
                'delta_1': 0.1,
                'delta_2': 0.1,
                'lambda_': 1,
                'eta': 1,
                'alpha': alpha,
                'gamma': 0.1,
                'beta': 0.5
            }
            model = MarkovModel(graph, MF_Model.MODE_RP, **params)
            model.set_initial_values(0.99, 0.01)
            model.render(50)
            markov_px_ensemble[sample, i_alpha] = np.mean(model.X_s)
            markov_py_ensemble[sample, i_alpha] = np.mean(model.Y_s)
            markov_pz_ensemble[sample, i_alpha] = np.mean(model.Z_s)
    print('')
    data = {
        'alpha_s': alpha_s,
        'samples': samples,
        'px_s': markov_px_ensemble,
        'py_s': markov_py_ensemble,
        'pz_s': markov_pz_ensemble,
    }
    np.save('data/Fig8_b_Markov_data.npy', data)


def fig8_data():
    alpha_number = 21
    micro_res = 5
    samples = 100
    micro_alpha_number = micro_res * (alpha_number - 1) + 1
    graph_a = nx.read_gpickle('networks/configuration_model_2.7_1000_100.gpickle')
    graph_b = nx.read_gpickle('networks/configuration_model_2.7_1000_10.gpickle')
    alpha_s = np.linspace(0, 1, alpha_number)
    micro_alpha_s = np.linspace(0, 1, micro_alpha_number)

    a_MF_data(graph_a, micro_alpha_s)
    a_Markov_data(graph_a, alpha_s, samples)

    b_MF_data(graph_b, micro_alpha_s)
    b_Markov_data(graph_b, alpha_s, samples)


def a_plot():
    MF_data = np.load('data/Fig8_a_MF_data.npy', allow_pickle=True).tolist()
    Markov_data = np.load('data/Fig8_a_Markov_data.npy', allow_pickle=True).tolist()
    MF_px = MF_data['px_s']
    MF_py = MF_data['py_s']
    MF_pz = MF_data['pz_s']
    micro_alpha_s = MF_data['alpha_s']
    Markov_px_ensemble = Markov_data['px_s']
    Markov_py_ensemble = Markov_data['py_s']
    Markov_pz_ensemble = Markov_data['pz_s']
    alpha_s = Markov_data['alpha_s']
    samples = Markov_data['samples']

    Markov_px_avg = np.mean(Markov_px_ensemble, axis=0)
    Markov_px_err = np.std(Markov_px_ensemble, axis=0) / np.sqrt(samples - 1)
    Markov_py_avg = np.mean(Markov_py_ensemble, axis=0)
    Markov_py_err = np.std(Markov_py_ensemble, axis=0) / np.sqrt(samples - 1)
    Markov_pz_avg = np.mean(Markov_pz_ensemble, axis=0)
    Markov_pz_err = np.std(Markov_pz_ensemble, axis=0) / np.sqrt(samples - 1)

    plt.figure(figsize=(9, 6.5), dpi=200)
    plt.plot(micro_alpha_s, MF_px, color='firebrick')
    plt.plot(micro_alpha_s, MF_pz, color='indigo')
    plt.plot(micro_alpha_s, MF_py, color=(0.1, 0.7, 0.9))
    plt.errorbar(alpha_s, Markov_py_avg, yerr=Markov_py_err, linestyle='', marker='o', color=(0.1, 0.7, 0.9), label='Spreader')
    plt.scatter(alpha_s, Markov_py_avg, marker='o', color=(0.1, 0.7, 0.9), edgecolor='black', linewidth=1, s=60)
    plt.errorbar(alpha_s, Markov_px_avg, yerr=Markov_px_err, linestyle='', marker='d', color='firebrick', label='Ignorant')
    plt.scatter(alpha_s, Markov_px_avg, marker='d', color='firebrick', edgecolor='black', linewidth=1, s=60)
    plt.errorbar(alpha_s, Markov_pz_avg, yerr=Markov_pz_err, linestyle='', marker='s', color='indigo', label='Stifler')
    plt.scatter(alpha_s, Markov_pz_avg, marker='s', color='indigo', edgecolor='black', linewidth=1, s=60)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\rho$')
    plt.legend(prop={'size': 15})
    plt.savefig('images/Fig8_a.jpg')
    plt.show()


def b_plot():
    MF_data = np.load('data/Fig8_b_MF_data.npy', allow_pickle=True).tolist()
    Markov_data = np.load('data/Fig8_b_Markov_data.npy', allow_pickle=True).tolist()
    MF_px = MF_data['px_s']
    MF_py = MF_data['py_s']
    MF_pz = MF_data['pz_s']
    micro_alpha_s = MF_data['alpha_s']
    Markov_px_ensemble = Markov_data['px_s']
    Markov_py_ensemble = Markov_data['py_s']
    Markov_pz_ensemble = Markov_data['pz_s']
    alpha_s = Markov_data['alpha_s']
    samples = Markov_data['samples']

    Markov_px_avg = np.mean(Markov_px_ensemble, axis=0)
    Markov_px_err = np.std(Markov_px_ensemble, axis=0) / np.sqrt(samples - 1)
    Markov_py_avg = np.mean(Markov_py_ensemble, axis=0)
    Markov_py_err = np.std(Markov_py_ensemble, axis=0) / np.sqrt(samples - 1)
    Markov_pz_avg = np.mean(Markov_pz_ensemble, axis=0)
    Markov_pz_err = np.std(Markov_pz_ensemble, axis=0) / np.sqrt(samples - 1)

    plt.figure(figsize=(9, 6.5), dpi=200)
    plt.plot(micro_alpha_s, MF_px, color='firebrick')
    plt.plot(micro_alpha_s, MF_pz, color='indigo')
    plt.plot(micro_alpha_s, MF_py, color=(0.1, 0.7, 0.9))
    plt.errorbar(alpha_s, Markov_py_avg, yerr=Markov_py_err, linestyle='', marker='o', color=(0.1, 0.7, 0.9), label='Spreader')
    plt.scatter(alpha_s, Markov_py_avg, marker='o', color=(0.1, 0.7, 0.9), edgecolor='black', linewidth=1, s=60)
    plt.errorbar(alpha_s, Markov_px_avg, yerr=Markov_px_err, linestyle='', marker='d', color='firebrick', label='Ignorant')
    plt.scatter(alpha_s, Markov_px_avg, marker='d', color='firebrick', edgecolor='black', linewidth=1, s=60)
    plt.errorbar(alpha_s, Markov_pz_avg, yerr=Markov_pz_err, linestyle='', marker='s', color='indigo', label='Stifler')
    plt.scatter(alpha_s, Markov_pz_avg, marker='s', color='indigo', edgecolor='black', linewidth=1, s=60)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\rho$')
    plt.legend(prop={'size': 15})
    plt.savefig('images/Fig8_b.jpg')
    plt.show()


fig8_data()
a_plot()
b_plot()
