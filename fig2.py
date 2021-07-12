import numpy as np
import matplotlib.pyplot as plt
from H_M_Model import H_M_Model


def H_M_data(alpha_s, mode):
    H_M_px = np.zeros(len(alpha_s))
    H_M_py = np.zeros(len(alpha_s))
    H_M_pz = np.zeros(len(alpha_s))

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
        model = H_M_Model(H_M_Model.MODE_CP if mode == 'CP' else H_M_Model.MODE_RP, **params)
        model.set_initial_values(0.99, 0.01)
        model.go_to_steady_state(threshold=5e-5, check_period=5)
        H_M_px[i_alpha] = model.Px
        H_M_py[i_alpha] = model.Py
        H_M_pz[i_alpha] = model.Pz
    print('')
    data = {
        'alpha_s': alpha_s,
        'px_s': H_M_px,
        'py_s': H_M_py,
        'pz_s': H_M_pz,
    }
    np.save(f'data/Fig2_{mode}_data.npy', data)


def fig2_data():
    alpha_number = 101
    alpha_s = np.linspace(0, 1, alpha_number)
    H_M_data(alpha_s, 'CP')
    H_M_data(alpha_s, 'RP')


def fig2_plot():
    H_M_CP_data = np.load('data/Fig2_CP_data.npy', allow_pickle=True).tolist()
    H_M_RP_data = np.load('data/Fig2_RP_data.npy', allow_pickle=True).tolist()
    H_M_CP_px = H_M_CP_data['px_s']
    H_M_CP_py = H_M_CP_data['py_s']
    H_M_CP_pz = H_M_CP_data['pz_s']
    CP_alpha_s = H_M_CP_data['alpha_s']
    H_M_RP_px = H_M_RP_data['px_s']
    H_M_RP_py = H_M_RP_data['py_s']
    H_M_RP_pz = H_M_RP_data['pz_s']
    RP_alpha_s = H_M_RP_data['alpha_s']

    plt.figure(figsize=(9, 5), dpi=200)
    plt.plot(CP_alpha_s, H_M_CP_py, color=(0.1, 0.7, 0.9), label='Spreader (CP)')
    plt.plot(CP_alpha_s, H_M_CP_px, color='firebrick', label='Ignorant (CP)')
    plt.plot(CP_alpha_s, H_M_CP_pz, color='indigo', label='Stifler (CP)')
    plt.plot(RP_alpha_s, H_M_RP_py, linestyle='--', color=(0.1, 0.7, 0.9), label='Spreader (RP)')
    plt.plot(RP_alpha_s, H_M_RP_px, linestyle='--', color='firebrick', label='Ignorant (RP)')
    plt.plot(RP_alpha_s, H_M_RP_pz, linestyle='--', color='indigo', label='Stifler (RP)')
    plt.xlim(0, 1)
    plt.ylim(0, 0.65)
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\rho$')
    plt.legend(ncol=2, handleheight=2, labelspacing=0.05, prop={'size': 10})
    plt.savefig('images/Fig2.jpg')
    plt.show()


fig2_data()
fig2_plot()
