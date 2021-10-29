import numpy as np
import scipy as sc
import matplotlib.pyplot as plt


def plot_partials():
    beta = 0.25
    beta_target = 0.1
    eps = 0.001
    t = 1.0
    N = 20
    h = 1 / N
    dF_dbeta = lambda x: -2 * np.pi * t * np.cos(2 * np.pi * (x - beta * t))
    adj = lambda x: h ** 2 * (
            np.sin(2 * np.pi * ((x + beta * t) - beta)) - np.sin(2 * np.pi * ((x + beta * t) - beta_target)))

    x_vec = np.linspace(0, 1, N)
    for t in np.linspace(0, 1, 20):
        plt.plot(x_vec, np.array(list(map(dF_dbeta, x_vec))), label='dF_dbeta t=' + str(t))

    plt.legend()
    plt.show()

    for t in np.linspace(0, 20, 5):
        lam0 = np.array(list(map(adj, x_vec)))
        plt.plot(x_vec, lam0, label='lam t=' + str(t))
    plt.legend()
    plt.show()


def controllability():
    beta = 0.25
    beta_target = 0.1
    eps = 0.001
    J_vec = []
    DJ_vec = []
    DJ_FD_vec = []
    N_vec = [20, 40, 100]  # 200, 400
    for N in N_vec:
        x_vec = np.linspace(0, 1, N)
        h = 1 / N
        print(h)
        u_k = np.zeros((N, N))
        u_k_eps1 = np.zeros((N, N))
        u_k_eps2 = np.zeros((N, N))
        u_target_k = np.zeros((N, N))
        du_dbeta = np.zeros((N, N))
        lam0 = np.zeros((N, N))
        adj = lambda x: h ** 2 * (
                np.sin(2 * np.pi * ((x + beta * t) - beta)) - np.sin(2 * np.pi * ((x + beta * t) - beta_target)))

        for j, y in enumerate(x_vec):
            t = 1.0
            u_k[j, :] = list(map(lambda x: np.sin(2 * np.pi * (x - beta * t)), x_vec))
            u_k_eps1[j, :] = list(map(lambda x: np.sin(2 * np.pi * (x - (beta + eps) * t)), x_vec))
            u_k_eps2[j, :] = list(map(lambda x: np.sin(2 * np.pi * (x - (beta - eps) * t)), x_vec))
            u_target_k[j, :] = list(map(lambda x: np.sin(2 * np.pi * (x - beta_target * t)), x_vec))
            du_dbeta[j, :] = list(map(lambda x: -2 * np.pi * t * np.cos(2 * np.pi * (x - beta * t)), x_vec))
            t = 0.0
            lam0[j, :] = list(map(adj, x_vec))

        # cost function
        J = h ** 2 * 0.5 * np.sum((u_k - u_target_k) ** 2, axis=(1, 0))
        diff = (u_k - u_target_k)
        dJ_dbeta = h ** 2 * np.dot(diff.flatten(), du_dbeta.flatten())

        # analytic gradient
        # 1. DJ_hat = h**2 * diff*du/dbeta
        DJ_hat_analytic = h ** 2 * np.dot(diff.flatten(), du_dbeta.flatten())
        print(DJ_hat_analytic)
        # adjoint gradient
        # (1. dFdu lam = dJdu)
        # 2. DJ = dJ/dbeta + lam*dF/dbeta
        DJ_adjoint = np.dot(lam0.flatten(), du_dbeta.flatten())

        # plt.plot(x_vec, lam0[0, :])
        # plt.show()

        # finite differences derivative of cost function
        J_eps1 = h ** 2 * 0.5 * np.sum((u_k_eps1 - u_target_k) ** 2, axis=(1, 0))
        J_eps2 = h ** 2 * 0.5 * np.sum((u_k_eps2 - u_target_k) ** 2, axis=(1, 0))
        DJ_fd = (J_eps1 - J_eps2) / (2 * eps)

        J_vec.append(J)
        DJ_vec.append(DJ_adjoint)
        DJ_FD_vec.append(DJ_fd)

    for i in range(0, len(J_vec)):
        print(N_vec[i], " & ", " $J_C$ ", " & ", J_vec[i], " & ", DJ_FD_vec[i], " & ", DJ_FD_vec[i], "\\\\")


def time_tracking():
    beta = 0.25
    beta_target = 0.1
    eps = 0.001
    J_vec = []
    DJ_vec = []
    DJ_FD_vec = []
    N_vec = [20, 40, 100]
    for N in N_vec:  #
        x_vec = np.linspace(0, 1, N)
        h = np.diff(x_vec)[0]

        J = 0.0
        DJ = 0.0
        DJ_fd = 0.0
        for t in np.linspace(0, 1, 20):
            u_k = np.zeros((N, N))
            u_k_eps1 = np.zeros((N, N))
            u_k_eps2 = np.zeros((N, N))
            u_target_k = np.zeros((N, N))
            du_dbeta = np.zeros((N, N))
            # t = 1.0

            for j, y in enumerate(x_vec):
                for i, x in enumerate(x_vec):
                    u_k[j, i] = np.sin(2 * np.pi * (x - beta * t))

                    u_k_eps1[j, i] = np.sin(2 * np.pi * (x - (beta + eps) * t))
                    u_k_eps2[j, i] = np.sin(2 * np.pi * (x - (beta - eps) * t))

                    u_target_k[j, i] = np.sin(2 * np.pi * (x - beta_target * t))
                    du_dbeta[j, i] = -2 * np.pi * t * np.cos(2 * np.pi * (x - beta * t))

            # matrix derivative
            dt = 1 / 20
            diff = (u_k - u_target_k)

            J += dt * h ** 2 * 0.5 * np.sum((u_k - u_target_k) ** 2, axis=(1, 0))
            J_eps1 = dt * h ** 2 * 0.5 * np.sum((u_k_eps1 - u_target_k) ** 2, axis=(1, 0))
            J_eps2 = dt * h ** 2 * 0.5 * np.sum((u_k_eps2 - u_target_k) ** 2, axis=(1, 0))

            dJ_dbeta = dt * h ** 2 * np.sum((u_k - u_target_k) * du_dbeta, axis=(1, 0))
            dJ_du = dt * h ** 2 * 0.5 * (u_k - u_target_k)
            DJ += dt * h ** 2 * np.dot(diff.flatten(), du_dbeta.flatten())

            if t == 1:
                J += h ** 2 * 0.5 * np.sum((u_k - u_target_k) ** 2, axis=(1, 0))
                J_eps1 += h ** 2 * 0.5 * np.sum((u_k_eps1 - u_target_k) ** 2, axis=(1, 0))
                J_eps2 += h ** 2 * 0.5 * np.sum((u_k_eps2 - u_target_k) ** 2, axis=(1, 0))

                dJ_du += h ** 2 * 0.5 * (u_k - u_target_k)
                dJ_dbeta += h ** 2 * np.sum((u_k - u_target_k) * du_dbeta, axis=(1, 0))

                diff = (u_k - u_target_k)
                DJ += h ** 2 * np.dot(diff.flatten(), du_dbeta.flatten())

            DJ_fd += (J_eps1 - J_eps2) / (2 * eps)

        J_vec.append(J)
        DJ_vec.append(DJ)
        DJ_FD_vec.append(DJ_fd)

    for i in range(0, len(J_vec)):
        print(N_vec[i], " & ", " $J_C$ ", " & ", J_vec[i], " & ", DJ_vec[i], " & ", DJ_FD_vec[i], "\\\\")


if __name__ == '__main__':
    controllability()
    time_tracking()
