import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hankel
from numpy.polynomial import Polynomial

def make_data(M, N, seed, e_1):
    rng = np.random.default_rng(seed=seed)
    w = np.array([0, 0.8, 2.0])
    rho_0 = rng.uniform(low = 0.5, high = 1.5)
    rho_pos = rng.uniform(low = 0.5, high = 1.5, size = M) + 1j * rng.uniform(low = 0.5, high = 1.5, size = M)
    rho = np.concatenate(([rho_0], rho_pos))
    err = rng.uniform(low = -e_1, high = e_1, size = 2 * N + 1)
    k = np.arange(2 * N + 1)
    h = rho_0 + 2 * np.real(rho_pos @ np.exp(1j * np.outer(w[1:], k)))
    h_perturbed = h + err
    return w, rho, h, h_perturbed


    pass

def main():
    M = 2
    N = 20
    L = 6
    seed = 1
    e_1 = 1e-6
    w, rho, h, h_perturbed = make_data(M, N, seed, e_1)
    print(f"w: {w}\n rho={rho}\n h={h}\n h_perturbed={h_perturbed}")

if __name__ == '__main__':
    main()
