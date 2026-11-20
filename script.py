import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hankel
from numpy.polynomial import Polynomial

def make_data(M, N, seed, e_1):
    """Randomly generates a exponential trigonometric signal for given of the form h_k=f(k)=sum_{j=-M}^{j=M} \rho_{j} \exp{i*w_j*k} sampled at k=0,1...2N.

    Args:
        M (int): number of positive frequencies from w_{-M}, w_{-(M-1),...w_{M-1}, w_M}
        N (int): related to the number of samples taken, 2 * N + 1
        seed (int): testing parameter for reproducibility
        e_1 (float): the noise bound, |e_{j}| <= e_1 for all j
    
    Returns:
        w: randomly generated frequencies 0 = w_0 <= w_1 <= w_2 <= ... <= w_M 
        rho: signal coefficients
        h: signal samples
        h_perturbed: signal samples with noise
    """
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

def build_hankel(h, N, L):
    return hankel(h[0:2 * N - L + 1], h[2 * N - L:2 * N + 1]) 
    # The function hankel(A, B) takes in two vectors. 
    # A is the first column of the desired Hankel matrix
    # B is the last row of the desired Hankel matrix
    
def main():
    M = 2
    N = 20
    L = 6
    seed = 1
    e_1 = 1e-6
    w, rho, h, h_perturbed = make_data(M, N, seed, e_1)
    print(f"w: {w}\n rho={rho}\n h={h}\n h_perturbed={h_perturbed}")

    H_perturbed = build_hankel(h_perturbed, N, L)
    print(f"H_perturbed shape: {H_perturbed.shape}")

if __name__ == '__main__':
    main()
