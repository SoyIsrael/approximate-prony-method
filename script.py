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
    e_2 = 1e-5
    e_4 = 1e-6
    w, rho, h, h_perturbed = make_data(M, N, seed, e_1)

    H = build_hankel(h, N, L)
    H_perturbed = build_hankel(h_perturbed, N, L)

    V_perturbed, S_perturbed, Uh_perturbed = np.linalg.svd(H_perturbed, full_matrices=False)
    V, S, Uh = np.linalg.svd(H, full_matrices=False)

    idx = [i for i, s in enumerate(S_perturbed) if 0 < s <= e_2]

    smallest_sing_value = S_perturbed[idx[-1]]
    smallest_right_sing_vector = Uh_perturbed[idx[-1]].conj()
    # smallest_left_sing_vector = V_perturbed[:, idx[-1]]

    second_smallest_sing_value = S_perturbed[idx[-2]]
    second_smallest_right_sing_vector = Uh_perturbed[idx[-2]].conj()
    # second_smallest_left_sing_vector = V_perturbed[:, idx[-2]]


    P_1 = Polynomial(smallest_right_sing_vector)
    roots_P_1 = P_1.roots()
    print(roots_P_1)

    valid_roots_P_1 = []
    for root in roots_P_1:
        modulus = np.abs(root)
        angle = np.angle(root)
        if np.abs(modulus - 1) <= e_4 and 0 < angle < np.pi:
            valid_roots_P_1.append(root)

    P_2 = Polynomial(second_smallest_right_sing_vector)
    roots_P_2 = P_2.roots()
    print(roots_P_2)

    valid_roots_P_2 = []
    for root in roots_P_2:
        modulus = np.abs(root)
        angle = np.angle(root)
        if np.abs(modulus - 1) <= e_4 and 0 < angle < np.pi:
            valid_roots_P_2.append(root)

    print(valid_roots_P_2)

if __name__ == '__main__':
    main()
