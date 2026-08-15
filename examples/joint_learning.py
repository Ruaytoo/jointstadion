from functools import partial

import numpy as onp
from jax import random, numpy as jnp

from stadion.joint import (JointLinearSDE, assemble_quadratic, solve_exact,
                           fit_joint, rbf_kernel, median_heuristic_bandwidth,
                           ou_stationary_moments, gaussian_w2)


if __name__ == "__main__":

    rng = onp.random.default_rng(0)
    n, d = 1000, 4

    # ground truth: reversible OU, A* = -Lam Theta, Q* = 2 Lam, beta* = -A* m0
    theta = onp.array([[1.0, 0.4, 0.0, 0.0],
                       [0.4, 1.0, -0.3, 0.0],
                       [0.0, -0.3, 1.0, 0.2],
                       [0.0, 0.0, 0.2, 1.0]])
    sigma = onp.linalg.inv(theta)
    lam = onp.diag(onp.array([0.6, 1.0, 0.8, 1.2]))
    A_true = -lam @ theta
    m0 = onp.array([0.5, -1.0, 0.0, 1.0])
    beta_true = -A_true @ m0

    # stationary data: observational + two shift interventions
    shifts_true = [onp.zeros(d), 2.0 * onp.eye(d)[1], -2.0 * onp.eye(d)[3]]
    targets = [onp.zeros(d), onp.eye(d)[1], onp.eye(d)[3]]
    xs = []
    for s in shifts_true:
        mean = -onp.linalg.solve(A_true, beta_true + s)
        xs.append(jnp.asarray(rng.multivariate_normal(mean, sigma, size=n)))

    kernel = partial(rbf_kernel, bandwidth=median_heuristic_bandwidth(xs))
    model = JointLinearSDE()

    # option 1: exact solve of the quadratic objective (linear model class)
    H, pack, unpack, layout = assemble_quadratic(model, kernel, xs, onp.stack(targets))
    W, info = solve_exact(H, layout, lam="auto", normalization="isotropic")
    param, shift = unpack(jnp.asarray(W))

    # option 2: gradient pipeline 
    A = onp.array(param["weights"])
    Q = onp.diag(onp.array(param["noise_var"]))
    beta = onp.array(param["biases"])

    print("recovered precision (up to scale):")
    print(onp.round(-2.0 * onp.linalg.inv(Q) @ A, 2))
    print("true precision:")
    print(theta)

    for k in range(3):
        m_hat, S_hat = ou_stationary_moments(A, Q, beta, onp.array(shift)[k])
        mean_k = -onp.linalg.solve(A_true, beta_true + shifts_true[k])
        print(f"env {k}: predicted stationary mean {onp.round(m_hat, 2)}, "
              f"true {onp.round(mean_k, 2)}, "
              f"W2 = {gaussian_w2(m_hat, S_hat, mean_k, sigma):.3f}")
