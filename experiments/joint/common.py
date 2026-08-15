import os
import time

import numpy as onp

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
GRAY = "#52514e"


def set_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.6,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def savefig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(RESULTS_DIR, f"{name}.{ext}"), bbox_inches="tight")
    print(f"saved {os.path.join(RESULTS_DIR, name)}.png/.pdf")


# ground truth: reversible OU systems
def sparse_spd_precision(d, rng, edge_prob=None):
    edge_prob = edge_prob if edge_prob is not None else 2.0 / d
    B = onp.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < edge_prob:
                B[i, j] = B[j, i] = rng.uniform(0.2, 0.5) * rng.choice([-1, 1])
    theta = B + onp.eye(d) * (onp.abs(B).sum(1).max() + 1.0)
    Dinv = onp.diag(1.0 / onp.sqrt(onp.diag(theta)))
    return Dinv @ theta @ Dinv


def make_reversible_ou(d, targets_list, seed=0, shift_size=2.0, full_lambda=False):
    rng = onp.random.default_rng(seed)
    theta = sparse_spd_precision(d, rng)
    sigma = onp.linalg.inv(theta)

    if full_lambda:
        G = rng.normal(size=(d, d)) / onp.sqrt(d)
        lam = G @ G.T + onp.eye(d)
    else:
        lam = onp.diag(rng.uniform(0.5, 1.5, size=d))

    A = -lam @ theta
    Q = 2.0 * lam
    m0 = rng.normal(0.0, 1.0, size=d)
    beta = -A @ m0

    envs = []
    for j in targets_list:
        if j is None:
            envs.append(dict(shift=onp.zeros(d), target=onp.zeros(d),
                             mean=m0.copy(), cov=sigma.copy()))
        else:
            target = onp.eye(d)[j]
            delta = shift_size * rng.choice([-1, 1])
            shift = delta * target
            mean = -onp.linalg.solve(A, beta + shift)
            envs.append(dict(shift=shift, target=target, mean=mean, cov=sigma.copy()))

    return dict(A=A, Q=Q, beta=beta, theta=theta, sigma=sigma, m0=m0, lam=lam,
                envs=envs)


def sample_envs(gt, n_per_env, seed=0):
    rng = onp.random.default_rng(seed)
    xs, targets = [], []
    for env in gt["envs"]:
        xs.append(jnp.asarray(rng.multivariate_normal(env["mean"], env["cov"],
                                                      size=n_per_env)))
        targets.append(env["target"])
    return xs, onp.stack(targets)


# metrics
def theta_from_fit(A_hat, Q_hat):
    return -2.0 * onp.linalg.inv(Q_hat) @ A_hat


def rel_err(est, true):
    return float(onp.linalg.norm(est - true) / onp.linalg.norm(true))


def env_w2s(A, Q, beta, shifts, envs):
    from stadion.joint import ou_stationary_moments, gaussian_w2
    out = []
    for k, env in enumerate(envs):
        m_hat, S_hat = ou_stationary_moments(A, Q, beta, shifts[k])
        out.append(gaussian_w2(m_hat, S_hat, env["mean"], env["cov"]))
    return onp.array(out)


def predict_heldout_w2(A, Q, beta, test_env, test_x):
    from stadion.joint import ou_stationary_moments, gaussian_w2
    j = int(onp.argmax(test_env["target"]))
    x = onp.asarray(test_x)
    target_mean_obs = x[:, j].mean()

    A_inv = onp.linalg.inv(A)
    m_base = -A_inv @ beta
    # m(shift) = m_base - A^{-1} e_j * s  =>  match coordinate j
    denom = -A_inv[j, j]
    s = (target_mean_obs - m_base[j]) / denom
    shift = onp.eye(len(m_base))[j] * s

    m_hat, S_hat = ou_stationary_moments(A, Q, beta, shift)
    return gaussian_w2(m_hat, S_hat, x.mean(0), onp.cov(x.T))


class Timer:
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        self.dt = time.time() - self.t
