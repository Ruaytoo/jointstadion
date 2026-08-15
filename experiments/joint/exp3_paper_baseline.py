import os
import sys
from functools import partial

import numpy as onp
import jax
from jax import random
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (make_reversible_ou, sample_envs, theta_from_fit, rel_err,
                    predict_heldout_w2, set_style, savefig, Timer,
                    PALETTE, GRAY, RESULTS_DIR)

from stadion.models import LinearSDE
from stadion.joint import (JointLinearSDE, assemble_quadratic, solve_exact,
                           rbf_kernel, median_heuristic_bandwidth,
                           ou_stationary_moments, gaussian_w2)

D = 5
TRAIN_TARGETS = [None, 0, 1, 2, 3]
TEST_TARGET = 4
N = 1000
SEEDS = list(range(10))
SHIFT = 2.0

# paper baseline budget: the codebase defaults of Lorch et al. / Bleile et al.
BASELINE_STEPS = 10000
BASELINE_BATCH = 128
BASELINE_LR = 0.003
BASELINE_REG = 0.001


def evaluate(A, Q_envs, beta, shifts, gt, test_x):
    train_envs = gt["envs"][:len(TRAIN_TARGETS)]

    w2s = []
    for k, env in enumerate(train_envs):
        m_hat, S_hat = ou_stationary_moments(A, Q_envs[k], beta, shifts[k])
        w2s.append(gaussian_w2(m_hat, S_hat, env["mean"], env["cov"]))
    w2s = onp.array(w2s)

    theta_hat = theta_from_fit(A, Q_envs[0])
    e_theta = rel_err(theta_hat, gt["theta"])

    errs = []
    A_inv = onp.linalg.inv(A)
    for k in range(1, len(TRAIN_TARGETS)):
        delta_hat = -A_inv @ shifts[k]
        delta_true = gt["envs"][k]["mean"] - gt["m0"]
        errs.append(onp.linalg.norm(delta_hat - delta_true) / onp.linalg.norm(delta_true))

    w2_test = predict_heldout_w2(A, Q_envs[0], beta, gt["envs"][-1], test_x)

    return dict(theta_err=e_theta, shift_err=float(onp.mean(errs)),
                w2_train=float(w2s.mean()), w2_test=float(w2_test))


def fit_algorithm1(seed, xs, targets, bw, objective):
    model = LinearSDE()
    model.fit(
        random.PRNGKey(seed),
        [jnp.asarray(x) for x in xs],
        targets=[jnp.asarray(t) for t in targets],
        objective=objective,
        estimator="linear",
        bandwidth=float(bw),
        learning_rate=BASELINE_LR,
        steps=BASELINE_STEPS,
        batch_size=BASELINE_BATCH,
        reg=BASELINE_REG,
        verbose=0,
    )
    A = onp.array(model.param["weights"])
    beta = onp.array(model.param["biases"])
    c = onp.array(model.param["log_noise_scale"])
    shifts = onp.array(model.intv_param["shift"])
    log_scale = onp.array(model.intv_param["log_scale"])
    # per-environment squared diffusion of the shift-scale intervention model
    Q_envs = [onp.diag(onp.exp(2.0 * (c + ls))) for ls in log_scale]
    return A, Q_envs, beta, shifts


def summarize(rows, keys=("theta_err", "shift_err", "w2_train", "w2_test", "time")):
    import collections
    by_method = collections.defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    lines = []
    header = f"{'method':<22}" + "".join(f"{k:>22}" for k in keys)
    lines.append(header)
    for m, rs in by_method.items():
        vals = []
        for k in keys:
            arr = onp.array([r[k] for r in rs])
            if k == "time":
                vals.append(f"{onp.median(arr):>15.2f} s    ")
            else:
                vals.append(f"{arr.mean():>12.4f} ±{arr.std():>7.4f}")
        lines.append(f"{m:<22}" + "".join(f"{v:>22}" for v in vals))
    return "\n".join(lines)


def write_csv(rows, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


def main():
    rows = []
    for seed in SEEDS:
        gt = make_reversible_ou(D, TRAIN_TARGETS + [TEST_TARGET], seed=seed,
                                shift_size=SHIFT)
        xs_all, targets_all = sample_envs(gt, N, seed=1000 + seed)
        xs, targets = xs_all[:-1], targets_all[:-1]
        test_x = xs_all[-1]

        bw = median_heuristic_bandwidth(xs)
        kernel = partial(rbf_kernel, bandwidth=bw)

        # joint stratified U-statistic
        model = JointLinearSDE()
        with Timer() as t0:
            H, pack, unpack, layout = assemble_quadratic(model, kernel, xs, targets)
            W, info = solve_exact(H, layout, lam="auto", normalization="isotropic")
        param, shift = unpack(jnp.asarray(W))
        A = onp.array(param["weights"])
        Q = onp.diag(onp.array(param["noise_var"]))
        beta = onp.array(param["biases"])
        Q_envs = [Q] * len(xs)
        rows.append(dict(seed=seed, method="JSKDS joint (thesis)", time=t0.dt,
                         **evaluate(A, Q_envs, beta, onp.array(shift), gt, test_x)))

        # SKDS + Algorithm 1 
        with Timer() as t1:
            A, Q_envs, beta, shifts = fit_algorithm1(seed, xs, targets, bw, "skds")
        rows.append(dict(seed=seed, method="SKDS Alg. 1 (paper)", time=t1.dt,
                         **evaluate(A, Q_envs, beta, shifts, gt, test_x)))

        # Lorch et al.: KDS + Algorithm 1 
        with Timer() as t2:
            A, Q_envs, beta, shifts = fit_algorithm1(seed, xs, targets, bw, "kds")
        rows.append(dict(seed=seed, method="KDS Alg. 1 (Lorch)", time=t2.dt,
                         **evaluate(A, Q_envs, beta, shifts, gt, test_x)))

        print(f"seed {seed} done", flush=True)

    write_csv(rows, os.path.join(RESULTS_DIR, "exp3_paper_baseline.csv"))

    print(f"\nJoint JSKDS vs. Algorithm 1 baselines "
          f"(d={D}, {len(TRAIN_TARGETS) - 1} train + 1 held-out intervention, "
          f"N={N}/env, {len(SEEDS)} seeds):")
    print(summarize(rows))


    # figure: strip plots per metric
    set_style()
    import matplotlib.pyplot as plt

    metrics = [("theta_err", "relative error of $\\Theta$"),
               ("shift_err", "rel. error of predicted mean shifts"),
               ("w2_test", "$W_2$ to true law, held-out intervention")]
    methods = []
    for r in rows:
        if r["method"] not in methods:
            methods.append(r["method"])

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for ax, (metric, label) in zip(axes, metrics):
        for i, method in enumerate(methods):
            vals = onp.array([r[metric] for r in rows if r["method"] == method])
            jitter = onp.random.default_rng(3).uniform(-0.10, 0.10, len(vals))
            ax.scatter(onp.full(len(vals), i) + jitter, vals, s=14,
                       color=PALETTE[i], alpha=0.85, zorder=3)
            ax.hlines(onp.median(vals), i - 0.22, i + 0.22, color=PALETTE[i],
                      linewidth=2.2, zorder=4)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(["JSKDS joint\n(thesis)", "SKDS Alg. 1\n(paper)",
                            "KDS Alg. 1\n(Lorch)"])
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
    fig.suptitle(f"Joint objective vs. Algorithm 1 "
                 f"(d={D}, N={N}/env, {len(SEEDS)} seeds)", y=1.04)
    fig.tight_layout()
    savefig(fig, "exp3_paper_baseline")


if __name__ == "__main__":
    main()
