import os
import sys
from functools import partial

import numpy as onp
import jax
from jax import random
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (make_reversible_ou, sample_envs, theta_from_fit, rel_err,
                    env_w2s, predict_heldout_w2, set_style, savefig, Timer,
                    PALETTE, GRAY, RESULTS_DIR)

from stadion.joint import (JointLinearSDE, assemble_quadratic, solve_exact,
                           fit_joint, rbf_kernel, median_heuristic_bandwidth,
                           theorem45_params, ou_stationary_moments, gaussian_w2)

D = 5
TRAIN_TARGETS = [None, 0, 1, 2, 3]
TEST_TARGET = 4
N_MAIN = 1000
N_SECONDARY = 300
SEEDS = list(range(10))
SHIFT = 2.0


def evaluate(A, Q, beta, shifts, gt, test_x):
    train_envs = gt["envs"][:len(TRAIN_TARGETS)]
    w2s = env_w2s(A, Q, beta, shifts, train_envs)

    # degeneracy-invariant precision error
    theta_hat = theta_from_fit(A, Q)
    e_theta = rel_err(theta_hat, gt["theta"])

    # mean-shift recovery on train interventions
    errs = []
    A_inv = onp.linalg.inv(A)
    m_base, _ = ou_stationary_moments(A, Q, beta, None)
    for k in range(1, len(TRAIN_TARGETS)):
        delta_hat = (-A_inv @ shifts[k])
        delta_true = gt["envs"][k]["mean"] - gt["m0"]
        errs.append(onp.linalg.norm(delta_hat - delta_true) / onp.linalg.norm(delta_true))

    # held-out intervention
    w2_test = predict_heldout_w2(A, Q, beta, gt["envs"][-1], test_x)

    return dict(theta_err=e_theta, w2_train=float(w2s.mean()),
                w2_train_max=float(w2s.max()), shift_err=float(onp.mean(errs)),
                w2_test=float(w2_test))


def unpack_np(unpack, W):
    param, shift = unpack(jnp.asarray(W))
    A = onp.array(param["weights"])
    Q = onp.diag(onp.array(param["noise_var"]))
    beta = onp.array(param["biases"])
    return A, Q, beta, onp.array(shift)


def run_main():
    rows = []
    for seed in SEEDS:
        gt = make_reversible_ou(D, TRAIN_TARGETS + [TEST_TARGET], seed=seed,
                                shift_size=SHIFT)
        xs_all, targets_all = sample_envs(gt, N_MAIN, seed=1000 + seed)
        xs, targets = xs_all[:-1], targets_all[:-1]
        test_x = xs_all[-1]

        bw = median_heuristic_bandwidth(xs)
        kernel = partial(rbf_kernel, bandwidth=bw)
        model = JointLinearSDE()

        # JSKDS exact 
        with Timer() as t_exact:
            H, pack, unpack, layout = assemble_quadratic(model, kernel, xs, targets)
            W, info = solve_exact(H, layout, lam="auto", normalization="isotropic")
        A, Q, beta, shifts = unpack_np(unpack, W)
        rows.append(dict(seed=seed, method="JSKDS (exact)", time=t_exact.dt,
                         **evaluate(A, Q, beta, shifts, gt, test_x)))

        # JSKDS adam 
        import optax
        model_a = JointLinearSDE()
        with Timer() as t_adam:
            schedule = optax.cosine_decay_schedule(0.02, 4000, alpha=0.02)
            model_a = fit_joint(model_a, random.PRNGKey(seed), xs, list(targets),
                                kernel=kernel, steps=4000, batch_size=256,
                                learning_rate=schedule, reg=1e-3, verbose=0)
        A, Q, beta = (onp.array(model_a.param["weights"]),
                      onp.diag(onp.array(model_a.param["noise_var"])),
                      onp.array(model_a.param["biases"]))
        shifts = onp.array(model_a.intv_param["shift"])
        rows.append(dict(seed=seed, method="JSKDS (adam)", time=t_adam.dt,
                         **evaluate(A, Q, beta, shifts, gt, test_x)))

        #  moment-based estimator (Theorem 4.5
        with Timer() as t_mom:
            means = [onp.asarray(x).mean(0) for x in xs]
            covs = [onp.cov(onp.asarray(x).T) for x in xs]
            Sigma_hat = onp.mean(onp.stack(covs), axis=0)
            Lam_iso = (2.0 / (2.0 * D)) * onp.eye(D)
            rec = theorem45_params(Lam_iso, Sigma_hat, means[0], means, targets)
        A, Q, beta = rec["A"], rec["Q"], rec["beta"]
        shifts = onp.stack([s * t for s, t in zip(rec["shifts"], targets)])
        theta_mom = onp.linalg.inv(Sigma_hat)
        res = evaluate(A, Q, beta, shifts, gt, test_x)
        res["theta_err"] = rel_err(theta_mom, gt["theta"])  # invariant version
        rows.append(dict(seed=seed, method="moments (Thm 4.5)", time=t_mom.dt, **res))

        print(f"seed {seed} done", flush=True)
    return rows


def run_secondary():
    rows = []
    for seed in SEEDS:
        gt = make_reversible_ou(D, TRAIN_TARGETS + [TEST_TARGET], seed=seed,
                                shift_size=SHIFT)
        xs_all, targets_all = sample_envs(gt, N_SECONDARY, seed=2000 + seed)
        xs, targets = xs_all[:-1], targets_all[:-1]
        test_x = xs_all[-1]

        bw = median_heuristic_bandwidth(xs)
        kernel = partial(rbf_kernel, bandwidth=bw)
        model = JointLinearSDE()

        for objective, label in [("skds", "JSKDS"), ("kds", "joint KDS"),
                                 ("pooled", "pooled SKDS")]:
            with Timer() as t:
                H, pack, unpack, layout = assemble_quadratic(
                    model, kernel, xs, targets, objective=objective)
                W, info = solve_exact(H, layout, lam="auto", normalization="isotropic")
            A, Q, beta, shifts = unpack_np(unpack, W)
            rows.append(dict(seed=seed, method=label, time=t.dt,
                             **evaluate(A, Q, beta, shifts, gt, test_x)))
        print(f"secondary seed {seed} done", flush=True)
    return rows


def summarize(rows, keys=("theta_err", "shift_err", "w2_train", "w2_test", "time")):
    import collections
    by_method = collections.defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    lines = []
    header = f"{'method':<18}" + "".join(f"{k:>22}" for k in keys)
    lines.append(header)
    for m, rs in by_method.items():
        vals = []
        for k in keys:
            arr = onp.array([r[k] for r in rs])
            vals.append(f"{arr.mean():>12.4f} ±{arr.std():>7.4f}")
        lines.append(f"{m:<18}" + "".join(f"{v:>22}" for v in vals))
    return "\n".join(lines)


def write_csv(rows, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


def plot(rows_main, rows_secondary):
    import matplotlib.pyplot as plt
    set_style()

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.9))

    def strip(ax, rows, metric, methods, title, ylabel):
        for i, m in enumerate(methods):
            vals = onp.array([r[metric] for r in rows if r["method"] == m])
            x = onp.full(len(vals), i) + onp.linspace(-0.12, 0.12, len(vals))
            ax.scatter(x, vals, s=14, color=PALETTE[i], alpha=0.85, zorder=3,
                       edgecolors="white", linewidths=0.4)
            ax.hlines(onp.median(vals), i - 0.22, i + 0.22, color=PALETTE[i],
                      linewidth=2.0, zorder=4)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace(" (", "\n(") for m in methods])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)

    methods_main = ["JSKDS (exact)", "JSKDS (adam)", "moments (Thm 4.5)"]
    strip(axes[0], rows_main, "theta_err", methods_main,
          "shared precision $\\Theta$", "relative error")
    strip(axes[1], rows_main, "w2_test", methods_main,
          "held-out intervention", "$W_2$ to true law")
    methods_sec = ["JSKDS", "joint KDS", "pooled SKDS"]
    strip(axes[2], rows_secondary, "theta_err", methods_sec,
          f"objectives (N={N_SECONDARY})", "relative error of $\\Theta$")

    fig.suptitle(f"Joint learning from interventional data (d={D}, "
                 f"{len(TRAIN_TARGETS) - 1} train + 1 held-out intervention, "
                 f"{len(SEEDS)} seeds)", y=1.04)
    fig.tight_layout()
    savefig(fig, "exp1_recovery")


if __name__ == "__main__":
    print("=== Experiment 1: recovery and held-out interventions ===")
    rows_main = run_main()
    write_csv(rows_main, os.path.join(RESULTS_DIR, "exp1_main.csv"))
    print("\nMain study (N=1000/env):")
    print(summarize(rows_main))

    rows_sec = run_secondary()
    write_csv(rows_sec, os.path.join(RESULTS_DIR, "exp1_secondary.csv"))
    print(f"\nSecondary study (N={N_SECONDARY}/env):")
    print(summarize(rows_sec))

    plot(rows_main, rows_sec)
