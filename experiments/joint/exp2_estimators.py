import os
import sys
from functools import partial

import numpy as onp
import jax
from jax import vmap
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (make_reversible_ou, set_style, savefig, PALETTE, GRAY,
                    RESULTS_DIR)

from stadion.skds import skds_pair_a
from stadion.joint import JointLinearSDE, rbf_kernel

D = 3
NS = [50, 100, 200, 400, 800]
R = 300
BW = 2.0
SEED = 0


def main():
    import matplotlib.pyplot as plt

    gt = make_reversible_ou(D, [None], seed=SEED)
    env = gt["envs"][0]

    # evaluation parameter: normalized truth with perturbed bias -> SKDS > 0
    s = 2.0 / onp.trace(gt["Q"])
    param = dict(weights=jnp.asarray(s * gt["A"]),
                 biases=jnp.asarray(s * gt["beta"] + 0.3),
                 noise_var=jnp.asarray(onp.diag(s * gt["Q"])))

    model = JointLinearSDE()
    kernel = partial(rbf_kernel, bandwidth=BW)
    h_fn = skds_pair_a(lambda x, p: model.f(x, p, None),
                       lambda x, p: model.a(x, p, None),
                       kernel)

    @jax.jit
    def both_estimators(x):
        n = x.shape[0]
        hmat = vmap(vmap(h_fn, (None, 0, None)), (0, None, None))(x, x, param)
        u_stat = (hmat.sum() - jnp.trace(hmat)) / (n * (n - 1))
        pairs = x[: (n // 2) * 2].reshape(2, -1, D)
        lin = vmap(h_fn, (0, 0, None))(pairs[0], pairs[1], param).mean()
        return u_stat, lin

    rng = onp.random.default_rng(1)
    stats = {}
    for n in NS:
        us, ls = [], []
        for r in range(R):
            x = jnp.asarray(rng.multivariate_normal(env["mean"], env["cov"], size=n))
            u, l = both_estimators(x)
            us.append(float(u))
            ls.append(float(l))
        stats[n] = dict(u=onp.array(us), lin=onp.array(ls))
        print(f"N={n:4d}: mean_U={stats[n]['u'].mean():.4f} "
              f"mean_lin={stats[n]['lin'].mean():.4f} "
              f"var_U={stats[n]['u'].var():.6f} var_lin={stats[n]['lin'].var():.6f} "
              f"ratio={stats[n]['lin'].var() / stats[n]['u'].var():.2f}")

    ref = stats[NS[-1]]["u"].mean() 

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))

    ax = axes[0]
    for i, (key, label) in enumerate([("u", "U-statistic"), ("lin", "linear statistic")]):
        means = onp.array([stats[n][key].mean() for n in NS])
        sds = onp.array([stats[n][key].std() for n in NS])
        x = onp.array(NS) * (1.0 + 0.03 * i)
        ax.errorbar(x, means, yerr=sds, fmt="o-", ms=4, capsize=2.5,
                    color=PALETTE[i], label=label)
    ax.axhline(ref, color=GRAY, linestyle="--", linewidth=1.0,
               label="population value (ref.)")
    ax.set_xscale("log")
    ax.set_xlabel("sample size $N$")
    ax.set_ylabel("estimate ($\\pm$ 1 sd over draws)")
    ax.set_title("both estimators are unbiased")
    ax.legend(fontsize=7)

    ax = axes[1]
    for i, (key, label) in enumerate([("u", "U-statistic"), ("lin", "linear statistic")]):
        variances = onp.array([stats[n][key].var() for n in NS])
        ax.plot(NS, variances, "o-", ms=4, color=PALETTE[i], label=label)
    guide = stats[NS[0]]["u"].var() * (NS[0] / onp.array(NS, dtype=float))
    ax.plot(NS, guide, linestyle="--", color=GRAY, linewidth=1.0,
            label="$\\propto 1/N$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sample size $N$")
    ax.set_ylabel("variance over draws")
    ax.set_title("U-statistic dominates at rate $1/N$")
    ax.legend(fontsize=7)

    fig.suptitle(f"SKDS estimators at a fixed parameter (d={D}, {R} draws per N)",
                 y=1.04)
    fig.tight_layout()
    savefig(fig, "exp2_estimators")


if __name__ == "__main__":
    print("=== Experiment 2: U-statistic vs linear statistic ===")
    main()
