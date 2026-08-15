import numpy as onp
import jax
import jax.numpy as jnp
from jax import random

from stadion.joint import (
    JointLinearSDE,
    joint_skds_ustat,
    pooled_skds_ustat,
    assemble_quadratic,
    solve_exact,
    project_simplex,
    rbf_kernel,
    ou_stationary_moments,
    gaussian_w2,
    theorem45_params,
)

jax.config.update("jax_enable_x64", True)


# ground-truth generator: reversible OU with diagonal Q and shift interventions
def make_reversible_ou(d, n_intv, seed=0, shift_size=2.0):
    rng = onp.random.default_rng(seed)

    # sparse SPD precision via a random partial correlation structure
    B = onp.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < 2.0 / d:
                B[i, j] = B[j, i] = rng.uniform(0.2, 0.5) * rng.choice([-1, 1])
    theta = B + onp.eye(d) * (onp.abs(B).sum(1).max() + 1.0)
    # scale to unit diagonal 
    Dinv = onp.diag(1.0 / onp.sqrt(onp.diag(theta)))
    theta = Dinv @ theta @ Dinv
    sigma = onp.linalg.inv(theta)

    lam = onp.diag(rng.uniform(0.5, 1.5, size=d))
    A = -lam @ theta
    Q = 2.0 * lam
    m0 = rng.normal(0.0, 1.0, size=d)
    beta = -A @ m0

    envs = [dict(shift=onp.zeros(d), target=onp.zeros(d), mean=m0, cov=sigma)]
    for k in range(n_intv):
        j = k % d
        target = onp.eye(d)[j]
        delta = shift_size * rng.choice([-1, 1])
        shift = delta * target
        mean = -onp.linalg.solve(A, beta + shift)
        envs.append(dict(shift=shift, target=target, mean=mean, cov=sigma))

    return dict(A=A, Q=Q, beta=beta, theta=theta, sigma=sigma, m0=m0, envs=envs)


def sample_envs(gt, n_per_env, seed=0):
    rng = onp.random.default_rng(seed)
    xs, targets = [], []
    for env in gt["envs"]:
        xs.append(rng.multivariate_normal(env["mean"], env["cov"], size=n_per_env))
        targets.append(env["target"])
    return [jnp.asarray(x) for x in xs], onp.stack(targets)

# tests
def test_project_simplex():
    v = jnp.array([0.5, -0.2, 1.9])
    p = project_simplex(v, 2.0)
    assert jnp.all(p >= 0)
    assert jnp.isclose(p.sum(), 2.0)
    # already feasible point is a fixed point
    v2 = jnp.array([0.5, 0.5, 1.0])
    assert jnp.allclose(project_simplex(v2, 2.0), v2, atol=1e-9)


def test_quadratic_matches_loss():
    d, M, n = 3, 2, 60
    gt = make_reversible_ou(d, M, seed=1)
    xs, targets = sample_envs(gt, n, seed=2)

    model = JointLinearSDE()
    kernel = lambda x, y: rbf_kernel(x, y, bandwidth=2.0)
    H, pack, unpack, layout = assemble_quadratic(model, kernel, xs, targets)

    loss = joint_skds_ustat(model, kernel, xs, targets)
    key = random.PRNGKey(0)
    for seed in range(5):
        W = random.normal(random.PRNGKey(seed), (layout["d_W"],))
        param, shift = unpack(W)
        lval = loss(param, {"shift": shift})
        qval = float(W @ H @ W)
        assert onp.isclose(float(lval), qval, rtol=1e-6, atol=1e-9), (lval, qval)

    # H is symmetric; its population counterpart is PSD (Corollary 3.4), and
    # the U-statistic version concentrates around it
    assert onp.allclose(H, H.T, atol=1e-10)
    evals = onp.linalg.eigvalsh(H)
    assert evals.min() > -0.05, evals.min()

    # with more samples, concentration improves
    xs_big, targets_big = sample_envs(gt, 500, seed=20)
    H_big, *_ = assemble_quadratic(model, kernel, xs_big, targets_big)
    evals_big = onp.linalg.eigvalsh(H_big)
    assert evals_big.min() > 0.5 * evals.min(), (evals.min(), evals_big.min())


def test_truth_in_zero_set_and_solver_recovers_laws():
    d, M, n = 3, 3, 400
    gt = make_reversible_ou(d, M, seed=3)
    xs, targets = sample_envs(gt, n, seed=4)

    model = JointLinearSDE()
    kernel = lambda x, y: rbf_kernel(x, y, bandwidth=2.0)
    H, pack, unpack, layout = assemble_quadratic(model, kernel, xs, targets)

    # normalized truth: scale so that tr(Q) = 2
    s = 2.0 / onp.trace(gt["Q"])
    W_true = onp.concatenate([
        (s * gt["A"]).reshape(-1),
        s * gt["beta"],
        onp.diag(s * gt["Q"]),
        [s * gt["envs"][k]["shift"][onp.argmax(targets[k])] for k in range(1, M + 1)],
    ])
    val_true = float(W_true @ H @ W_true)

    # a generic parameter has much larger objective
    rngW = onp.random.default_rng(0)
    W_rand = W_true + 0.5 * rngW.normal(size=layout["d_W"])
    val_rand = float(W_rand @ H @ W_rand)
    assert val_true < 0.05 * val_rand, (val_true, val_rand)

    # exact solve 
    W_star, info = solve_exact(H, layout, lam="auto")
    assert info["feasible"]

    param, shift = unpack(jnp.asarray(W_star))
    A_hat = onp.array(param["weights"])
    Q_hat = onp.diag(onp.array(param["noise_var"]))
    beta_hat = onp.array(param["biases"])

    # A_hat must be Hurwitz
    assert onp.max(onp.linalg.eigvals(A_hat).real) < 0

    for k, env in enumerate(gt["envs"]):
        m_hat, S_hat = ou_stationary_moments(A_hat, Q_hat, beta_hat,
                                             onp.array(shift)[k])
        w2 = gaussian_w2(m_hat, S_hat, env["mean"], env["cov"])
        assert w2 < 0.35, (k, w2, m_hat, env["mean"])


def test_joint_beats_pooled_on_cancellation_example():
    rng = onp.random.default_rng(7)
    n = 3000
    xs = [jnp.asarray(rng.normal(size=(n, 1))) for _ in range(3)]
    targets = onp.array([[0.0], [1.0], [1.0]])

    model = JointLinearSDE()
    kernel = lambda x, y: rbf_kernel(x, y, bandwidth=1.0)

    joint_loss = joint_skds_ustat(model, kernel, xs, targets)
    pooled_loss = pooled_skds_ustat(model, kernel, xs, targets)

    param = dict(weights=-jnp.eye(1), biases=jnp.zeros(1), noise_var=2.0 * jnp.ones(1))

    delta = 1.0
    spurious = {"shift": jnp.array([[0.0], [delta], [-delta]])}
    truthful = {"shift": jnp.array([[0.0], [0.0], [0.0]])}

    j_spur = float(joint_loss(param, spurious))
    j_true = float(joint_loss(param, truthful))
    p_spur = float(pooled_loss(param, spurious))
    p_true = float(pooled_loss(param, truthful))

    assert abs(p_spur - p_true) < 0.1 * abs(j_spur - j_true), (p_spur, p_true, j_spur, j_true)
    assert j_spur > j_true + 0.5 * abs(j_spur), (j_spur, j_true)


def test_moment_pipeline_theorem45():
    d, M = 4, 4
    gt = make_reversible_ou(d, M, seed=8)

    means = [env["mean"] for env in gt["envs"]]
    covs = [env["cov"] for env in gt["envs"]]
    targets = onp.stack([env["target"] for env in gt["envs"]])
    Sigma = onp.mean(onp.stack(covs), axis=0)

    Lam_iso = onp.eye(d) / d
    rec = theorem45_params(Lam_iso, Sigma, means[0], means, targets)
    for k, env in enumerate(gt["envs"]):
        m_hat, S_hat = ou_stationary_moments(rec["A"], rec["Q"], rec["beta"],
                                             rec["shifts"][k])
        assert gaussian_w2(m_hat, S_hat, env["mean"], env["cov"]) < 1e-6


if __name__ == "__main__":
    test_project_simplex()
    print("project_simplex OK")
    test_quadratic_matches_loss()
    print("quadratic OK")
    test_truth_in_zero_set_and_solver_recovers_laws()
    print("zero set + exact solve OK")
    test_joint_beats_pooled_on_cancellation_example()
    print("pooled vs joint OK")
    test_moment_pipeline_theorem45()
    print("theorem 4.5 moments OK")
    print("ALL JOINT TESTS PASSED")
