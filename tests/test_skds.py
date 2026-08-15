import jax
import jax.numpy as jnp
from jax import random
import numpy as onp

from stadion.skds import skds_pair, skds_pair_mixed, skds_loss

jax.config.update("jax_enable_x64", True)


def rbf_kernel(x, y, bandwidth=2.0):
    return jnp.exp(-jnp.square(x - y).sum(-1) / (2.0 * bandwidth ** 2))


def _random_linear_sde(key, d):
    k0, k1, k2 = random.split(key, 3)
    A = random.normal(k0, (d, d)) / jnp.sqrt(d)
    beta = random.normal(k1, (d,))
    G = random.normal(k2, (d, d)) / jnp.sqrt(d)

    def f(x, param):
        return param["A"] @ x + param["beta"]

    def sigma(x, param):
        return param["G"]

    return f, sigma, dict(A=A, beta=beta, G=G)


def _nested_reference_pair(f, sigma, kernel):
    def s1k(x, y, *args):
        b_x = f(x, *args)
        sig_x = sigma(x, *args)
        a_x = sig_x @ sig_x.T
        return 2.0 * b_x * kernel(x, y) + a_x @ jax.grad(kernel, 0)(x, y)

    def h(x, y, *args):
        b_y = f(y, *args)
        sig_y = sigma(y, *args)
        a_y = sig_y @ sig_y.T
        g_val = s1k(x, y, *args)
        # jac[i, l] = d/dy_l (S1 K)_i
        jac = jax.jacfwd(s1k, argnums=1)(x, y, *args)
        return 2.0 * (b_y @ g_val) + jnp.sum(a_y * jac)

    return h


def test_closed_form_matches_nested_operator():
    key = random.PRNGKey(0)
    d = 4
    f, sigma, param = _random_linear_sde(key, d)

    h_closed = skds_pair(f, sigma, rbf_kernel)
    h_ref = _nested_reference_pair(f, sigma, rbf_kernel)

    for seed in range(10):
        kx, ky = random.split(random.PRNGKey(seed))
        x = random.normal(kx, (d,))
        y = random.normal(ky, (d,))
        v1 = h_closed(x, y, param)
        v2 = h_ref(x, y, param)
        assert jnp.allclose(v1, v2, rtol=1e-9, atol=1e-9), (v1, v2)


def test_mixed_pair_consistent_with_plain_pair():
    key = random.PRNGKey(1)
    d = 3
    f, sigma, param = _random_linear_sde(key, d)

    h_plain = skds_pair(f, sigma, rbf_kernel)
    h_mixed = skds_pair_mixed(f, sigma, rbf_kernel)

    kx, ky = random.split(random.PRNGKey(7))
    x = random.normal(kx, (d,))
    y = random.normal(ky, (d,))
    assert jnp.allclose(h_plain(x, y, param), h_mixed(x, y, (param,), (param,)))


def test_estimators_agree_in_expectation():
    key = random.PRNGKey(2)
    d = 2
    f, sigma, param = _random_linear_sde(key, d)

    x = random.normal(random.PRNGKey(3), (2000, d))

    loss_u = skds_loss(f, sigma, rbf_kernel, estimator="u-statistic")
    loss_lin = skds_loss(f, sigma, rbf_kernel, estimator="linear")
    loss_v = skds_loss(f, sigma, rbf_kernel, estimator="v-statistic")

    ju = loss_u(x, param)
    jl = loss_lin(x, param)
    jv = loss_v(x, param)

    # unbiased estimators of the same population quantity
    assert jnp.abs(ju - jl) < 0.15 * (jnp.abs(ju) + 1.0)
    # v-statistic has a diagonal O(1/n) bias but should be close as well
    assert jnp.abs(ju - jv) < 0.15 * (jnp.abs(ju) + 1.0)


def test_skds_zero_at_reversible_truth_1d():
    def f(x, param):
        return param["b"] * (x - param["alpha"])

    def sigma(x, param):
        return param["sig"] * jnp.eye(1)

    truth = dict(b=jnp.array(-4.0), alpha=jnp.array(1.0), sig=jnp.array(2.0))

    key = random.PRNGKey(4)
    x = 1.0 + jnp.sqrt(0.5) * random.normal(key, (4000, 1))

    kernel = lambda x_, y_: rbf_kernel(x_, y_, bandwidth=0.5)
    loss = skds_loss(f, sigma, kernel, estimator="u-statistic")

    j_true = loss(x, truth)
    j_wrong_mean = loss(x, dict(truth, alpha=jnp.array(0.0)))
    j_wrong_sig = loss(x, dict(truth, sig=jnp.array(1.0)))

    assert jnp.abs(j_true) < 0.05 * jnp.abs(j_wrong_mean)
    assert jnp.abs(j_true) < 0.05 * jnp.abs(j_wrong_sig)

    # gradient w.r.t. parameters approximately vanishes at the truth
    g_true = jax.grad(lambda p: loss(x, p))(truth)
    g_wrong = jax.grad(lambda p: loss(x, p))(dict(truth, alpha=jnp.array(0.0)))
    g_true_norm = jnp.sqrt(sum(jnp.sum(v ** 2) for v in g_true.values()))
    g_wrong_norm = jnp.sqrt(sum(jnp.sum(v ** 2) for v in g_wrong.values()))
    assert g_true_norm < 0.1 * g_wrong_norm


def test_skds_zero_at_reversible_truth_multivariate():
    d = 3
    rng = onp.random.default_rng(0)

    # sparse SPD precision
    theta = onp.eye(d)
    theta[0, 1] = theta[1, 0] = 0.4
    theta[1, 2] = theta[2, 1] = -0.3
    sig = onp.linalg.inv(theta)
    m = onp.array([0.5, -1.0, 0.2])

    lam = onp.diag(onp.array([0.6, 1.0, 0.8]))
    A = -lam @ theta
    Q = 2.0 * lam

    x = rng.multivariate_normal(m, sig, size=4000)
    x = jnp.array(x)

    def f(z, param):
        return param["A"] @ z + param["beta"]

    def sigma_fn(z, param):
        return param["sqrtQ"]

    truth = dict(A=jnp.array(A), beta=jnp.array(-A @ m), sqrtQ=jnp.array(onp.sqrt(Q) * onp.eye(d) ** 0 * (onp.diag(onp.sqrt(onp.diag(Q))) != 0)))
    truth["sqrtQ"] = jnp.array(onp.diag(onp.sqrt(onp.diag(Q))))

    loss = skds_loss(f, sigma_fn, lambda a, b: rbf_kernel(a, b, bandwidth=2.0),
                     estimator="u-statistic")

    j_true = loss(x, truth)
    j_wrong = loss(x, dict(truth, beta=truth["beta"] + 1.0))
    assert jnp.abs(j_true) < 0.05 * jnp.abs(j_wrong), (j_true, j_wrong)


if __name__ == "__main__":
    test_closed_form_matches_nested_operator()
    test_mixed_pair_consistent_with_plain_pair()
    test_estimators_agree_in_expectation()
    test_skds_zero_at_reversible_truth_1d()
    test_skds_zero_at_reversible_truth_multivariate()
    print("ALL SKDS TESTS PASSED")
