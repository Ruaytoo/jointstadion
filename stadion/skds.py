from functools import partial

import jax
from jax import vmap
from jax.tree_util import tree_map
import jax.numpy as jnp


def _wrapped_mean(func, axis=None):
    def wrapped(*args):
        return tree_map(partial(jnp.mean, axis=axis), func(*args))
    return wrapped


def skds_pair_a(f, a_fn, kernel):
    """
    Same as :func:`skds_pair`, but parametrized directly by the squared
    diffusion coefficient :math:`a(x) = \\sigma(x) \\sigma(x)^T` instead of
    :math:`\\sigma(x)`.
    """

    grad_x_kernel = jax.grad(kernel, argnums=0)
    grad_y_kernel = jax.grad(kernel, argnums=1)
    grad_xy_kernel = jax.jacfwd(grad_x_kernel, argnums=1)

    def h(x, y, *args):
        assert x.ndim == y.ndim == 1
        assert x.shape == y.shape

        b_x, b_y = f(x, *args), f(y, *args)
        a_x, a_y = a_fn(x, *args), a_fn(y, *args)

        k_xy = kernel(x, y)
        dk_x = grad_x_kernel(x, y)
        dk_y = grad_y_kernel(x, y)
        d2k_xy = grad_xy_kernel(x, y)

        return 4.0 * (b_x @ b_y) * k_xy \
               + 2.0 * b_x @ (a_y @ dk_y) \
               + 2.0 * b_y @ (a_x @ dk_x) \
               + jnp.sum((a_x @ a_y) * d2k_xy)

    return h


def skds_pair(f, sigma, kernel):
    """
    Stein kernel :math:`h(x, y) = (\\mathcal{S}_1 \\mathcal{S}_2 K)(x, y)` of the
    Stein-type KDS (SKDS) for the diffusion Stein operator applied to both 
    arguments of the matrix-valued kernel :math:`K = k I_d`.
    """

    def a_fn(x, *args):
        sig_x = sigma(x, *args)
        return sig_x @ sig_x.T

    return skds_pair_a(f, a_fn, kernel)


def skds_pair_mixed(f, sigma, kernel):
    """
    Mixed-parameter Stein kernel,
    where the operator acting on the first argument uses parameters ``args_x``
    and the operator acting on the second argument uses parameters ``args_y``.
    """

    grad_x_kernel = jax.grad(kernel, argnums=0)
    grad_xy_kernel = jax.jacfwd(grad_x_kernel, argnums=1)

    def h(x, y, args_x, args_y):
        assert x.ndim == y.ndim == 1
        assert x.shape == y.shape

        b_x, b_y = f(x, *args_x), f(y, *args_y)
        sig_x, sig_y = sigma(x, *args_x), sigma(y, *args_y)
        a_x = sig_x @ sig_x.T
        a_y = sig_y @ sig_y.T

        k_xy = kernel(x, y)
        dk_x = grad_x_kernel(x, y)
        dk_y = jax.grad(kernel, argnums=1)(x, y)
        d2k_xy = grad_xy_kernel(x, y)

        return 4.0 * (b_x @ b_y) * k_xy \
               + 2.0 * b_x @ (a_y @ dk_y) \
               + 2.0 * b_y @ (a_x @ dk_x) \
               + jnp.sum((a_x @ a_y) * d2k_xy)

    return h


def skds_loss(f, sigma, kernel, estimator="linear"):
    """
    SKDS loss function for arbitrary SDE functions :math:`f(x, \\dots)`
    and :math:`\\sigma(x, \\dots)`.
    """

    loss_term = skds_pair(f, sigma, kernel)

    if estimator == "v-statistic":
        # run check to make sure kernel is differentiable at x = x'
        x0, x1 = jnp.array([0.0]), jnp.array([1.0])
        x0_check = jnp.isnan(jax.grad(kernel)(x0, x0)).any()
        x1_check = jnp.isnan(jax.grad(kernel)(x1, x1)).any()
        assert not x0_check and not x1_check, \
            ("Kernel is not differentiable at x = x', "
             "which is required for the v-statistic. "
             "Try another estimator or re-writing the kernel. "
             "For example, for kernels involving L2 norms, "
             "`jnp.linalg.norm(x - y) ** 2` is not differentiable at x = y, "
             "but `jnp.square(x - y).sum(-1)` is.")

        @partial(_wrapped_mean, axis=(0, 1))
        @partial(vmap, in_axes=(0, None, None))
        @partial(vmap, in_axes=(None, 0, None))
        def _loss(x, y, args):
            return loss_term(x, y, *args)

        def loss(x, *args):
            assert x.ndim == 2
            return _loss(x, x, args)

    elif estimator == "u-statistic":
        @partial(_wrapped_mean, axis=(0, 1))
        @partial(vmap, in_axes=(0, None, None, 0), out_axes=0)
        @partial(vmap, in_axes=(None, 0, None, 0), out_axes=0)
        def _loss(x, y, args, mask):
            return jnp.where(mask, 0.0, loss_term(x, y, *args))

        def loss(x, *args):
            assert x.ndim == 2
            n, _ = x.shape
            return _loss(x, x, args, jnp.eye(n)) * n / (n - 1)

    elif estimator == "linear":
        @partial(_wrapped_mean, axis=(0,))
        @partial(vmap, in_axes=(0, 0, None))
        def _loss(x, y, args):
            return loss_term(x, y, *args)

        def loss(x, *args):
            assert x.ndim == 2
            n, d = x.shape
            x = (x[:-1] if n % 2 else x).reshape(2, -1, d)
            return _loss(x[0], x[1], args)

    else:
        raise ValueError(f"Unknown estimator `{estimator}`. "
                         f"Options: `linear`, `u-statistic`, `v-statistic`.")

    return loss
