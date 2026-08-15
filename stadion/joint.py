from functools import partial

import numpy as onp
import scipy.linalg as sla

import jax
from jax import vmap, random
import jax.numpy as jnp
import optax

from stadion.sde import SDE
from stadion.parameters import ModelParameters, InterventionParameters
from stadion.skds import skds_pair_a
from stadion.utils import to_diag, tree_init_normal


__all__ = [
    "rbf_kernel",
    "median_heuristic_bandwidth",
    "JointLinearSDE",
    "joint_ustat",
    "joint_skds_ustat",
    "pooled_skds_ustat",
    "assemble_quadratic",
    "resolve_ridge",
    "solve_exact",
    "project_simplex",
    "project_simplex_lb",
    "fit_joint",
    "ou_stationary_moments",
    "gaussian_w2",
    "theorem45_params",
]


# kernels
def rbf_kernel(x, y, bandwidth=2.0):
    return jnp.exp(-jnp.square(x - y).sum(-1) / (2.0 * bandwidth ** 2))


def median_heuristic_bandwidth(xs, max_points=1000, seed=0):

    pooled = onp.concatenate([onp.asarray(x) for x in xs], axis=0)
    rng = onp.random.default_rng(seed)
    if pooled.shape[0] > max_points:
        pooled = pooled[rng.choice(pooled.shape[0], size=max_points, replace=False)]
    dists = onp.sqrt(onp.square(pooled[:, None, :] - pooled[None, :, :]).sum(-1))
    med = onp.median(dists[onp.triu_indices_from(dists, k=1)])
    return float(med)


# shared-mechanism linear (OU) model with shift interventions
class JointLinearSDE(SDE):
    def __init__(self, trQ=2.0, normalization="isotropic", gamma_min_frac=0.05,
                 sde_kwargs=None):
        assert normalization in ("isotropic", "trQ")
        sde_kwargs = sde_kwargs or {}
        SDE.__init__(self, **sde_kwargs)
        self.trQ = trQ
        self.normalization = normalization
        self.gamma_min_frac = gamma_min_frac

    # parameter initialization
    def init_param(self, key, d, scale=1e-6):

        gamma0 = self.trQ / d
        shape = {
            "weights": -0.5 * gamma0 * jnp.eye(d),
            "biases": jnp.zeros((d,)),
            "noise_var": gamma0 * jnp.ones((d,)),
        }
        param = tree_init_normal(key, shape, scale=scale)
        param["noise_var"] = jnp.abs(param["noise_var"])
        return ModelParameters(parameters=param)

    def init_intv_param(self, key, d, n_envs=None, scale=1e-6, targets=None, x=None):
        """
        Initialize intervention shift parameters, optionally warm-started with
        the mean shift of each environment w.r.t. the first (observational)
        dataset, and masked by the known targets.
        """
        vec_shape = (n_envs, d) if n_envs is not None else (d,)
        shape = {"shift": jnp.zeros(vec_shape)}
        intv_param = tree_init_normal(key, shape, scale=scale)

        if targets is not None:
            targets = jnp.array(targets, dtype=jnp.float32)
            assert targets.shape == vec_shape

        if x is not None and targets is not None and n_envs is not None:
            assert len(x) == n_envs
            ref = x[0].mean(-2)
            mean_shift = jnp.array([
                jnp.where(t_, x_.mean(-2) - ref, 0.0) for x_, t_ in zip(x, targets)
            ])
            intv_param["shift"] += mean_shift

        return InterventionParameters(parameters=intv_param, targets=targets)

    # mechanism
    def f(self, x, param, intv_param):
        f_vec = x @ param["weights"].T + param["biases"]
        if intv_param is not None:
            f_vec = f_vec + intv_param["shift"]
        assert f_vec.shape == x.shape
        return f_vec

    def a(self, x, param, intv_param):
        return to_diag(jnp.ones_like(x) * param["noise_var"])

    def sigma(self, x, param, intv_param):
        c = jnp.sqrt(jnp.clip(param["noise_var"], 0.0, None))
        return to_diag(jnp.ones_like(x) * c)

    # constraint projection (normalization + nonnegativity)
    def project_params(self, param, intv_param):
        d = param["noise_var"].shape[0]
        if self.normalization == "isotropic":
            param["noise_var"] = (self.trQ / d) * jnp.ones_like(param["noise_var"])
        else:
            lb = self.gamma_min_frac * self.trQ / d
            param["noise_var"] = project_simplex_lb(param["noise_var"], self.trQ, lb)
        return param, intv_param

    # parameter vector packing (for the quadratic form W^T H W)
    def packer(self, d, targets):
        """
        Create packing helpers between the (param, intv_param) pytrees and the
        global parameter vector
        ``W = (vec A, beta, gamma, varsigma_1, ..., varsigma_M)``,
        where each ``varsigma_k`` collects only the shift entries of the
        *targeted* coordinates of environment ``k`` (the matrices ``R_k``
        of the thesis).
        """
        targets = onp.asarray(targets)
        n_envs = targets.shape[0]
        env_idx, var_idx = onp.nonzero(targets)
        n_shift = len(env_idx)

        iA = onp.arange(d * d)
        ibeta = d * d + onp.arange(d)
        igamma = d * d + d + onp.arange(d)
        ishift = d * d + 2 * d + onp.arange(n_shift)
        d_W = d * d + 2 * d + n_shift

        layout = dict(A=iA, beta=ibeta, gamma=igamma, shift=ishift,
                      env_idx=env_idx, var_idx=var_idx, d_W=d_W,
                      n_envs=n_envs, d=d)

        def pack(param, intv_param):
            shift = jnp.asarray(intv_param["shift"])
            return jnp.concatenate([
                jnp.asarray(param["weights"]).reshape(-1),
                jnp.asarray(param["biases"]),
                jnp.asarray(param["noise_var"]),
                shift[env_idx, var_idx],
            ])

        def unpack(W):
            A = W[:d * d].reshape(d, d)
            beta = W[d * d:d * d + d]
            gamma = W[d * d + d:d * d + 2 * d]
            shift = jnp.zeros((n_envs, d)).at[env_idx, var_idx].set(W[ishift])
            return dict(weights=A, biases=beta, noise_var=gamma), shift

        return pack, unpack, layout


def project_simplex(v, total=1.0):
    v = jnp.asarray(v)
    n = v.shape[0]
    u = jnp.sort(v)[::-1]
    css = jnp.cumsum(u) - total
    ind = jnp.arange(1, n + 1)
    cond = u - css / ind > 0
    rho = jnp.max(jnp.where(cond, ind, 0))
    tau = (jnp.cumsum(u)[rho - 1] - total) / rho
    return jnp.clip(v - tau, 0.0, None)


def project_simplex_lb(v, total=1.0, lb=0.0):
    v = jnp.asarray(v)
    n = v.shape[0]
    return lb + project_simplex(v - lb, total - n * lb)


# joint and pooled multi-environment objectives
def _pairwise_ustat(h_fn, x, args):
    """U-statistic (1 / (n (n-1))) sum_{i != j} h(x_i, x_j) of a symmetrized kernel."""
    n = x.shape[0]
    hmat = vmap(vmap(h_fn, (None, 0, None)), (0, None, None))(x, x, args)
    return (hmat.sum() - jnp.trace(hmat)) / (n * (n - 1))


def joint_ustat(model, kernel, xs, targets, weights=None, pair="skds"):
    """
    Stratified U-statistic multi-environment objective for a generic pairwise
    kernel: ``pair="skds"`` uses the Stein kernel,
    ``pair="kds"`` uses the generator kernel.
    """
    from stadion.kds import kds_pair_a

    n_envs = len(xs)
    if weights is None:
        weights = onp.ones(n_envs) / n_envs
    weights = jnp.asarray(weights, dtype=jnp.float32)

    def f_single(x, param, shift):
        return model.f(x, param, {"shift": shift} if shift is not None else None)

    def a_single(x, param, shift):
        return model.a(x, param, {"shift": shift} if shift is not None else None)

    if pair == "skds":
        h_fn = skds_pair_a(f_single, a_single, kernel)
    elif pair == "kds":
        h_fn = kds_pair_a(f_single, a_single, kernel)
    else:
        raise ValueError(f"Unknown pair kernel `{pair}`")

    def h_wrapped(x, y, args):
        param, shift = args
        return h_fn(x, y, param, shift)

    xs = tuple(jnp.asarray(x) for x in xs)

    def loss(param, intv_param):
        shifts = intv_param["shift"]
        total = 0.0
        for k in range(n_envs):
            total = total + weights[k] * _pairwise_ustat(h_wrapped, xs[k], (param, shifts[k]))
        return total

    return loss


def joint_skds_ustat(model, kernel, xs, targets, weights=None):
    """
    Build the stratified U-statistic estimator of the joint SKDS.
    """
    return joint_ustat(model, kernel, xs, targets, weights, pair="skds")


def pooled_skds_ustat(model, kernel, xs, targets, weights=None):
    """
    Build the shared-witness objective
    This estimator exists for comparison only.
    """
    n_envs = len(xs)
    if weights is None:
        weights = onp.ones(n_envs) / n_envs
    weights = jnp.asarray(weights, dtype=jnp.float32)

    def f_single(x, param, shift):
        return model.f(x, param, {"shift": shift} if shift is not None else None)

    def a_single(x, param, shift):
        return model.a(x, param, {"shift": shift} if shift is not None else None)

    def _h_mixed_a(f, a_fn, kern):
        grad_x_kernel = jax.grad(kern, argnums=0)
        grad_y_kernel = jax.grad(kern, argnums=1)
        grad_xy_kernel = jax.jacfwd(grad_x_kernel, argnums=1)

        def h(x, y, args_x, args_y):
            b_x, b_y = f(x, *args_x), f(y, *args_y)
            a_x, a_y = a_fn(x, *args_x), a_fn(y, *args_y)
            k_xy = kern(x, y)
            dk_x = grad_x_kernel(x, y)
            dk_y = grad_y_kernel(x, y)
            d2k_xy = grad_xy_kernel(x, y)
            return 4.0 * (b_x @ b_y) * k_xy \
                   + 2.0 * b_x @ (a_y @ dk_y) \
                   + 2.0 * b_y @ (a_x @ dk_x) \
                   + jnp.sum((a_x @ a_y) * d2k_xy)
        return h

    h_fn = _h_mixed_a(lambda x, p, s: f_single(x, p, s),
                      lambda x, p, s: a_single(x, p, s),
                      kernel)

    xs = tuple(jnp.asarray(x) for x in xs)

    def loss(param, intv_param):
        shifts = intv_param["shift"]
        total = 0.0
        for k in range(n_envs):
            for l in range(n_envs):
                args_k = (param, shifts[k])
                args_l = (param, shifts[l])
                if k == l:
                    def h_sym(x, y, args):
                        return h_fn(x, y, args, args)
                    term = _pairwise_ustat(h_sym, xs[k], args_k)
                else:
                    hmat = vmap(vmap(h_fn, (None, 0, None, None)),
                                (0, None, None, None))(xs[k], xs[l], args_k, args_l)
                    term = hmat.mean()
                total = total + weights[k] * weights[l] * term
        return total

    return loss


# exact quadratic form of the empirical objective 
def assemble_quadratic(model, kernel, xs, targets, weights=None, objective="skds",
                       verbose=True):
    """
    Assemble the exact quadratic representation of a stratified multi-environment
    objective for the linear model class.
    The matrix is computed as one half of the (constant) Hessian of the loss
    with respect to the packed parameter vector, which is exact because the
    loss is a quadratic form.
    """
    d = xs[0].shape[-1]
    pack, unpack, layout = model.packer(d, targets)
    if objective in ("skds", "kds"):
        loss = joint_ustat(model, kernel, xs, targets, weights, pair=objective)
    elif objective == "pooled":
        loss = pooled_skds_ustat(model, kernel, xs, targets, weights)
    else:
        raise ValueError(f"Unknown objective `{objective}`")

    def loss_vec(W):
        param, shift = unpack(W)
        return loss(param, {"shift": shift})

    W0 = jnp.zeros(layout["d_W"])
    H = jax.jit(jax.hessian(loss_vec))(W0)
    H = 0.5 * (H + H.T)  # symmetrize
    H = onp.array(H, dtype=onp.float64) / 2.0  
    return H, pack, unpack, layout


def resolve_ridge(H, lam="auto", lam_floor=1e-4, margin=1.1):
    """
    Resolve the ridge strength. With ``lam="auto"``, choose
    ``lam = max(lam_floor, margin * (-lambda_min(H))_+)`` so that
    ``H + lam I`` is positive definite.
    """
    if lam == "auto":
        eig_min = float(onp.linalg.eigvalsh(H)[0])
        lam = max(lam_floor, margin * max(0.0, -eig_min))
    return float(lam)


def solve_exact(H, layout, lam=1e-3, trQ=2.0, normalization="trQ"):
    """
    Exact minimizer of the ridge-regularized quadratic objective under the
    chosen normalization.

    Under ``normalization="trQ"``: solved via the KKT system.

    Under ``normalization="isotropic"``, the gamma block is fixed at
    ``trQ / d`` and the remaining coordinates solve the (strongly convex)
    ridge-regularized linear system.

    Pass ``lam="auto"`` to pick the smallest ridge rendering ``H + lam I``
    positive definite.
    """
    lam = resolve_ridge(H, lam)
    d_W = layout["d_W"]
    g = layout["gamma"]
    M = H + lam * onp.eye(d_W)

    if normalization == "isotropic":
        gamma_fix = onp.full(len(g), trQ / len(g))
        r = onp.setdiff1d(onp.arange(d_W), g)
        M_rr = M[onp.ix_(r, r)]
        M_rg = M[onp.ix_(r, g)]
        eig_min = float(onp.linalg.eigvalsh(M_rr)[0])
        assert eig_min > 0, \
            f"H_rr + lam I is not positive definite (min eig {eig_min:.2e}); increase lam."
        W_star = onp.zeros(d_W)
        W_star[g] = gamma_fix
        W_star[r] = -onp.linalg.solve(M_rr, M_rg @ gamma_fix)
        info = dict(gamma_min=float(gamma_fix.min()), feasible=True,
                    value=float(W_star @ M @ W_star), lam=lam)
        return W_star, info

    c = onp.zeros(d_W)
    c[g] = 1.0

    eig_min = float(onp.linalg.eigvalsh(M)[0])
    assert eig_min > 0, \
        f"H + lam I is not positive definite (min eig {eig_min:.2e}); increase lam."
    Minv_c = onp.linalg.solve(M, c)
    W_star = trQ * Minv_c / (c @ Minv_c)

    gamma = W_star[g]
    info = dict(gamma_min=float(gamma.min()),
                feasible=bool(gamma.min() >= -1e-9),
                value=float(W_star @ M @ W_star),
                lam=lam)
    return W_star, info


# generic (gradient-descent) joint fitting pipeline in repo style
def fit_joint(
    model,
    key,
    x,
    targets,
    weights=None,
    kernel=None,
    bandwidth="median",
    learning_rate=0.01,
    steps=2000,
    reg=1e-3,
    optimizer="adam",
    warm_start_intv=True,
    warm_start_mechanism=True,
    batch_size=None,
    verbose=10,
):
    """
    Practical joint-learning pipeline: full-batch gradient descent 
    on the stratified U-statistic JSKDS objective with ridge
    regularization.
    """
    xs = [jnp.asarray(x_, dtype=jnp.float32) for x_ in x]
    targets = onp.stack([onp.asarray(t) for t in targets])
    n_envs = len(xs)
    d = xs[0].shape[-1]
    model.n_vars = d

    if kernel is None:
        if bandwidth == "median":
            bandwidth = median_heuristic_bandwidth(xs)
        kernel = partial(rbf_kernel, bandwidth=bandwidth)

    key, subk = random.split(key)
    param = model.init_param(subk, d)
    key, subk = random.split(key)
    intv_param = model.init_intv_param(subk, d, n_envs=n_envs, targets=targets,
                                       x=xs if warm_start_intv else None)

    if warm_start_mechanism and all(k in param for k in ("weights", "biases", "noise_var")):
        # reversible moment initialization
        trQ = getattr(model, "trQ", 2.0)
        m0_hat = onp.asarray(xs[0]).mean(0)
        sigma0_hat = onp.cov(onp.asarray(xs[0]).T) + 1e-6 * onp.eye(d)
        A0 = -(trQ / (2.0 * d)) * onp.linalg.inv(sigma0_hat)
        param["weights"] = jnp.asarray(A0, dtype=jnp.float32)
        param["biases"] = jnp.asarray(-A0 @ m0_hat, dtype=jnp.float32)
        param["noise_var"] = (trQ / d) * jnp.ones(d, dtype=jnp.float32)

    if batch_size is None:
        loss_fun = joint_skds_ustat(model, kernel, xs, targets, weights)
        sample_batches = None
    else:
        sample_batches = True

    if weights is None:
        weights_arr = onp.ones(n_envs) / n_envs
    else:
        weights_arr = onp.asarray(weights)

    def objective(param_tup, xs_batch):
        param_, intv_param_ = param_tup
        if sample_batches:
            loss_ = joint_skds_ustat(model, kernel, xs_batch, targets, weights_arr)(
                param_, intv_param_)
        else:
            loss_ = loss_fun(param_, intv_param_)
        # ridge regularization on all learnable parameters
        sq = sum(jnp.sum(jnp.square(l)) for l in jax.tree_util.tree_leaves(param_._store)) \
             + sum(jnp.sum(jnp.square(l)) for l in jax.tree_util.tree_leaves(intv_param_._store))
        return loss_ + reg * sq, loss_

    value_and_grad = jax.value_and_grad(objective, 0, has_aux=True)

    if optimizer == "adam":
        opt = optax.adam(learning_rate)
    elif optimizer == "sgd":
        opt = optax.sgd(learning_rate)
    else:
        raise ValueError(f"Unknown optimizer `{optimizer}`")
    opt_state = opt.init((param, intv_param))

    @jax.jit
    def update_step(param_, intv_param_, opt_state_, xs_batch):
        (l, l_skds), (dparam, dintv) = value_and_grad((param_, intv_param_), xs_batch)
        dparam = dparam.masked(grad=True)
        dintv = dintv.masked(grad=True)
        (pu, iu), opt_state_ = opt.update((dparam, dintv), opt_state_, (param_, intv_param_))
        pu = pu.masked(grad=True)
        iu = iu.masked(grad=True)
        param_, intv_param_ = optax.apply_updates((param_, intv_param_), (pu, iu))
        # constraint projection (normalization + nonnegativity)
        if hasattr(model, "project_params"):
            param_, intv_param_ = model.project_params(param_, intv_param_)
        return param_, intv_param_, opt_state_, l, l_skds

    rng = onp.random.default_rng(int(random.randint(key, (), 0, 2 ** 31 - 1)))
    log_every = max(1, steps // verbose) if verbose else steps + 1
    for t in range(steps):
        if sample_batches:
            xs_batch = tuple(
                jnp.asarray(x_[rng.choice(x_.shape[0], size=min(batch_size, x_.shape[0]),
                                          replace=False)])
                for x_ in xs
            )
        else:
            xs_batch = None
        param, intv_param, opt_state, l, l_skds = update_step(param, intv_param, opt_state, xs_batch)
        if verbose and (t % log_every == 0 or t == steps - 1):
            print(f"step {t: >5d}  objective: {float(l): >12.6f}  jskds: {float(l_skds): >12.6f}",
                  flush=True)

    model.param = param
    model.intv_param = intv_param
    return model


# closed-form stationary laws and evaluation metrics 
def ou_stationary_moments(A, Q, beta, shift=None):
    """
    Stationary law N(m, Sigma) of ``dX = (A X + beta + shift) dt + G dW``,
    ``Q = G G^T``: solves ``A m + beta + shift = 0`` and the Lyapunov equation
    ``A Sigma + Sigma A^T + Q = 0``. Requires A Hurwitz.
    """
    A = onp.asarray(A, dtype=onp.float64)
    Q = onp.asarray(Q, dtype=onp.float64)
    b = onp.asarray(beta, dtype=onp.float64).copy()
    if shift is not None:
        b = b + onp.asarray(shift, dtype=onp.float64)
    m = -onp.linalg.solve(A, b)
    Sigma = sla.solve_continuous_lyapunov(A, -Q)
    return m, Sigma


def gaussian_w2(m1, S1, m2, S2):
    """Closed-form Wasserstein-2 distance between two Gaussians."""
    m1, m2 = onp.asarray(m1), onp.asarray(m2)
    S1, S2 = onp.asarray(S1), onp.asarray(S2)
    sqS1 = sla.sqrtm(S1).real
    cross = sla.sqrtm(sqS1 @ S2 @ sqS1).real
    w2sq = onp.sum((m1 - m2) ** 2) + onp.trace(S1 + S2 - 2 * cross)
    return float(onp.sqrt(max(w2sq, 0.0)))

# moment-based recovery 
def theorem45_params(Lam, Sigma, m0, means, targets, obs_index=0):
    """
    Recover the shared reversible OU parameters from ``Lambda`` via
    Theorem 4.5:
    """
    Lam = onp.asarray(Lam, dtype=onp.float64)
    Sigma = onp.asarray(Sigma, dtype=onp.float64)
    Sigma_inv = onp.linalg.inv(Sigma)
    A = -Lam @ Sigma_inv
    Q = 2.0 * Lam
    beta = Lam @ Sigma_inv @ onp.asarray(m0)

    shifts = []
    for k, m in enumerate(means):
        if k == obs_index:
            shifts.append(onp.zeros_like(beta))
        else:
            shifts.append(Lam @ Sigma_inv @ (onp.asarray(m) - onp.asarray(m0)))
    return dict(A=A, Q=Q, beta=beta, shifts=shifts)
