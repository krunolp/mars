import itertools
import haiku as hk
import gpjax as gpx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import GridSearchCV
from MARS.modules.data_modules.gp_jax import GP as GPJ
from typing import Any, Sequence, Union
from pathlib import Path
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from tqdm import tqdm
from typing import List, Dict

KeyArray = Union[Any, jax.random.PRNGKey]

PARENT_LOG_DIR = Path.home() / "tuning_logs"
PARENT_ERR_DIR = Path.home() / "tuning_errs"


def generate_commands(
        model_names: iter,
        experiments: iter,
        n_cpus=1,
        n_gpus=1,
        mem=1024,
        long=False,
        interpreter='/cluster/project/infk/krause/pgeorges/miniconda3/envs/thesis/bin/python',
        dry=False,
):
    for experiment in experiments:
        log_dir = PARENT_LOG_DIR / experiment
        err_dir = PARENT_ERR_DIR / experiment

        # make directories
        log_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        for model_name in model_names:
            log_path = log_dir / (model_name + '.log')
            err_path = err_dir / (model_name + '.err')

            # delete previous files
            log_path.unlink(missing_ok=True)
            err_path.unlink(missing_ok=True)

            command = (
                f'bsub -N '
                f'-n {int(n_cpus)} '
                f'-W {23 if long else 3}:59 '
                f'-R "rusage[mem={int(mem)}, ngpus_excl_p={int(n_gpus)}]" '
                f'-o {str(log_path)} '
                f'-e {str(err_path)} '
                f'-J {model_name}.{experiment[:3]} '
                f'{interpreter} {experiment}/{model_name}.py '
            )
            if dry:
                import os
                os.system(command)
            else:
                print(command)


def tree_expand_leading_by(pytree, n):
    """
    Converts pytree with leading pytrees with additional `n` leading dimensions
    """
    return jax.tree_util.tree_map(lambda leaf: jax.numpy.expand_dims(leaf, axis=tuple(range(n))), pytree)


def tree_zip_leading(pytree_list):
    """
    Converts n pytrees without leading dimension into one pytree with leading dim [n, ...]
    """
    return jax.tree_util.tree_multimap(
        lambda *args: jax.numpy.stack([*args]) if len(args) > 1 else tree_expand_leading_by(*args, 1),
        *pytree_list)


def tree_unzip_leading(pytree, n):
    """
    Converts pytree with leading dim [n, ...] into n pytrees without the leading dimension
    """
    leaves, treedef = jax.tree_util.tree_flatten(pytree)

    return [
        jax.tree_util.tree_unflatten(treedef, [leaf[i] for leaf in leaves])
        for i in range(n)
    ]


def plot_gpr_samples(gpr_model, n_samples, ax, x_range, y_range=None, c='BuPu', fill='Blue'):
    """Plot samples drawn from the Gaussian process model.

    If the Gaussian process model is not trained then the drawn samples are
    drawn from the prior distribution. Otherwise, the samples are drawn from
    the posterior distribution. Be aware that a sample here corresponds to a
    function.

    Parameters
    ----------
    gpr_model : `GaussianProcessRegressor`
        A :class:`~sklearn.gaussian_process.GaussianProcessRegressor` model.
    n_samples : int
        The number of samples to draw from the Gaussian process distribution.
    ax : matplotlib axis
        The matplotlib axis where to plot the samples.
    x_range : abscissa range
        The matplotlib range of the x-axis.
    y_range : ordinate range
        The matplotlib range of the y-axis.
    c : colour/colour map
        The matpltlib colourmap.
    fill : fill colour
        The matplotlib filling colour.
    """
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    sns.set(rc={"figure.dpi": 300, 'savefig.dpi': 300})
    sns.set_context('notebook')
    x = jnp.linspace(x_range[0], x_range[1], 100)
    X = x.reshape(-1, 1)

    y_mean, y_std = gpr_model.predict(X, return_std=True)
    y_samples = gpr_model.sample_y(X, n_samples)

    for idx, single_prior in enumerate(y_samples.T):
        ax.plot(
            x,
            single_prior,
            linestyle="-",
            alpha=0.4,
            color=c
        )
    ax.plot(x, y_mean, color="#42529C", label="Mean")
    ax.fill_between(
        x,
        y_mean - y_std,
        y_mean + y_std,
        alpha=0.15,
        color=fill,
        label=r"$\pm$ 1 std. dev.",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if y_range is not None:
        ax.set_ylim(y_range)


def normalize(x: jnp.ndarray, _mean: jnp.ndarray, _std: jnp.ndarray, eps: float = 1e-8):
    return (x - _mean[None, ...]) / (_std[None, ...] + eps)


def get_gps_jax(meta_train_data, hk_key) -> tuple[list, dict, tuple[jnp.ndarray, jnp.ndarray]]:
    stats, normalized_data = get_overall_norm_stats(meta_train_data)
    gps, mins, maxs, all_scores, params = [], [], [], [], {}

    kernel = gpx.Matern52()
    for x_, y_ in tqdm(zip(*normalized_data), total=len(meta_train_data), desc='Fitting GPs to meta tasks.'):
        gpr = GPJ(gp_kernel=kernel, random_state=hk_key)
        gps.append(gpr.fit(x_, y_))
        mins.append(x_.min(axis=0))
        maxs.append(x_.max(axis=0))

    mins, maxs = jnp.array(mins).min(axis=0), jnp.array(maxs).max(axis=0)
    return gps, stats, (mins, maxs)


def get_gps(meta_train_data) -> tuple[list, dict, tuple[jnp.ndarray, jnp.ndarray]]:
    stats, normalized_data = get_overall_norm_stats(meta_train_data)
    gps, mins, maxs, all_scores, params = [], [], [], [], {}

    for x_, y_ in tqdm(zip(*normalized_data), total=len(meta_train_data), desc='Performing CV for GP hyperparam.'):
        _, param_scores, params = cv(x_, y_, 'neg_mean_squared_error')
        all_scores.append(param_scores)

    kernel = params[jnp.argmax(jnp.array(all_scores).mean(axis=0))]['kernel']
    for x_, y_ in tqdm(zip(*normalized_data), total=len(meta_train_data), desc='Fitting GPs to meta tasks.'):
        gpr = GaussianProcessRegressor(kernel=kernel)
        gps.append(gpr.fit(x_, y_))
        mins.append(x_.min(axis=0))
        maxs.append(x_.max(axis=0))

    mins, maxs = jnp.array(mins).min(axis=0), jnp.array(maxs).max(axis=0)
    return gps, stats, (mins, maxs)


class BiasInitializer(hk.initializers.Initializer):
    """Custom scaling initializer according to https://arxiv.org/pdf/1903.11482.pdf."""

    def __init__(self, domain_l=None, domain_u=None, n=5, hull_pos=True, scaling="ball", key: hk.PRNGSequence = None):
        """Constructs the :class:`BiasInitializer` initializer.
        """
        self.hull_pos = hull_pos
        self.n = n if self.hull_pos else jax.random.randint(key, shape=(1,), minval=1, maxval=5)
        self.scaling = scaling
        self.domain_l = domain_l
        self.domain_u = domain_u

    def __call__(self, weights: jnp.ndarray, shape: Sequence[int],
                 key: jax.random.PRNGKey, dtype: Any = None) -> jnp.ndarray:
        key, subkey = jax.random.split(key)
        if self.scaling is None:
            pass
        elif self.scaling == "sphere":
            weights = weights / jnp.linalg.norm(weights)
        elif self.scaling == "ball":
            weights = weights / jnp.linalg.norm(weights) * jax.random.uniform(key,
                                                                              shape=weights.shape, minval=0,
                                                                              maxval=2)
        else:
            raise NotImplementedError

        biases = []
        for _ in range(jnp.array(shape).prod()):
            key1, key2, subkey = jax.random.split(subkey, 3)
            rand_weight = jax.random.choice(key2, weights.flatten())

            x_star = jax.random.uniform(key2, shape=(1,), minval=self.domain_l, maxval=self.domain_u)
            bias = -x_star * rand_weight
            biases.append(bias)

        biases = jnp.array(biases, dtype=dtype).reshape(shape) if dtype is not None else jnp.array(biases).reshape(
            shape)

        return biases


class hk_BiasInitializer(hk.initializers.Initializer):
    """Custom scaling initializer according to https://arxiv.org/pdf/1903.11482.pdf."""

    def __init__(self, domain_l=None, domain_u=None, n=5, hull_pos=True, scaling=None, key: jax.random.PRNGKey = None):
        """Constructs the :class:`BiasInitializer` initializer.
        """
        self.hull_pos = hull_pos
        self.n = n if self.hull_pos else jax.random.randint(key, shape=(1,), minval=1, maxval=5)
        self.scaling = scaling
        self.domain_l = domain_l
        self.domain_u = domain_u
        self.key = key

    def __call__(self, shape: Sequence[int], dtype: Any = None) -> jnp.ndarray:

        temp_shape = (shape[0], 1)
        temp_init = jax.nn.initializers.kaiming_normal()
        weights = temp_init(next(self.key), temp_shape, jnp.float32)

        if self.scaling is None:
            pass
        elif self.scaling == "sphere":
            weights = weights / jnp.linalg.norm(weights)
        elif self.scaling == "ball":
            weights = weights / jnp.linalg.norm(weights) * jax.random.uniform(next(self.key),
                                                                              shape=weights.shape, minval=0,
                                                                              maxval=2)
        else:
            raise NotImplementedError

        biases = []
        for _ in range(jnp.array(temp_shape).prod()):
            rand_weight = jax.random.choice(next(self.key), weights.flatten())
            x_star = jax.random.uniform(next(self.key), shape=(1,), minval=self.domain_l, maxval=self.domain_u)
            bias = -x_star * rand_weight
            biases.append(bias)

        biases = jnp.array(biases, dtype=dtype).reshape(shape) if dtype is not None else jnp.array(biases).reshape(
            shape)

        return biases


def cv(x, y, score='neg_mean_squared_error'):
    param_grid = [{
        "kernel": [param[0] * RBF(param[1]) for param in
                   list(itertools.product(np.logspace(-3, 3, 2), np.logspace(-3, 3, 2)))]
    }, {
        "kernel": [param[0] * Matern(length_scale=param[1], nu=param[2]) for param in
                   list(itertools.product(np.logspace(-3, 3, 2), np.logspace(-3, 3, 2), [2.5]))]
    }]

    # param_grid = [{ #fixme
    #     "kernel": [param[0] * RBF(param[1]) for param in
    #                list(itertools.product(np.logspace(-3, 3, 10), np.logspace(-3, 3, 10)))]
    # }, {
    #     "kernel": [param[0] * Matern(length_scale=param[1], nu=param[2]) for param in
    #                list(itertools.product(np.logspace(-3, 3, 10), np.logspace(-3, 3, 10), [1.5, 2.5]))]
    # }]

    gp = GaussianProcessRegressor()

    clf = GridSearchCV(estimator=gp, param_grid=param_grid, cv=4,
                       scoring='%s' % score)
    clf.fit(x, y)

    return clf.best_params_['kernel'], clf.cv_results_['mean_test_score'], clf.cv_results_['params']


def get_overall_norm_stats(meta_data):
    x_concat = jnp.concatenate([jnp.array(x) for x, _ in meta_data], axis=0)
    f_concat = jnp.concatenate([jnp.array(f) for _, f in meta_data], axis=0)
    x_concat = x_concat[..., None] if x_concat.ndim == 1 else x_concat
    f_concat = f_concat[..., None] if f_concat.ndim == 1 else f_concat
    assert x_concat.ndim == f_concat.ndim == 2

    normalization_stats = {
        'x_mean': jnp.mean(x_concat, 0),
        'x_std': jnp.std(x_concat, 0),
        'y_mean': jnp.mean(f_concat, 0),
        'y_std': jnp.std(f_concat, 0),
    }

    x_normalized, f_normalized = [], []
    for x_, y_ in meta_data:
        x_ = x_[..., None] if x_.ndim == 1 else x_
        y_ = y_[..., None] if y_.ndim == 1 else y_
        x_ = normalize(x_, normalization_stats['x_mean'], normalization_stats['x_std'])
        y_ = normalize(y_, normalization_stats['y_mean'], normalization_stats['y_std'])
        x_normalized.append(x_)
        f_normalized.append(y_)

    return normalization_stats, [x_normalized, f_normalized]


def tree_stack(trees):
    """Takes a list of trees and stacks every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list, treedef_list = list(zip(*[jax.tree_flatten(tree) for tree in trees]))
    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(leaves) for leaves in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def tree_unstack(tree):
    """Takes a tree and turns it into a list of trees. Inverse of tree_stack.
    For example, given a tree ((a, b), c), where a, b, and c all have first
    dimension k, will make k trees
    [((a[0], b[0]), c[0]), ..., ((a[k], b[k]), c[k])]
    Useful for turning the output of a vmapped function into normal objects.
    """
    leaves, treedef = jax.tree_flatten(tree)
    n_trees = leaves[0].shape[0]
    new_leaves = [[] for _ in range(n_trees)]
    for leaf in leaves:
        for i in range(n_trees):
            new_leaves[i].append(leaf[i])
    new_trees = [treedef.unflatten(leaves) for leaves in new_leaves]
    return new_trees


class RngKeyMixin:

    def __init__(self, rng_key: jax.random.PRNGKey):
        self._rng_key = rng_key

    def _next_rng_key(self) -> jax.random.PRNGKey:
        new_key, self._rng_key = jax.random.split(self._rng_key)
        return new_key

    @property
    def rng_key(self) -> jax.random.PRNGKey:
        return self._next_rng_key()


def aggregate_stats(stats_list: List[Dict]) -> Dict:
    return jax.tree_map(jnp.mean, tree_stack(stats_list))


def numeric_integration(x, y):
    int_a, int_b = 0, 0

    x_neg, x_pos = jnp.split(x, 2)
    y_neg, y_pos = jnp.split(y, 2)

    x_neg, y_neg = jnp.flip(x_neg), jnp.flip(y_neg)

    lens_pos, lens_neg = [], []
    for j in range(int(len(y) / 2) - 1):
        int_a += jnp.trapz(y_pos[j:j + 2], x_pos[j:j + 2])
        lens_pos.append(int_a)

        int_b += jnp.trapz(y_neg[j:j + 2], x_neg[j:j + 2])
        lens_neg.append(int_b)
    y_combined = jnp.hstack((jnp.flip(jnp.array(lens_neg)), jnp.array(lens_pos)))
    x_combined = jnp.hstack((jnp.flip(x_neg), x_pos))[1:-1]
    final_x = x_combined
    final_y = jnp.exp(y_combined) / jnp.sum(jnp.exp(y_combined))
    return final_x, final_y


def weighted_loss(preds, true, weight=5.):
    diff = preds - true
    loss = jnp.nanmean(jnp.abs(diff * (diff >= 0))) * weight + jnp.nanmean(jnp.abs(diff * (diff < 0)))
    return loss
