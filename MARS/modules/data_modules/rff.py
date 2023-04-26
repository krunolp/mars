import jax
import jax.numpy as jnp

from tensorflow_probability.substrates import jax as tfp
from functools import partial
from jax import jit, vmap
from tqdm import tqdm

tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels


def get_data(num_pts: int, num_fns: int, num_iters: int = 30, key: jax.random.PRNGKey = None, num_features: int = 1000,
             lengthscale: float = 1., x_min: float = -3., x_max: float = 3., kernel: str = 'eq',
             coefficient: float = 1., max_num_pts: int = 1000):
    """ Obtains the dataset containing functional evaluations of GPS usinr RFF (random Fourier features). """
    assert num_pts <= max_num_pts
    keys = iter(jax.random.split(key, int(num_iters + 5)))
    x = jnp.linspace(x_min, x_max, max_num_pts)[:, None]
    y = sample_rff(x=x,
                   kernel=kernel,
                   lengthscale=lengthscale,
                   coefficient=coefficient,
                   num_functions=num_fns,
                   num_features=num_features,
                   key=key)

    index_pts = []
    fun_evals = []
    scores = []
    for _ in tqdm(range(num_iters + 1), desc="Obtaining RFF data"):
        indices = jax.random.randint(next(keys), (num_pts,), minval=0, maxval=max_num_pts)
        xs = x[indices]
        ys = y[indices].T
        index_pts.append(xs)
        fun_evals.append(ys)
        scores.append(vmap(lambda y_: get_score(xs, y_))(ys))
    return iter(index_pts), iter(fun_evals), iter(scores)


@partial(jit, static_argnums=(1, 3, 4, 6))
def sample_rff(x: jnp.ndarray, kernel: psd_kernels, lengthscale: float, num_functions: int, num_features: int,
               key: jax.random.PRNGKey, coefficient: float = 1.) -> jnp.ndarray:
    """ Estimates functional samples from a GP distribution using random Fourier features (RFF). """
    keys = iter(jax.random.split(key, 3))
    # Dimension of data space
    x_dim = x.shape[-1]
    omega_shape = (num_functions, num_features, x_dim)

    # Handle each of three possible kernels separately
    if kernel == 'eq':
        omega = jax.random.normal(next(keys), shape=omega_shape)

    elif kernel == 'laplace':
        omega = jax.random.cauchy(next(keys), shape=omega_shape)

    elif kernel == 'cauchy':
        omega = jax.random.laplace(next(keys), shape=omega_shape)

    else:
        raise NotImplementedError

    # Scale omegas by lengthscale -- same operation for all three kernels
    omega = omega / lengthscale

    weights = jax.random.normal(next(keys), shape=(num_functions, num_features))

    phi = jax.random.uniform(next(keys), minval=0.,
                             maxval=(2 * jnp.pi),
                             shape=(num_functions, num_features, 1))

    features = jnp.cos(jnp.einsum('sfd, nd -> sfn', omega, x) + phi)
    features = (2 / num_features) ** 0.5 * features * coefficient

    functions = jnp.einsum('sf, sfn -> sn', weights, features).T

    return functions


@partial(jit, static_argnums=(2,))
def get_score(xs: jnp.ndarray, ys: jnp.ndarray, kernel: psd_kernels = psd_kernels.ExponentiatedQuadratic()):
    """ Helper function for calculation of the GP score. """
    gp = tfd.GaussianProcess(
        kernel, index_points=xs, observation_noise_variance=1e-7
    )
    return jax.grad(lambda x: jnp.squeeze(gp.log_prob(x)))(ys)
