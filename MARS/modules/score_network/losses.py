import haiku as hk
from typing import Union, Any
from tensorflow_probability.substrates import jax as tfp
from jax.tree_util import PyTreeDef
from jax import vmap, jacrev
from MARS.modules.score_network.kernels import *
from functools import partial

jax.config.update("jax_enable_x64", False)

tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels

# Set to False by default, unless debugging:
KeyArray = Union[Any, jax.random.PRNGKey]


def spectr_norm(x):
    return hk.SpectralNorm()(x)


def get_f(parameters):
    f_stateful = hk.transform_with_state(spectr_norm)
    f_stateful = hk.without_apply_rng(f_stateful)

    # initialize to obtain an initial state
    params_spectr, state_spectr = f_stateful.init(jax.random.PRNGKey(2), parameters)
    # wrap initial state & don't return further states
    init_stateless = lambda rng, x: f_stateful.init(rng, x)[0]
    apply_stateless = lambda params, x: f_stateful.apply(params, state_spectr, x)[0]
    # spectral_normalisation = hk.Transformed(init_stateless, apply_stateless)
    return hk.Transformed(init_stateless, apply_stateless)


def rkhs_norm(f, k):
    # c = jax.scipy.linalg.solve(k, f)
    # for numerical stability
    c = jax.numpy.linalg.lstsq(k, f)
    return c.T @ k @ c


class score_net_loss:
    def __init__(self, loss_type: str, nn: Any, x_dim: int,
                 spectr_penalty_multiplier: float = 1., rkhs_pen_coeff: float = 1e-4,
                 bandwidth: float = None, spectr_norm_const: float = 1., grad_pen_const: float = 1.) -> None:

        self.nn = nn
        self.x_dim = x_dim
        self.spectr_penalty_multiplier = spectr_penalty_multiplier
        self.spectr_norm_const = spectr_norm_const
        self.grad_pen_const = grad_pen_const
        self.rkhs_pen_coeff = rkhs_pen_coeff
        self.bandwidth = bandwidth

        if loss_type == 'exact_sm' or loss_type == 'exact_w_spectr_norm':
            self.apply = self.exact_sm
        elif loss_type == 'exact_w_grad_pen':
            self.apply = self.exact_sm_with_grad_penalties
        elif loss_type == 'exact_w_spectr_pen':
            self.apply = self.exact_sm_with_spectr_norm_penalties
        elif loss_type == 'exact_w_kern_pen':
            self.apply = self.exact_sm_with_kern_pen
            raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def aux_nn_apply(self, params_: PyTreeDef, rng_key: Any, x_: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """ Auiliary forward pass for returning the gradients and outputs simultaneously. """
        ret = self.nn.apply(params_, rng_key, x_)
        return ret, ret

    @partial(jax.jit, static_argnums=(0,))
    def exact_sm(self, params: PyTreeDef, x_and_fx: jnp.ndarray, rng_key: Any) -> float:
        """ Exact score matching loss by Hyvarinen. """
        loss1, loss2 = vmap(jacrev(lambda x_: self.aux_nn_apply(params, rng_key, x_), has_aux=True))(x_and_fx)
        loss1 = loss1[..., self.x_dim:].squeeze()
        loss = vmap(jnp.trace)(loss1) + 0.5 * jnp.linalg.norm(loss2, axis=-1) ** 2
        return loss.mean()

    @partial(jax.jit, static_argnums=(0,))
    def exact_w_spectr_norm(self, params: PyTreeDef, x_and_fx: jnp.ndarray, rng_key: Any) -> float:
        """ Exact score matching loss with spectrally normalized layers. """

        loss1, loss2 = vmap(jacrev(lambda x_: self.aux_nn_apply(params, rng_key, x_), has_aux=True))(x_and_fx)
        loss1 = loss1[..., self.x_dim:].squeeze()
        loss = vmap(jnp.trace)(loss1) + 0.5 * jnp.linalg.norm(loss2, axis=-1) ** 2
        return loss.mean()

    @partial(jax.jit, static_argnums=(0,))
    def exact_sm_with_grad_penalties(self, params: PyTreeDef, x_and_fx: jnp.ndarray, rng_key: KeyArray) -> tuple[
        float, jnp.ndarray]:
        """ Exact score matching loss with gradient penalties. """
        keys = iter(jax.random.split(rng_key, 5))
        loss1, loss2 = vmap(jacrev(lambda x_: self.aux_nn_apply(params, next(keys), x_), has_aux=True))(x_and_fx)
        loss1 = loss1[..., self.x_dim:].squeeze()
        loss = vmap(jnp.trace)(loss1) + 0.5 * jnp.linalg.norm(loss2, axis=-1) ** 2

        fx_min = vmap(jnp.min, 1)(x_and_fx[..., self.x_dim:])
        fx_max = vmap(jnp.max, 1)(x_and_fx[..., self.x_dim:])

        rand_sampl = vmap(lambda a, b: jax.random.uniform(key=next(keys), shape=(x_and_fx.shape[0],), minval=a,
                                                          maxval=b))(fx_min, fx_max)
        rand_sampl = jnp.transpose(rand_sampl)
        rand_x_fx = jnp.concatenate((x_and_fx[..., :self.x_dim], rand_sampl[..., None]), -1)

        assert rand_x_fx.shape == x_and_fx.shape

        grad_norm = vmap(jacrev(lambda x_: self.nn.apply(params, next(keys), x_)))(rand_x_fx)[..., self.x_dim:]
        grad_norm = jnp.linalg.norm(grad_norm.reshape((grad_norm.shape[0], -1)), axis=-1)
        grad_penalty = (grad_norm - self.grad_pen_const) ** 2
        return loss.mean() + grad_penalty.mean()

    @partial(jax.jit, static_argnums=(0,))
    def exact_sm_with_spectr_norm_penalties(self, params: PyTreeDef, x_and_fx: jnp.ndarray, rng_key: KeyArray):
        """ Exact score matching loss with spectral penalization. """
        loss1, loss2 = vmap(jacrev(lambda x_: self.aux_nn_apply(params, rng_key, x_), has_aux=True))(x_and_fx)
        loss1 = loss1[..., self.x_dim:].squeeze()
        loss = vmap(jnp.trace)(loss1) + 0.5 * jnp.linalg.norm(loss2, axis=-1) ** 2

        spectr_penalty = 1
        for key_ in params.keys():
            if key_[:16] == 'BaseModel/linear':
                spectr_penalty *= jnp.linalg.norm(params[key_]['w'], ord=2)

        return loss.mean() + self.spectr_penalty_multiplier * spectr_penalty

    @partial(jax.jit, static_argnums=(0,))
    def exact_sm_with_kern_pen(self, params: PyTreeDef, x_and_fx: jnp.ndarray, rng_key: KeyArray) -> float:
        """ Exact score matching loss with RKHS norm penalization. """
        loss1, loss2 = vmap(jacrev(lambda x_: self.aux_nn_apply(params, rng_key, x_), has_aux=True))(x_and_fx)
        loss1 = loss1[..., self.x_dim:].squeeze()
        loss = vmap(jnp.trace)(loss1) + 0.5 * jnp.linalg.norm(loss2, axis=-1) ** 2

        kern_op = vmap(lambda l: CurlFreeIMQp().kernel_operator(l, l, kernel_hyperparams=self.bandwidth,
                                                                compute_divergence=False, return_matr=True,
                                                                flatten_matr=True))(loss2[..., None])
        penalty = vmap(lambda l, k: rkhs_norm(f=l, k=k))(loss2[..., None], kern_op).squeeze()
        return (loss + self.rkhs_pen_coeff * penalty).mean()

    @partial(jax.jit, static_argnums=(0,))
    def spectr_norm_apply(self, params: dict) -> dict:
        """ Spectral normalization of linear layers. """
        for key_ in params.keys():
            if key_[:16] == 'BaseModel/linear':
                params[key_]['w'] = get_f(params[key_]['w']).apply({}, params[key_]['w']) * self.spectr_norm_const
        return params
