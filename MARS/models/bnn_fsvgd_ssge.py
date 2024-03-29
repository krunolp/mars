import jax
import jax.numpy as jnp
import optax
import haiku as hk
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates import jax as tfp
import warnings
import numpy as np

from typing import List, Optional, Callable, Dict, Union
from collections import OrderedDict
from MARS.models.abstract_model import BatchedNeuralNetworkModel
from functools import partial
from MARS.modules.ssge.estimator import SSGE
from MARS.modules.data_modules.simulator_base import GPMetaDataset

jax.config.update("jax_enable_x64", False)


def warn(*args, **kwargs):
    pass


warnings.warn = warn


class BNN_FSVGD_SSGE(BatchedNeuralNetworkModel):

    def __init__(self,
                 input_size: int,
                 rng_key: jax.random.PRNGKey,
                 output_size: int = 1,
                 likelihood_std: float = 0.2,
                 num_particles: int = 10,
                 bandwidth_svgd: float = 0.2,
                 data_batch_size: int = 8,
                 num_measurement_points: int = 8,
                 num_train_steps: int = 10000,
                 lr=1e-3,
                 n_fn_samples=None,
                 x_dim=None,
                 function_sim=None,
                 norm_stats=None,
                 clip_value: float = 1.,
                 normalize_data: bool = True,
                 hidden_layer_sizes: List[int] = (32, 32, 32),
                 hidden_activation: Optional[Callable] = jax.nn.leaky_relu,
                 initializer_b: Callable = jax.nn.initializers.constant(0.01),
                 last_activation: Optional[Callable] = None):
        self.function_sim = function_sim
        self.num_f_samples = n_fn_samples
        self.ndim_x = x_dim
        norm_stats = norm_stats if norm_stats is not None else self.function_sim.normalization_stats

        assert output_size == 1
        super().__init__(input_size=input_size, output_size=output_size, rng_key=rng_key,
                         data_batch_size=data_batch_size, num_train_steps=num_train_steps,
                         num_batched_nns=num_particles, hidden_layer_sizes=hidden_layer_sizes,
                         hidden_activation=hidden_activation, last_activation=last_activation,
                         normalize_data=normalize_data, normalization_stats=norm_stats,
                         initializer_b=initializer_b)

        self.SSGE = SSGE(bandwidth=0.2)

        self.likelihood_std = likelihood_std * jnp.ones(output_size)
        self.num_particles = num_particles
        self.bandwidth_svgd = bandwidth_svgd
        self.num_measurement_points = num_measurement_points
        self.clip_value = clip_value

        # check and set function sim
        assert self.function_sim.output_size == self.output_size and self.function_sim.input_size == self.input_size
        self.ndim_x = self.function_sim.input_size
        # initialize batched NN
        self.params_stack = self.batched_model.param_vectors_stacked

        # initialize optimizer
        self.optim = optax.adam(learning_rate=lr)
        self.opt_state = self.optim.init(self.params_stack)

        # initialize kernel and ssge algo
        self.kernel_svgd = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=self.bandwidth_svgd)

    def _sample_measurement_points(self, key: jax.random.PRNGKey, num_points: int = 10) -> jnp.ndarray:
        """ Sampling the measurement points according to a Uniform distribution. """
        x_domain = self.function_sim.sample_measurement_pts(num_points=num_points, rng_key=key)
        x_domain = self._normalize_data(x_domain)
        assert x_domain.shape == (num_points, self.input_size)
        return x_domain

    @partial(jax.jit, static_argnums=(0,))
    def _evaluate_kernel(self, pred_raw: jnp.ndarray):
        """ fSVGD kernel evaluation function. """
        assert pred_raw.ndim == 3 and pred_raw.shape[-1] == self.output_size
        pred_raw = pred_raw.reshape((pred_raw.shape[0], -1))
        particles_copy = jax.lax.stop_gradient(pred_raw)
        k = self.kernel_svgd.matrix(pred_raw, particles_copy)
        return jnp.sum(k), k

    def step(self, x_batch: jnp.ndarray, y_batch: jnp.ndarray, num_train_points: Union[float, int]) -> Dict[str, float]:
        self.opt_state, self.params_stack, stats = self._step_jit(self.opt_state, self.params_stack, x_batch, y_batch,
                                                                  key=self.rng_key, num_train_points=num_train_points)
        return stats

    @partial(jax.jit, static_argnums=(0,))
    def _surrogate_loss(self, param_vec_stack: jnp.array, x_batch: jnp.array, y_batch: jnp.array,
                        num_train_points: int, key: jax.random.PRNGKey) -> [jnp.ndarray, Dict]:
        """ Approximating the true fSVGD loss function."""
        key1, key2 = jax.random.split(key, 2)

        # combine the training data batch with a batch of sampled measurement points
        train_batch_size = x_batch.shape[0]
        x_domain = self._sample_measurement_points(key1, num_points=self.num_measurement_points)
        x_stacked = jnp.concatenate([x_batch, x_domain], axis=0)

        # likelihood
        f_raw = self.batched_model.forward_vec(x_stacked, param_vec_stack)
        (_, post_stats), grad_post = jax.value_and_grad(self._neg_log_posterior_surrogate, has_aux=True)(
            f_raw, x_stacked, y_batch, train_batch_size, num_train_points, key2)

        # kernel
        grad_k, k = jax.grad(self._evaluate_kernel, has_aux=True)(f_raw)

        surrogate_loss = jnp.sum(f_raw * jax.lax.stop_gradient(jnp.einsum('ij,jkm', k, grad_post)
                                                               + grad_k / self.num_particles))
        avg_triu_k = jnp.sum(jnp.triu(k, k=1)) / ((self.num_particles - 1) * self.num_particles / 2)
        stats = OrderedDict(**post_stats, avg_triu_k=avg_triu_k)
        return surrogate_loss, stats

    @partial(jax.jit, static_argnums=(0,))
    def _step_jit(self, opt_state: optax.OptState, param_vec_stack: jnp.array, x_batch: jnp.array, y_batch: jnp.array,
                  key: jax.random.PRNGKey, num_train_points: Union[float, int]):
        (loss, stats), grad = jax.value_and_grad(self._surrogate_loss, has_aux=True)(
            param_vec_stack, x_batch, y_batch, num_train_points, key)
        updates, opt_state = self.optim.update(grad, opt_state, param_vec_stack)
        param_vec_stack = optax.apply_updates(param_vec_stack, updates)
        return opt_state, param_vec_stack, stats

    def _nll(self, pred_raw: jnp.ndarray, y_batch: jnp.ndarray, train_data_till_idx: int):
        likelihood_std = self.likelihood_std
        log_prob = tfd.MultivariateNormalDiag(pred_raw[:, :train_data_till_idx, :], likelihood_std).log_prob(y_batch)
        return - jnp.mean(log_prob)

    def _neg_log_posterior_surrogate(self, pred_raw: jnp.ndarray, x_stacked: jnp.ndarray, y_batch: jnp.ndarray,
                                     train_data_till_idx: int, num_train_points: Union[float, int],
                                     key: jax.random.PRNGKey):
        """ Approximate negative log-posterior distribution given the data and predictions. """
        nll = self._nll(pred_raw, y_batch, train_data_till_idx)
        prior_score = self._estimate_prior_score(x_stacked, pred_raw, key) / num_train_points
        neg_log_post = nll - jnp.sum(jnp.mean(pred_raw * jax.lax.stop_gradient(prior_score), axis=-2))
        stats = OrderedDict(train_nll_loss=nll)
        return neg_log_post, stats

    def _estimate_prior_score(self, x: jnp.array, pred_raw: jnp.array, key: jax.random.PRNGKey) -> jnp.ndarray:
        """ Approximation of the implicit prior score using SSGE. """
        keys = iter(jax.random.split(key, len(self.function_sim.gps)))
        x_unnormalized = self._unnormalize_data(x)
        f_prior = jnp.array(
            [gp.sample_y(x_unnormalized, n_samples=1, random_state=next(keys)) for gp in self.function_sim.gps])
        f_prior_normalized = self._normalize_y(f_prior)
        if f_prior_normalized.shape[-1] == 1:
            f_prior_normalized = f_prior_normalized.squeeze(-1)
            pred_raw_squeezed = pred_raw.squeeze(-1)
        else:
            raise AssertionError
        ssge_score = self.SSGE.estimate_gradients_s_x(x_query=pred_raw_squeezed, x_sample=f_prior_normalized)
        ssge_score = jnp.expand_dims(ssge_score, axis=-1)
        assert ssge_score.shape == pred_raw.shape
        return ssge_score

    def predict_dist(self, x: jnp.ndarray, include_noise: bool = True) -> tfd.Distribution:
        """ Calculation of the predictive distribution of the network. """
        self.batched_model.param_vectors_stacked = self.params_stack
        x = self._normalize_data(x)
        y_pred_raw = self.batched_model(x)
        pred_dist = self._to_pred_dist(y_pred_raw, likelihood_std=self.likelihood_std, include_noise=include_noise)
        assert pred_dist.batch_shape == x.shape[:-1]
        assert pred_dist.event_shape == (self.output_size,)
        return pred_dist

    def predict_post_samples(self, x: jnp.ndarray) -> jnp.ndarray:
        """ Sampling the posterior sampes from the predictive distribution implied by the model. """
        x = self._normalize_data(x)
        y_pred_raw = self.batched_model(x)
        y_pred = y_pred_raw * self._y_std + self._y_mean
        assert y_pred.ndim == 3 and y_pred.shape[-2:] == (x.shape[0], self.output_size)
        return y_pred


if __name__ == '__main__':
    seed = 10
    hk_key = hk.PRNGSequence(seed)
    num_meas_pts = 3
    num_train_pts = 2
    domain_l, domain_u = np.array([-7.]), np.array([7.])

    # fit the GP
    sim = GPMetaDataset(dataset="sin_2", init_seed=next(hk_key), num_input_pts=num_train_pts + num_meas_pts)

    x_train, y_train, x_test, y_test = sim.meta_test_data[0]

    # initialize fSVGD
    bnn = BNN_FSVGD_SSGE(sim.input_size,
                         rng_key=next(hk_key),
                         lr=1e-3,
                         function_sim=sim,
                         x_dim=sim.input_size,
                         n_fn_samples=10,
                         data_batch_size=num_train_pts,
                         likelihood_std=0.6,
                         num_measurement_points=num_meas_pts)

    # train fSVGD
    for i in range(10):
        eval_stats = bnn.fit(x_train, y_train, x_eval=x_test, y_eval=y_test, num_steps=1000)
        bnn.plot_1d(x_train, y_train, plot_data=(x_test, y_test), title=f'iter {(i + 1) * 5000}', domain_l=domain_l,
                    domain_u=domain_u)
