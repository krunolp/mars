import jax
import jax.numpy as jnp
import optax
import haiku as hk
import tensorflow_probability.substrates.jax.distributions as tfd
import numpy as np

from typing import List, Optional, Callable, Dict, Union
from collections import OrderedDict
from MARS.models.abstract_model import BatchedNeuralNetworkModel
from functools import partial
from tensorflow_probability.substrates import jax as tfp

jax.config.update("jax_enable_x64", False)


class BNN_SVGD(BatchedNeuralNetworkModel):

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 rng_key: jax.random.PRNGKey,
                 likelihood_std: float = 0.2,
                 num_particles: int = 10,
                 bandwidth_svgd: float = 10.0,
                 data_batch_size: int = 16,
                 num_train_steps: int = 10000,
                 lr=1e-3,
                 normalize_data: bool = True,
                 normalization_stats: Optional[Dict[str, jnp.ndarray]] = None,
                 hidden_layer_sizes: List[int] = (32, 32, 32),
                 hidden_activation: Optional[Callable] = jax.nn.leaky_relu,
                 last_activation: Optional[Callable] = None,
                 use_prior: bool = True,
                 weight_prior_std: float = 0.5,
                 bias_prior_std: float = 1e1):
        super().__init__(input_size=input_size, output_size=output_size, rng_key=rng_key,
                         data_batch_size=data_batch_size, num_train_steps=num_train_steps,
                         num_batched_nns=num_particles, hidden_layer_sizes=hidden_layer_sizes,
                         hidden_activation=hidden_activation, last_activation=last_activation,
                         normalize_data=normalize_data, normalization_stats=normalization_stats)
        self.likelihood_std = likelihood_std * jnp.ones(output_size)
        self.num_particles = num_particles
        self.bandwidth_svgd = bandwidth_svgd

        # get batched NN params as a stack of vectors
        self.params_stack = self.batched_model.param_vectors_stacked

        # construct the neural network prior distribution
        self.use_prior = use_prior
        if use_prior:
            self.prior_dist = self._construct_nn_param_prior(weight_prior_std, bias_prior_std)

        # initialize optimizer
        self.optim = optax.adam(learning_rate=lr)
        self.opt_state = self.optim.init(self.params_stack)

        # initialize kernel
        self.kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=self.bandwidth_svgd)

    @partial(jax.jit, static_argnums=(0,))
    def _evaluate_kernel(self, particles: jnp.ndarray):
        """ SVGD kernel evaluation function. """
        particles_copy = jax.lax.stop_gradient(particles)
        k = self.kernel.matrix(particles, particles_copy)
        return jnp.sum(k), k

    def step(self, x_batch: jnp.ndarray, y_batch: jnp.ndarray, num_train_points: Union[float, int]) -> Dict[str, float]:
        self.opt_state, self.params_stack, stats = self._step_jit(self.opt_state, self.params_stack, x_batch, y_batch,
                                                                  num_train_points)
        return stats

    @partial(jax.jit, static_argnums=(0,))
    def _step_jit(self, opt_state: optax.OptState, param_vec_stack: jnp.array, x_batch: jnp.array, y_batch: jnp.array,
                  num_train_points: Union[float, int]):
        # SVGD updates
        (log_post, post_stats), grad_q = jax.value_and_grad(self._neg_log_posterior, has_aux=True)(param_vec_stack,
                                                                                                   x_batch, y_batch,
                                                                                                   num_train_points)
        grad_k, k = jax.grad(self._evaluate_kernel, has_aux=True)(param_vec_stack)
        grad = k @ grad_q + grad_k / self.num_particles

        updates, opt_state = self.optim.update(grad, opt_state, param_vec_stack)
        param_vec_stack = optax.apply_updates(param_vec_stack, updates)

        avg_triu_k = jnp.sum(jnp.triu(k, k=1)) / ((self.num_particles - 1) * self.num_particles / 2)
        stats = OrderedDict(**post_stats, avg_grad_q=jnp.mean(grad_q), avg_grad_k=jnp.mean(grad_q),
                            avg_triu_k=avg_triu_k)

        return opt_state, param_vec_stack, stats

    def _ll(self, param_vec_stack: jnp.ndarray, x_batch: jnp.ndarray, y_batch: jnp.ndarray):
        """ Log-likelihood calculation using the multivariate diagonal Normal distribution. """
        pred_raw = self.batched_model.forward_vec(x_batch, param_vec_stack)
        likelihood_std = self.likelihood_std
        log_prob = tfd.MultivariateNormalDiag(pred_raw, likelihood_std).log_prob(y_batch)
        return jnp.mean(log_prob)

    def _neg_log_posterior(self, param_vec_stack: jnp.ndarray, x_batch: jnp.ndarray, y_batch: jnp.ndarray,
                           num_train_points: Union[float, int]):
        """ Approximate negative log-posterior distribution given the data and predictions. """
        ll = self._ll(param_vec_stack, x_batch=x_batch, y_batch=y_batch)
        if self.use_prior:
            log_prior = jnp.mean(self.prior_dist.log_prob(param_vec_stack))
            log_prior /= (num_train_points * self.prior_dist.event_shape[0])
            stats = OrderedDict(train_nll_loss=-ll, neg_log_prior=-log_prior)
            log_posterior = ll + log_prior
        else:
            log_posterior = ll
            stats = OrderedDict(train_nll_loss=-ll)
        return - log_posterior, stats

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

    def calib_error(self, y_true, pred_dist, use_circular_region=False):
        pass


if __name__ == '__main__':
    seed = 10
    hk_key = hk.PRNGSequence(seed)
    noise = 0.01
    domain_l_, domain_u_ = np.array([-7.]), np.array([7.])

    fun = lambda x: 2 * x + 2 * jnp.sin(2 * x)

    num_train_pts = 10
    x_train = jax.random.uniform(next(hk_key), shape=(num_train_pts, 1), minval=domain_l_, maxval=domain_u_)
    y_train = fun(x_train) + noise * jax.random.normal(next(hk_key), shape=x_train.shape)

    num_test_pts = 100
    x_test = jax.random.uniform(next(hk_key), shape=(num_test_pts, 1), minval=domain_l_, maxval=domain_u_)
    y_test = fun(x_test) + noise * jax.random.normal(next(hk_key), shape=x_test.shape)

    bnn = BNN_SVGD(1, 1, next(hk_key), num_train_steps=20000, bandwidth_svgd=0.2)

    for i in range(10):
        bnn.fit(x_train, y_train, x_eval=x_test, y_eval=y_test, num_steps=2000)
        bnn.plot_1d(x_train, y_train, plot_data=(x_test, y_test), title=f'iter {(i + 1) * 5000}', domain_l=domain_l_,
                    domain_u=domain_u_)
