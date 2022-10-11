import jax
import jax.numpy as jnp
import optax
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates import jax as tfp
import haiku as hk
import numpy as np

from typing import List, Optional, Callable, Dict, Union
from collections import OrderedDict
from MARS.models.abstract_model import BatchedNeuralNetworkModel
from MARS.modules.data_modules.simulator_base import SinusoidsSim
from functools import partial

jax.config.update("jax_enable_x64", False)


class BNN_FSVGD(BatchedNeuralNetworkModel):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 domain_l: jnp.ndarray,
                 domain_u: jnp.ndarray,
                 rng_key: jax.random.PRNGKey,
                 likelihood_std: float = 0.2,
                 num_particles: int = 10,
                 bandwidth_svgd: float = 0.2,
                 bandwidth_gp_prior: float = 0.2,
                 data_batch_size: int = 16,
                 num_measurement_points: int = 16,
                 num_train_steps: int = 10000,
                 lr=1e-3,
                 normalize_data: bool = True,
                 normalization_stats: Optional[Dict[str, jnp.ndarray]] = None,
                 hidden_layer_sizes: List[int] = (32, 32, 32),
                 hidden_activation: Optional[Callable] = jax.nn.leaky_relu,
                 last_activation: Optional[Callable] = None):
        super().__init__(input_size=input_size, output_size=output_size, rng_key=rng_key,
                         data_batch_size=data_batch_size, num_train_steps=num_train_steps,
                         num_batched_nns=num_particles, hidden_layer_sizes=hidden_layer_sizes,
                         hidden_activation=hidden_activation, last_activation=last_activation,
                         normalize_data=normalize_data, normalization_stats=normalization_stats)
        self.likelihood_std = likelihood_std * jnp.ones(output_size)
        self.num_particles = num_particles
        self.bandwidth_svgd = bandwidth_svgd
        self.bandwidth_gp_prior = bandwidth_gp_prior
        self.num_measurement_points = num_measurement_points

        # check and set domain boundaries
        assert domain_u.shape == domain_l.shape == (self.input_size,)
        assert jnp.all(domain_l <= domain_u), 'lower bound of domain must be smaller than upper bound'
        self.domain_l = domain_l
        self.domain_u = domain_u

        # initialize batched NN
        self.params_stack = self.batched_model.param_vectors_stacked

        # initialize optimizer
        self.optim = optax.adam(learning_rate=lr)
        self.opt_state = self.optim.init(self.params_stack)

        # initialize kernel
        self.kernel_svgd = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=self.bandwidth_svgd)
        self.kernel_gp_prior = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=self.bandwidth_gp_prior)

    def _sample_measurement_points(self, key: jax.random.PRNGKey, num_points: int = 10) -> jnp.ndarray:
        """ Sampling the measurement points according to a Uniform distribution. """
        x_domain = jax.random.uniform(key, shape=(num_points, self.input_size),
                                      minval=self.domain_l, maxval=self.domain_u)
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
        # combine the training data batch with a batch of sampled measurement points
        train_batch_size = x_batch.shape[0]
        x_domain = self._sample_measurement_points(key, num_points=self.num_measurement_points)
        x_stacked = jnp.concatenate([x_batch, x_domain], axis=0)

        # likelihood
        f_raw = self.batched_model.forward_vec(x_stacked, param_vec_stack)
        (_, post_stats), grad_post = jax.value_and_grad(self._neg_log_posterior, has_aux=True)(
            f_raw, x_stacked, y_batch, train_batch_size, num_train_points)

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

    def _nll(self, pred_raw: jnp.ndarray, y_batch: jnp.ndarray, train_data_till_idx: int) -> float:
        """ Calculating the negative log-likelihood of the multivariate diagonal Normal distribution."""
        likelihood_std = self.likelihood_std
        log_prob = tfd.MultivariateNormalDiag(pred_raw[:, :train_data_till_idx, :], likelihood_std).log_prob(y_batch)
        return - jnp.mean(log_prob)

    def _neg_log_posterior(self, pred_raw: jnp.ndarray, x_stacked: jnp.ndarray, y_batch: jnp.ndarray,
                           train_data_till_idx: int, num_train_points: Union[float, int]):
        """ Approximate negative log-posterior distribution given the data and predictions. """
        nll = self._nll(pred_raw, y_batch, train_data_till_idx)
        neg_log_prior = - self._gp_prior_log_prob(x_stacked, pred_raw, eps=1e-3) / num_train_points
        neg_log_post = nll + neg_log_prior
        stats = OrderedDict(train_nll_loss=nll, neg_log_prior=neg_log_prior)
        return neg_log_post, stats

    def _gp_prior_log_prob(self, x: jnp.array, y: jnp.array, eps: float = 1e-3) -> jnp.ndarray:
        """ Explicit log-probability of the marginal GP prior distribution. """
        k = self.kernel_gp_prior.matrix(x, x) + eps * jnp.eye(x.shape[0])
        dist = tfd.MultivariateNormalFullCovariance(jnp.zeros(x.shape[0]), k)
        return jnp.mean(jnp.sum(dist.log_prob(jnp.swapaxes(y, -1, -2)), axis=-1)) / x.shape[0]

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
    seed = 2
    hk_key = hk.PRNGSequence(2)
    noise = 0.01
    domain_l_, domain_u_ = np.array([-5.]), np.array([5.])

    sim = SinusoidsSim()
    num_train_points_ = 1

    true_fn_key = next(hk_key)
    x_train = jax.random.uniform(next(hk_key), shape=(num_train_points_, 1), minval=domain_l_, maxval=domain_u_)
    y_train = sim.sample_function_vals(x_train, 1, true_fn_key)[0, ...] + noise * jax.random.normal(next(hk_key),
                                                                                                    shape=x_train.shape)

    num_test_points = 100
    x_test = jnp.linspace(domain_l_, domain_u_, 100)
    y_test = sim.sample_function_vals(x_test, 1, true_fn_key)[0, ...] + noise * jax.random.normal(next(hk_key),
                                                                                                  shape=x_test.shape)

    normalization_stats_ = {
        'x_mean': jnp.mean(x_test, axis=0),
        'x_std': jnp.std(x_test, axis=0),
        'y_mean': jnp.mean(y_test, axis=0),
        'y_std': jnp.std(y_test, axis=0),
    }

    bnn = BNN_FSVGD(1, 1, domain_l_, domain_u_, rng_key=next(hk_key), likelihood_std=noise,
                    normalization_stats=normalization_stats_, num_train_steps=20000)
    for i in range(10):
        bnn.fit(x_train, y_train, x_eval=x_test, y_eval=y_test, num_steps=5000)
        bnn.plot_1d(x_train, y_train, plot_data=(x_test, y_test),
                    title=f'iter {(i + 1) * 5000}',
                    domain_l=domain_l_, domain_u=domain_u_)
