import jax
import jax.numpy as jnp
from typing import Optional, Union
import pandas as pd
import os
import copy
import yaml
import haiku as hk
import h5py
from tensorflow_probability.substrates import jax as tfp
from functools import partial
import tensorflow_probability.substrates.jax.distributions as tfd
from MARS.modules.utils import get_gps
from sklearn.gaussian_process.kernels import RBF
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from MARS.modules.data_modules.regression_datasets import provide_data
from MARS.modules.utils import get_nns_dropout_sklearn_differentdim, get_nns_dropout_sklearn_multidim

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')

PHYSIONET_URL = 'https://www.dropbox.com/sh/0dq32lu5dr2ut0q/AAAdM0m_NJUpY9H_8s1G8gtZa?dl=1'
PHYSIONET_DIR = os.path.join(DATA_DIR, 'physionet2012')

SWISSFEL_URL = 'https://www.dropbox.com/sh/6wpkv12zqw0o8mh/AACrK8HE8DQoDIpSL4BkoY_9a?dl=1'
SWISSFEL_DIR = os.path.join(DATA_DIR, 'swissfel')

BERKELEY_SENSOR_URL = 'https://www.dropbox.com/sh/f8vo4gwtvmzxfea/AAD9-4vEn1lBOGt6SuXdDeKta?dl=1'
BERKELEY_SENSOR_DIR = os.path.join(DATA_DIR, 'sensor_data')

ARGUS_CONTROL_URL = 'https://www.dropbox.com/sh/kdzqcw2b0rm34or/AAD2XFzgB2PSjGbNtfNER75Ba?dl=1'
ARGUS_CONTROL_DIR = os.path.join(DATA_DIR, 'mhc_data')


class FunctionSimulator:

    def __init__(self, input_size: int = None, output_size: int = 1, finite_fns: bool = False,
                 num_fns: Union[int, None] = None,
                 init_seed: jax.random.PRNGKey = None):
        self.input_size = input_size
        self.output_size = output_size
        self.normalization_stats = None

        self.finite_fns = finite_fns
        if self.finite_fns:
            assert num_fns is not None and init_seed is not None
            self.init_seed = init_seed
            self.num_fns = num_fns
            self.seeds = iter(jax.random.split(init_seed, num_fns))

    def sample_function_vals(self, x: jnp.ndarray, num_samples: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        raise NotImplementedError

    def sample_function(self, x: jnp.ndarray, num_samples: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        pass

    def sample_measurement_pts(self, num_points: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        pass

    def get_rng_key(self) -> jax.random.PRNGKey:
        try:
            rng_key = next(self.seeds)
        except StopIteration:
            self.seeds = iter(jax.random.split(self.init_seed, self.num_fns))
            rng_key = next(self.seeds)
        return rng_key


class GaussianProcessSim(FunctionSimulator):

    def __init__(self, num_pts: int, input_size: int = 1, output_scale: float = 1.0, length_scale: float = 1.0,
                 minval=-5., maxval=5.,
                 mean_fn: Optional[str] = 'sin', eps_k: float = 1e-5, rng_key: jax.random.PRNGKey = 1):
        """ Samples functions from a Gaussian Process (GP) with SE kernel
        Args:
            input_size: dimensionality of the inputs
            output_scale: output_scale of the SE kernel (coincides with the std of the GP prior)
            length_scale: lengthscale of the SE kernel
            mean_fn (optional): mean function of the GP. If None, uses a zero mean.
            eps_k: scale of the identity matrix to be added to the kernel matrix for numerical stability
        """
        super().__init__(input_size=input_size, output_size=1, finite_fns=False, num_fns=None, init_seed=None)

        if mean_fn == 'sin':
            self.mean_fn = lambda x: 2 * x[..., 0] + 5 * jnp.sin(2 * x[..., 0])
        else:
            self.mean_fn = lambda x: jnp.zeros((x.shape[0],))

        self.output_scale = output_scale
        self.kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
            length_scale=1.)
        self.eps_k = eps_k
        self.mins, self.maxs = minval, maxval
        self.num_pts = num_pts

    @partial(jax.jit, static_argnums=(0, 2))
    def samples_and_grad(self, x: jnp.ndarray, n_evals: int, key: jax.random.PRNGKey, eps: float = 1e-3):
        k = self.kernel.matrix(x, x) + eps * jnp.eye(x.shape[0])
        dist = tfd.MultivariateNormalFullCovariance(self.mean_fn(x), k)
        samples = dist.sample(seed=key, sample_shape=(n_evals,))
        grads = jax.vmap(jax.jacrev(lambda s: dist.log_prob(s)))(samples)
        return samples[..., None], grads

    @partial(jax.jit, static_argnums=(0, 1, 3))
    def sample_x_fx_w_score(self, num_samples: int, key: jax.random.PRNGKey = None, n_evals: int = 20):
        key, subkey = jax.random.split(key)
        x = jax.random.uniform(key=key, minval=self.mins, maxval=self.maxs, shape=(num_samples, self.num_pts, 1))
        samples, grads = jax.vmap(lambda x_: self.samples_and_grad(x_, n_evals, subkey), 0, 1)(x)
        grads = jnp.vstack(grads)
        x_fx = jnp.vstack(jax.vmap(lambda s: jnp.concatenate((x, s), axis=2))(samples))
        return x_fx, grads

    def get_data(self, num_train_pts, num_test_pts, rng_key: jax.random.PRNGKey = None, eps: float = 1e-4):
        key, subkey = jax.random.split(rng_key)
        x_train = jax.random.uniform(key=key, minval=self.mins, maxval=self.maxs, shape=(num_train_pts, 1))
        x_test = jax.random.uniform(key=key, minval=self.mins, maxval=self.maxs, shape=(num_test_pts, 1))

        k = self.kernel.matrix(x_train, x_train) + eps * jnp.eye(x_train.shape[0])
        dist = tfd.MultivariateNormalFullCovariance(self.mean_fn(x_train), k)
        y_train = dist.sample(seed=key)

        k = self.kernel.matrix(x_test, x_test) + eps * jnp.eye(x_test.shape[0])
        dist = tfd.MultivariateNormalFullCovariance(self.mean_fn(x_test), k)
        y_test = dist.sample(seed=key)

        return x_train, y_train, x_test, y_test


class GaussianProcessFitted(FunctionSimulator):

    def __init__(self, num_pts: int, input_size: int = 1, output_scale: float = 1.0, length_scale: float = 1.0,
                 minval=-5., maxval=5.,
                 mean_fn: Optional[str] = 'sin', eps_k: float = 1e-5, rng_key: jax.random.PRNGKey = 1):
        """ Samples functions from a Gaussian Process (GP) with SE kernel
        Args:
            input_size: dimensionality of the inputs
            output_scale: output_scale of the SE kernel (coincides with the std of the GP prior)
            length_scale: lengthscale of the SE kernel
            mean_fn (optional): mean function of the GP. If None, uses a zero mean.
            eps_k: scale of the identity matrix to be added to the kernel matrix for numerical stability
        """
        super().__init__(input_size=input_size, output_size=1, finite_fns=False, num_fns=None, init_seed=None)

        if mean_fn == 'sin':
            self.mean_fn = lambda x: 2 * x[..., 0] + 5 * jnp.sin(2 * x[..., 0])
        else:
            self.mean_fn = lambda x: jnp.zeros((x.shape[0],))

        self.output_scale = output_scale
        self.kernel = RBF(length_scale=length_scale)
        self.eps_k = eps_k
        self.mins, self.maxs = minval, maxval
        self.num_pts = num_pts

        x = np.array([-3.5, 0.5]).reshape(-1, 1).astype(np.float32)
        y = self.mean_fn(x)
        self.gpr = GaussianProcessRegressor(kernel=self.kernel, random_state=np.random.RandomState(rng_key)).fit(x, y)

    def sample_x_fx_w_score(self, num_samples, key: jax.random.PRNGKey = None):
        xs, ys, grads = [], [], []
        for i in range(num_samples):
            key, subkey = jax.random.split(key)
            x = np.random.uniform(self.mins, self.maxs, self.num_pts).reshape(-1, 1).astype(np.float32)
            y = self.gpr.sample_y(x, n_samples=1, random_state=np.random.RandomState(subkey))  # (20,10)

            grad = []
            for x_, y_ in zip(x, y):
                temp_mu_cov = self.gpr.predict(x_.reshape(-1, 1), return_std=True, return_cov=False)
                true_grad = jax.jacrev(
                    lambda yy: tfd.Normal(loc=temp_mu_cov[0], scale=temp_mu_cov[1]).log_prob(yy))(y_)
                grad.append(true_grad[..., 0])
            xs.append(x)
            ys.append(y)
            grads.append(grad)
        x_fx = jnp.concatenate((jnp.array(xs), jnp.array(ys)), axis=2)
        true_grads = jnp.array(grads).squeeze()
        return x_fx, true_grads


class StudentTProcessSim(FunctionSimulator):

    def __init__(self, num_pts: int, input_size: int = 1, output_scale: float = 1.0, length_scale: float = 1.0,
                 minval=-1., maxval=1.,
                 mean_fn: Optional[str] = 'sin', eps_k: float = 1e-5, df: float = 3., num_ind_pts: int = 10,
                 rng_key: jax.random.PRNGKey = None):
        """ Samples functions from a Gaussian Process (GP) with SE kernel
        Args:
            input_size: dimensionality of the inputs
            output_scale: output_scale of the SE kernel (coincides with the std of the GP prior)
            length_scale: lengthscale of the SE kernel
            mean_fn (optional): mean function of the GP. If None, uses a zero mean.
            eps_k: scale of the identity matrix to be added to the kernel matrix for numerical stability
        """
        super().__init__(input_size=input_size, output_size=1, finite_fns=False, num_fns=None, init_seed=None)

        if mean_fn == 'sin':
            self.mean_fn = lambda x: 2 * x[..., 0] + 5 * jnp.sin(2 * x[..., 0])
        else:
            self.mean_fn = lambda x: jnp.zeros((x.shape[0],))

        self.output_scale = output_scale
        self.kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=length_scale)
        self.eps_k = eps_k
        self.mins, self.maxs = minval, maxval
        self.num_pts = num_pts

        self.tp = tfd.StudentTProcess(df, self.kernel, mean_fn=self.mean_fn, observation_noise_variance=1e-7)

    # @partial(jax.jit, static_argnums=(0, 3))
    def samples_and_grad(self, x: jnp.ndarray, key: jax.random.PRNGKey, n_evals: int = 10):
        distr_x = self.tp.get_marginal_distribution(x)
        samples = distr_x.sample(n_evals, seed=key)
        true_grads = jax.vmap(jax.jacrev(distr_x.log_prob))(samples)
        return samples[..., None], true_grads

    # @partial(jax.jit, static_argnums=(0, 1))
    def sample_x_fx_w_score(self, num_samples, rng_key: jax.random.PRNGKey = None):
        key, subkey = jax.random.split(rng_key)
        x = jax.random.uniform(key=key, shape=(num_samples, self.num_pts, 1), minval=self.mins, maxval=self.maxs)
        samples, grads = jax.vmap(lambda x_: self.samples_and_grad(x_, subkey), 0, 1)(x)
        grads = jnp.vstack(grads)
        x_fx = jnp.vstack(jax.vmap(lambda s: jnp.concatenate((x, s), axis=2))(samples))
        return x_fx, grads


class StudentTProcessFitted(FunctionSimulator):

    def __init__(self, num_pts: int, input_size: int = 1, output_scale: float = 1.0, length_scale: float = 1.0,
                 minval=-1., maxval=1.,
                 mean_fn: Optional[str] = 'sin', eps_k: float = 1e-5, df: float = 3., num_ind_pts: int = 10):
        """ Samples functions from a Gaussian Process (GP) with SE kernel
        Args:
            input_size: dimensionality of the inputs
            output_scale: output_scale of the SE kernel (coincides with the std of the GP prior)
            length_scale: lengthscale of the SE kernel
            mean_fn (optional): mean function of the GP. If None, uses a zero mean.
            eps_k: scale of the identity matrix to be added to the kernel matrix for numerical stability
        """
        super().__init__(input_size=input_size, output_size=1, finite_fns=False, num_fns=None, init_seed=None)

        if mean_fn == 'sin':
            self.mean_fn = lambda x: 2 * x[..., 0] + 5 * np.sin(2 * x[..., 0])
        else:
            self.mean_fn = lambda x: np.zeros((x.shape[0],))

        self.output_scale = output_scale
        self.kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=length_scale)
        self.eps_k = eps_k
        self.mins, self.maxs = minval, maxval
        self.num_pts = num_pts

        index_points = np.expand_dims(np.linspace(minval, maxval, num_ind_pts), -1).astype(np.float32)
        self.tp = tfd.StudentTProcess(df, self.kernel, index_points, mean_fn=self.mean_fn)

    def sample_x_fx_w_score(self, num_samples, rng_key: jax.random.PRNGKey = None):
        xs, ys, grads = [], [], []
        for i in range(num_samples):
            x = np.random.uniform(self.mins, self.maxs, self.num_pts).reshape(-1, 1).astype(np.float32)
            distr_x = self.tp.get_marginal_distribution(x)
            y = distr_x.sample(1, seed=rng_key).T
            fn = lambda f_: distr_x.log_prob(f_)
            true_grads = jax.vmap(jax.jacrev(fn))(y)
            xs.append(x)
            ys.append(y)
            grads.append(true_grads)
        x_fx = jnp.concatenate((jnp.array(xs), jnp.array(ys)), axis=2)
        true_grads = jnp.array(grads).squeeze()
        return x_fx, true_grads


class SinusoidsSim(FunctionSimulator):

    def __init__(self, finite_fns: bool = False, num_fns: int = None, minval=-5., maxval=5.,
                 init_seed: jax.random.PRNGKey = jax.random.PRNGKey(1234)):
        super().__init__(input_size=1, output_size=1, finite_fns=finite_fns, num_fns=num_fns, init_seed=init_seed)
        self.mins = minval
        self.maxs = maxval
        self.input_size = 1
        self.output_size = 1

    def sample_function_vals(self, x: jnp.ndarray, num_samples: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        rng_key = self.get_rng_key() if self.finite_fns else rng_key

        assert x.ndim == 2 and x.shape[-1] == self.input_size
        key1, key2, key3 = jax.random.split(rng_key, 3)
        freq = jax.random.uniform(key1, shape=(num_samples,), minval=1.7, maxval=2.3)
        amp = 2 + 0.4 * jax.random.normal(key2, shape=(num_samples,))
        slope = 2 + 0.3 * jax.random.normal(key2, shape=(num_samples,))
        f = amp[:, None, None] * jnp.sin(freq[:, None, None] * x) + slope[:, None, None] * x
        assert f.shape == (num_samples, x.shape[0], self.output_size)
        return f

class NNDropoutMetaDataset(FunctionSimulator):
    # need: sample meas points, sample_x_fx and sample_fn_and_score
    def __init__(self, init_seed, dataset, num_pts=16, start_nn_lr=1e-3,
                 start_nn_wd=0.,
                 start_nn_batch_size=16,
                 dropout=0.1,
                 start_nn_num_epochs=100, data_dir=None):
        super().__init__(output_size=1, init_seed=init_seed)
        # Getting data
        self.meta_train_data, _, self.meta_test_data = provide_data(dataset=dataset, seed=init_seed, data_dir=data_dir)
        self.num_pts = num_pts

        # Fitting GPs
        if 'physionet' in dataset:
            self.nns, self.normalization_stats, (self.mins, self.maxs) = get_nns_dropout_sklearn_differentdim(
                self.meta_train_data, init_seed, start_nn_lr=start_nn_lr, dropout=dropout,
                start_nn_wd=start_nn_wd,
                start_nn_batch_size=start_nn_batch_size,
                start_nn_num_epochs=start_nn_num_epochs)
            self.input_size = 1
        else:
            self.nns, self.normalization_stats, (self.mins, self.maxs) \
                = get_nns_dropout_sklearn_multidim(self.meta_train_data, init_seed, start_nn_lr=start_nn_lr, dropout=dropout,
                                                   start_nn_wd=start_nn_wd,
                                                   start_nn_batch_size=start_nn_batch_size,
                                                   start_nn_num_epochs=start_nn_num_epochs)
            self.input_size = self.meta_train_data[0][0].shape[-1]

        # Normalization stats
        self.x_mean, self.x_std, self.y_mean, self.y_std = self.normalization_stats.values()

        # Auxiliary variables

        self.num_f_train = len(self.meta_train_data)
        self.num_f_test = len(self.meta_test_data)

    def sample_measurement_pts(self, num_points: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        x = jax.random.uniform(rng_key, shape=(num_points, self.input_size), minval=self.mins, maxval=self.maxs)
        return x  # (num_points, self.input_size)

    def sample_x_fx(self, num_samples: int = 1,
                    rng_key: jax.random.PRNGKey = None) -> jnp.ndarray:
        key, subkey = jax.random.split(rng_key)

        xs, fxs = [], []
        for i in range(num_samples):
            key, subkey = jax.random.split(key)
            x = self.sample_measurement_pts(self.num_pts, key)
            x_normalized = self.normalize_x(x)  # (num_pts, input_size)
            assert x.ndim == 2 and x.shape[-1] == self.input_size
            xs.append(x_normalized)

            choice = int(jax.random.randint(subkey, shape=(1,), minval=0, maxval=len(self.nns)))
            f = self.nns[choice](x_normalized, subkey)
            fxs.append(f)

        xs = jnp.array(xs)  # (num_samples, num_pts, 1)
        fxs = jnp.array(fxs)  # (num_samples, num_pts, 1)

        assert xs.shape[:2] == fxs.shape[:2] == (num_samples, self.num_pts)

        x_fx = jnp.concatenate((xs, fxs), axis=-1)  # (num_samples, self.num_pts, x_dim+y_dim)
        assert x_fx.ndim == 3

        return x_fx

    def normalize_x(self, x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
        x = (x - self.x_mean[None, :]) / (self.x_std[None, :] + eps)

        return x

    def normalize_y(self, y: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
        y = (y - self.y_mean[None, :]) / (self.y_std[None, :] + eps)
        assert y.shape == y.shape
        return y

class GPMetaDataset(FunctionSimulator):
    def __init__(self, dataset, init_seed=hk.PRNGSequence(2), num_input_pts=16, data_dir=None,
                 buffer=0.2, n_samples=None, ):
        super().__init__(output_size=1, init_seed=init_seed)
        from MARS.modules.data_modules.regression_datasets import provide_data
        # Getting data
        self.meta_train_data, _, self.meta_test_data = provide_data(dataset=dataset, seed=init_seed, data_dir=data_dir,
                                                                    n_samples=n_samples)
        self.num_pts = num_input_pts

        # Fitting GPs
        self.gps, self.normalization_stats, (mins, maxs) = get_gps(
            self.meta_train_data)
        self.input_size = self.normalization_stats['x_mean'].shape[-1]

        # Normalization stats
        self.x_mean, self.x_std, self.y_mean, self.y_std = self.normalization_stats.values()

        # Auxiliary variables

        self.num_f_train = len(self.meta_train_data)
        self.num_f_test = len(self.meta_test_data)

        if buffer is None:
            self.mins, self.maxs = mins, maxs
        else:
            self.mins = mins - buffer * jnp.abs(mins)
            self.maxs = maxs - buffer * jnp.abs(maxs)

    def sample_measurement_pts(self, num_points: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        x = jax.random.uniform(rng_key, shape=(num_points, self.input_size), minval=self.mins, maxval=self.maxs)
        return x  # (num_points, self.input_size)

    def sample_x_fx(self, num_samples: int = 1,
                    rng_key: jax.random.PRNGKey = None) -> jnp.ndarray:
        key, subkey = jax.random.split(rng_key)

        xs, fxs = [], []
        for i in range(num_samples):
            key, subkey = jax.random.split(key)
            x = self.sample_measurement_pts(self.num_pts, key)
            x_normalized = self.normalize_x(x)  # (num_pts, input_size)
            assert x.ndim == 2 and x.shape[-1] == self.input_size
            xs.append(x_normalized)

            choice = int(jax.random.randint(subkey, shape=(1,), minval=0, maxval=len(self.gps)))
            f = self.gps[choice].sample_y(x_normalized, n_samples=1)
            fxs.append(f)

        xs = jnp.array(xs)  # (num_samples, num_pts, 1)
        fxs = jnp.array(fxs)  # (num_samples, num_pts, 1)

        assert xs.shape[:2] == fxs.shape[:2] == (num_samples, self.num_pts)

        x_fx = jnp.concatenate((xs, fxs), axis=-1)  # (num_samples, self.num_pts, x_dim+y_dim)
        assert x_fx.ndim == 3

        return x_fx

    def sample_x_fx_mean(self, num_samples: int = 1,
                         rng_key: jax.random.PRNGKey = None) -> jnp.ndarray:
        key, subkey = jax.random.split(rng_key)

        xs, fxs = [], []
        for i in range(num_samples):
            key, subkey = jax.random.split(key)
            x = self.sample_measurement_pts(self.num_pts, key)
            x_normalized = self.normalize_x(x)  # (num_pts, input_size)
            assert x.ndim == 2 and x.shape[-1] == self.input_size
            xs.append(x_normalized)

            choice = int(jax.random.randint(subkey, shape=(1,), minval=0, maxval=len(self.gps)))
            f = self.gps[choice].predict(x_normalized)
            fxs.append(f)

        xs = jnp.array(xs)  # (num_samples, num_pts, 1)
        fxs = jnp.array(fxs)  # (num_samples, num_pts, 1)

        assert xs.shape[:2] == fxs.shape[:2] == (num_samples, self.num_pts)

        x_fx = jnp.concatenate((xs, fxs[..., None]), axis=-1)  # (num_samples, self.num_pts, x_dim+y_dim)
        assert x_fx.ndim == 3

        return x_fx

    def sample_fn_and_score(self, rng_key: jax.random.PRNGKey) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x = self.sample_measurement_pts(self.num_pts, rng_key)  # (1, dim_x)
        assert x.ndim == 2 and x.shape[-1] == self.input_size
        x_normalized = self.normalize_x(x)

        f, mu, cov = [], [], []
        for gp in self.gps:
            f.append(jnp.array(gp.sample_y(x_normalized, n_samples=1)))
            mu_, cov_ = gp.predict(x_normalized, return_std=False, return_cov=True)
            mu.append(mu_), cov.append(cov_)

        f = jnp.array(f)  # (len(self.gps), self.num_pts, 1)
        mu = jnp.array(mu)  # (len(self.gps), self.num_pts)
        cov = jnp.array(cov)  # (len(self.gps), self.num_pts, self.num_pts)

        x_fx = jax.vmap(lambda f_: jnp.hstack((x_normalized, f_)))(f)  # (len(self.gps), 1, x_dim+y_dim)
        return x_fx, mu, cov

    def normalize_x(self, x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
        x_normalized = (x - self.x_mean[None, ...]) / (self.x_std[None, ...] + eps)
        assert x_normalized.shape == x.shape
        return x_normalized

    def normalize_y(self, y: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
        y_normalized = (y - self.y_mean[None, ...]) / (self.y_std[None, ...] + eps)
        assert y_normalized.shape == y.shape
        return y_normalized


class GPMetaDatasetExample(FunctionSimulator):
    def __init__(self, dataset, init_seed=hk.PRNGSequence(2), n_train_pts=1, n_meas_pts=2, data_dir=None,
                 buffer=0.2, ):
        super().__init__(output_size=1, init_seed=init_seed)
        from MARS.modules.data_modules.regression_datasets import provide_data, GPSinMetaDataset
        # Getting data
        self.dataset = GPSinMetaDataset(random_state=np.random.RandomState(init_seed),
                                        mean=lambda x: 2.5 * x + 7.5 * np.sin(1.25 * x))
        self.meta_train_data, _, self.meta_test_data = provide_data(dataset=dataset, seed=init_seed, data_dir=data_dir,
                                                                    n_samples=n_train_pts)

        import matplotlib.pyplot as plt
        for x, y in self.meta_train_data[:5]:
            plt.suptitle('Samples from the underlying stochastic process')
            plt.scatter(x, y)
        plt.show()

        self.num_pts = n_train_pts + n_meas_pts

        # Fitting GPs
        self.gps, self.normalization_stats, (mins, maxs) = get_gps(self.meta_train_data, print_kernel=False,
                                                                   plot_gps=True)
        self.input_size = self.normalization_stats['x_mean'].shape[-1]

        # Normalization stats
        self.x_mean, self.x_std, self.y_mean, self.y_std = self.normalization_stats.values()

        # Auxiliary variables

        self.num_f_train = len(self.meta_train_data)
        self.num_f_test = len(self.meta_test_data)

        if buffer is None:
            self.mins, self.maxs = mins, maxs
        else:
            self.mins = mins - buffer * jnp.abs(mins)
            self.maxs = maxs - buffer * jnp.abs(maxs)

        x = np.array([-3.75, 3.75])[..., None]
        y = 7.5 * np.sin(1.25 * x) + 2.5 * x
        _, _, x_test, y_test = self.meta_test_data[0]

        self.example_data = (x, y, x_test, y_test)

    def sample_measurement_pts(self, num_points: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        x = jax.random.uniform(rng_key, shape=(num_points, self.input_size), minval=self.mins, maxval=self.maxs)
        return x  # (num_points, self.input_size)

    def sample_x_fx(self, num_samples: int = 1,
                    rng_key: jax.random.PRNGKey = None) -> jnp.ndarray:
        key, subkey = jax.random.split(rng_key)

        xs, fxs = [], []
        for i in range(num_samples):
            key, subkey = jax.random.split(key)
            x = self.sample_measurement_pts(self.num_pts, key)
            x_normalized = self.normalize_x(x)  # (num_pts, input_size)
            assert x.ndim == 2 and x.shape[-1] == self.input_size
            xs.append(x_normalized)

            choice = int(jax.random.randint(subkey, shape=(1,), minval=0, maxval=len(self.gps)))
            f = self.gps[choice].sample_y(x_normalized, n_samples=1)
            fxs.append(f)

        xs = jnp.array(xs)  # (num_samples, num_pts, 1)
        fxs = jnp.array(fxs)  # (num_samples, num_pts, 1)

        assert xs.shape[:2] == fxs.shape[:2] == (num_samples, self.num_pts)

        x_fx = jnp.concatenate((xs, fxs), axis=-1)  # (num_samples, self.num_pts, x_dim+y_dim)
        assert x_fx.ndim == 3

        return x_fx

    def sample_x_fx_mean(self, num_samples: int = 1,
                         rng_key: jax.random.PRNGKey = None) -> jnp.ndarray:
        key, subkey = jax.random.split(rng_key)

        xs, fxs = [], []
        for i in range(num_samples):
            key, subkey = jax.random.split(key)
            x = self.sample_measurement_pts(self.num_pts, key)
            x_normalized = self.normalize_x(x)  # (num_pts, input_size)
            assert x.ndim == 2 and x.shape[-1] == self.input_size
            xs.append(x_normalized)

            choice = int(jax.random.randint(subkey, shape=(1,), minval=0, maxval=len(self.gps)))
            f = self.gps[choice].predict(x_normalized)
            fxs.append(f)

        xs = jnp.array(xs)  # (num_samples, num_pts, 1)
        fxs = jnp.array(fxs)  # (num_samples, num_pts, 1)

        assert xs.shape[:2] == fxs.shape[:2] == (num_samples, self.num_pts)

        x_fx = jnp.concatenate((xs, fxs[..., None]), axis=-1)  # (num_samples, self.num_pts, x_dim+y_dim)
        assert x_fx.ndim == 3

        return x_fx

    def sample_fn_and_score(self, rng_key: jax.random.PRNGKey) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x = self.sample_measurement_pts(self.num_pts, rng_key)  # (1, dim_x)
        assert x.ndim == 2 and x.shape[-1] == self.input_size
        x_normalized = self.normalize_x(x)

        f, mu, cov = [], [], []
        for gp in self.gps:
            f.append(jnp.array(gp.sample_y(x_normalized, n_samples=1)))
            mu_, cov_ = gp.predict(x_normalized, return_std=False, return_cov=True)
            mu.append(mu_), cov.append(cov_)

        f = jnp.array(f)  # (len(self.gps), self.num_pts, 1)
        mu = jnp.array(mu)  # (len(self.gps), self.num_pts)
        cov = jnp.array(cov)  # (len(self.gps), self.num_pts, self.num_pts)

        x_fx = jax.vmap(lambda f_: jnp.hstack((x_normalized, f_)))(f)  # (len(self.gps), 1, x_dim+y_dim)
        return x_fx, mu, cov

    def normalize_x(self, x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
        x_normalized = (x - self.x_mean[None, ...]) / (self.x_std[None, ...] + eps)
        assert x_normalized.shape == x.shape
        return x_normalized

    def unnormalize_x(self, x: jnp.ndarray, eps: float = 1e-8):
        if x.ndim == 1:
            x = jnp.expand_dims(x, -1)

        x_unnorm = x * (self.x_std[None, :] + eps) + self.x_mean[None, :]
        assert x_unnorm.shape == x.shape
        return x_unnorm

    def normalize_y(self, y: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
        y_normalized = (y - self.y_mean[None, ...]) / (self.y_std[None, ...] + eps)
        assert y_normalized.shape == y.shape
        return y_normalized


""" Swissfel Dataset"""


class SwissfelMetaDataset(FunctionSimulator):
    runs_12dim = [
        {'experiment': '2018_10_31/line_ucb_ascent', 'run': 0},
        {'experiment': '2018_10_31/line_ucb_ascent', 'run': 1},
        {'experiment': '2018_10_31/line_ucb_ascent', 'run': 2},
        {'experiment': '2018_10_31/line_ucb', 'run': 0},
        {'experiment': '2018_10_31/line_ucb', 'run': 1},
        {'experiment': '2018_10_31/line_ucb', 'run': 2},
        {'experiment': '2018_10_31/neldermead', 'run': 0},
        {'experiment': '2018_10_31/neldermead', 'run': 1},
        {'experiment': '2018_10_31/neldermead', 'run': 2},
    ]

    runs_24dim = [
        {'experiment': '2018_11_01/line_ucb_ascent_bpm_24', 'run': 0},
        {'experiment': '2018_11_01/line_ucb_ascent_bpm_24', 'run': 1},
        {'experiment': '2018_11_01/line_ucb_ascent_bpm_24', 'run': 3},
        {'experiment': '2018_11_01/line_ucb_ascent_bpm_24_small', 'run': 0},
        {'experiment': '2018_11_01/lipschitz_line_ucb_bpm_24', 'run': 0},
        {'experiment': '2018_11_01/neldermead_bpm_24', 'run': 0},
        {'experiment': '2018_11_01/neldermead_bpm_24', 'run': 1},
        {'experiment': '2018_11_01/parameter_scan_bpm_24', 'run': 0},
    ]

    def __init__(self, random_state=None, param_space_id=0, swissfel_dir=None):
        super().__init__(random_state)

        self.swissfel_dir = SWISSFEL_DIR if swissfel_dir is None else swissfel_dir

        if not os.path.isdir(self.swissfel_dir):
            print("Swissfel data does not exist in %s" % self.swissfel_dir)
            download_and_unzip_data(SWISSFEL_URL, self.swissfel_dir)

        if param_space_id == 0:
            run_specs = copy.deepcopy(self.runs_12dim)
        elif param_space_id == 1:
            run_specs = copy.deepcopy(self.runs_24dim)
        else:
            raise NotImplementedError

        self.random_state.shuffle(run_specs)
        self.run_specs_train = run_specs[:5]
        self.run_specs_test = run_specs[5:]

    def _load_data(self, experiment, run=0):
        path = os.path.join(self.swissfel_dir, experiment)

        # read hdf5
        hdf5_path = os.path.join(path, 'data/evaluations.hdf5')
        dset = h5py.File(hdf5_path, 'r')
        run = str(run)
        data = dset['1'][run][()]
        dset.close()

        # read config and recover parameter names

        config_path = os.path.join(path, 'experiment.yaml')
        config_file = open(config_path, 'r')  # 'document.yaml' contains a single YAML document.

        # get config files for parameters
        files = yaml.load(config_file)['swissfel.interface']['channel_config_set']
        if not isinstance(files, list):
            files = [files]

        files += ['channel_config_set.txt']  # backwards compatibility

        parameters = []
        for file in files:
            params_path = os.path.join(path, 'sf', os.path.split(file)[1])
            if not os.path.exists(params_path):
                continue

            frame = pd.read_csv(params_path, comment='#')

            parameters += frame['pv'].tolist()

        return data, parameters

    def _load_meta_dataset(self, train=True):
        run_specs = self.run_specs_train if train else self.run_specs_test
        data_tuples = []
        for run_spec in run_specs:
            data, parameters = self._load_data(**run_spec)
            data_tuples.append((data['x'], data['y']))

        assert len(set([X.shape[-1] for X, _ in data_tuples])) == 1
        assert all([X.shape[0] == Y.shape[0] for X, Y in data_tuples])
        return data_tuples

    def generate_meta_train_data(self, n_tasks=5, n_samples=200):
        assert n_tasks == len(self.run_specs_train), "number of tasks must be %i" % len(self.run_specs_train)
        meta_train_tuples = self._load_meta_dataset(train=True)

        max_n_samples = max([X.shape[0] for X, _ in meta_train_tuples])
        assert n_samples <= max_n_samples, 'only %i number of samples available' % max_n_samples

        meta_train_tuples = [(X[:n_samples], Y[:n_samples]) for X, Y in meta_train_tuples]

        return meta_train_tuples

    def generate_meta_test_data(self, n_tasks=None, n_samples_context=200, n_samples_test=400):
        if n_tasks is None:
            n_tasks = len(self.run_specs_test)

        assert n_tasks == len(self.run_specs_test), "number of tasks must be %i" % len(self.run_specs_test)
        meta_test_tuples = self._load_meta_dataset(train=False)

        max_n_samples = min([X.shape[0] for X, _ in meta_test_tuples])
        assert n_samples_context + n_samples_test <= max_n_samples, 'only %i number of samples available' % max_n_samples

        idx = np.arange(n_samples_context + n_samples_test)
        self.random_state.shuffle(idx)
        idx_context, idx_test = idx[:n_samples_context], idx[n_samples_context:]

        meta_test_tuples = [(X[idx_context], Y[idx_context], X[idx_test], Y[idx_test]) for X, Y in meta_test_tuples]

        return meta_test_tuples


def download_and_unzip_data(url, target_dir):
    from urllib.request import urlopen
    from zipfile import ZipFile
    print('Downloading %s' % url)
    tempfilepath = os.path.join(DATA_DIR, 'tempfile.zip')
    zipresp = urlopen(url)
    with open(tempfilepath, 'wb') as f:
        f.write(zipresp.read())
    zf = ZipFile(tempfilepath)
    print('Extracting to %s' % target_dir)
    zf.extractall(path=target_dir)
    zf.close()
    os.remove(tempfilepath)


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    key_out = jax.random.PRNGKey(984)
    sim = SinusoidsSim(finite_fns=False)
    x_plot = jnp.linspace(-5, 5, 200).reshape((-1, 1))

    y_samples = sim.sample_function_vals(x_plot, 10, key_out)
    for y_ in y_samples:
        plt.plot(x_plot, y_)
    plt.show()
