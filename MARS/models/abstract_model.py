import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_datasets as tfds
import time
import matplotlib.pyplot as plt
import numpy as np
import logging

from MARS.modules.utils import RngKeyMixin, aggregate_stats
from MARS.modules.fsvgd_modules.distribution import AffineTransform
from MARS.modules import BatchedMLP
from typing import Optional, Tuple, Callable, Dict, List, Union
from tensorflow_probability.substrates import jax as tfp


class AbstactRegressionModel(RngKeyMixin):

    def __init__(self, input_size: int, output_size: int, rng_key: jax.random.PRNGKey, normalize_data: bool = True,
                 normalization_stats: Optional[Dict[str, jnp.ndarray]] = None):
        super().__init__(rng_key)

        self.input_size = input_size
        self.output_size = output_size
        self.normalize_data = normalize_data
        self.need_to_compute_norm_stats = normalize_data and (normalization_stats is None)

        self.data = None

        if normalization_stats is None:
            # initialize normalization stats to neutral elements
            self._x_mean, self._x_std = jnp.zeros(input_size), jnp.ones(input_size)
            self._y_mean, self._y_std = jnp.zeros(output_size), jnp.ones(output_size)
        else:
            # check the provided normalization stats
            _n_stats = normalization_stats
            assert _n_stats['x_mean'].shape == _n_stats['x_std'].shape == (self.input_size,)
            assert _n_stats['y_mean'].shape == _n_stats['y_std'].shape == (self.output_size,)
            assert jnp.all(_n_stats['y_std'] > 0) and jnp.all(_n_stats['x_std'] > 0), 'stds need to be positive'
            # set the stats
            self._x_mean, self._x_std = _n_stats['x_mean'], _n_stats['x_std']
            self._y_mean, self._y_std = _n_stats['y_mean'], _n_stats['y_std']
        self.affine_transform_y = lambda dist: AffineTransform(shift=self._y_mean, scale=self._y_std)

        # disable some stupid tensorflow probability warnings
        self._add_checktypes_logging_filter()

    def _compute_normalization_stats(self, x: jnp.ndarray, y: jnp.ndarray) -> None:
        """ Computes the empirical normalization stats and stores as private variables. """
        x, y = self._ensure_atleast_2d_float(x, y)
        self._x_mean = jnp.mean(x, axis=0)
        self._y_mean = jnp.mean(y, axis=0)
        self._x_std = jnp.std(x, axis=0)
        self._y_std = jnp.std(y, axis=0)

    def _normalize_data(self, x: jnp.ndarray, y: Optional[jnp.ndarray] = None, eps: float = 1e-8):
        """ Normalizes the data given the normalization statistics. """
        if y is None:
            x = self._ensure_atleast_2d_float(x)
        else:
            x, y = self._ensure_atleast_2d_float(x, y)
        x_normalized = (x - self._x_mean[None, :]) / (self._x_std[None, :] + eps)
        assert x_normalized.shape == x.shape
        if y is None:
            return x_normalized
        else:
            y_normalized = self._normalize_y(y, eps=eps)
            assert y_normalized.shape == y.shape
            return x_normalized, y_normalized

    def _unnormalize_data(self, x: jnp.ndarray, y: Optional[jnp.ndarray] = None, eps: float = 1e-8):
        """ Unnormalizes the data given the normalization statistics. """
        if y is None:
            x = self._ensure_atleast_2d_float(x)
        else:
            x, y = self._ensure_atleast_2d_float(x, y)
        x_unnorm = x * (self._x_std[None, :] + eps) + self._x_mean[None, :]
        assert x_unnorm.shape == x.shape
        if y is None:
            return x_unnorm
        else:
            y_unnorm = self._unnormalize_y(y, eps=eps)
            return x_unnorm, y_unnorm

    def _normalize_y(self, y: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
        """ Output normalization. """

        y = self._ensure_atleast_2d_float(y)
        assert y.shape[-1] == self.output_size
        y_normalized = (y - self._y_mean) / (self._y_std + eps)
        assert y_normalized.shape == y.shape
        return y_normalized

    def _unnormalize_y(self, y: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
        """ Output unnormalization. """
        y = self._ensure_atleast_2d_float(y)
        assert y.shape[-1] == self.output_size
        y_unnorm = y * (self._y_std + eps) + self._y_mean
        assert y_unnorm.shape == y.shape
        return y_unnorm

    @staticmethod
    def _ensure_atleast_2d_float(x: jnp.ndarray, y: Optional[jnp.ndarray] = None, dtype: jnp.dtype = jnp.float32):
        """ Data handling. """

        if x.ndim == 1:
            x = jnp.expand_dims(x, -1)
        if y is None:
            return jnp.array(x).astype(dtype=dtype)
        else:
            assert len(x) == len(y)
            if y.ndim == 1:
                y = jnp.expand_dims(y, -1)
            return jnp.array(x).astype(dtype=dtype), jnp.array(y).astype(dtype=dtype)

    def _to_pred_dist(self, y_pred_raw: jnp.ndarray, likelihood_std: jnp.ndarray, include_noise: bool = False):
        """ Forms the predictive distribution p(y|x, D) given the batched NNs raw outputs and the likelihood_std. """
        assert y_pred_raw.ndim == 3 and y_pred_raw.shape[-1] == self.output_size
        num_post_samples = y_pred_raw.shape[0]
        if include_noise:
            independent_normals = tfd.MultivariateNormalDiag(jnp.moveaxis(y_pred_raw, 0, 1), likelihood_std)
            mixture_distribution = tfd.Categorical(probs=jnp.ones(num_post_samples) / num_post_samples)
            pred_dist_raw = tfd.MixtureSameFamily(mixture_distribution, independent_normals)
        else:
            pred_dist_raw = tfd.MultivariateNormalDiag(jnp.mean(y_pred_raw, axis=0),
                                                       jnp.std(y_pred_raw, axis=0))
        pred_dist = self.affine_transform_y(pred_dist_raw)
        return pred_dist

    def _preprocess_train_data(self, x_train: jnp.ndarray, y_train: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """ Convert data to float32, ensure 2d shape and normalize if necessary. """
        if self.normalize_data:
            if self.need_to_compute_norm_stats:
                self._compute_normalization_stats(x_train, y_train)
            x_train, y_train = self._normalize_data(x_train, y_train)
            self.affine_transform_y = AffineTransform(shift=self._y_mean, scale=self._y_std)
        else:
            x_train, y_train = self._ensure_atleast_2d_float(x_train, y_train)
        return x_train, y_train

    def _create_data_loader(self, x_data: jnp.ndarray, y_data: jnp.ndarray, batch_size: int = 64, shuffle: bool = True,
                            infinite: bool = True) -> tf.data.Dataset:
        """ Creating a Tensorflow data loader. """
        ds = tf.data.Dataset.from_tensor_slices((x_data, y_data))
        if shuffle:
            seed = int(jax.random.randint(self.rng_key, (1,), 0, 10 ** 8))
            ds = ds.shuffle(batch_size * 4, seed=seed, reshuffle_each_iteration=True)
        if infinite:
            ds = ds.repeat()
        ds = ds.batch(batch_size)
        ds = tfds.as_numpy(ds)
        return ds

    @property
    def likelihood_std(self):
        return jax.nn.softplus(self._likelihood_std_raw)

    @likelihood_std.setter
    def likelihood_std(self, std: jnp.ndarray):
        self._likelihood_std_raw = tfp.math.softplus_inverse(std)

    def predict_dist(self, x: jnp.ndarray, include_noise: bool = False) -> tfp.distributions.Distribution:
        raise NotImplementedError

    def predict(self, x: jnp.ndarray, include_noise: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError

    def calib_error(self, y_true, pred_dist, use_circular_region=False):
        """ Function for calculating the calibration error. """

        mean = pred_dist.mean
        cov = pred_dist.covariance()[..., 0]  # fixme for multidim

        cdf_vals = jax.vmap(lambda mean_, cov_, y_true_: tfd.Normal(mean_, cov_).cdf(y_true_))(mean, cov, y_true)
        cdf_vals = jnp.reshape(cdf_vals, (-1, 1))

        num_points = jnp.array(y_true.shape).prod()
        conf_levels = jnp.linspace(0.025, 0.975, 20)
        emp_freq_per_conf_level = jnp.sum(cdf_vals <= conf_levels, axis=0) / num_points

        calib_err = jnp.mean(jnp.abs((emp_freq_per_conf_level - conf_levels)))
        return calib_err

    def eval(self, x: jnp.ndarray, y: np.ndarray, prefix: str = '') -> Dict[str, jnp.ndarray]:
        """ Evaluation function, returning nll, RMSE and calibration error. """
        x, y = self._ensure_atleast_2d_float(x, y)
        pred_dist = self.predict_dist(x, include_noise=True)
        nll = - jnp.mean(pred_dist.log_prob(y))
        rmse = jnp.sqrt(jnp.mean(jnp.sum((pred_dist.mean - y) ** 2, axis=-1)))

        calib_err = jnp.mean(self.calib_error(y, pred_dist)) if self.__class__.__name__ != 'BNN_SVGD' else 0.

        eval_stats = {'nll': nll, 'rmse': rmse, "calib": calib_err}
        # add prefix to stats names
        eval_stats = {f'{prefix}{name}': val for name, val in eval_stats.items()}
        return eval_stats

    def plot_1d(self, x_train: jnp.ndarray, y_train: jnp.ndarray, domain_l: Optional[jnp.ndarray] = None,
                domain_u: Optional[jnp.ndarray] = None, plot_data: tuple[jnp.ndarray, jnp.ndarray] = None,
                title: Optional[str] = '', show: bool = True):
        """ Plotting function for experiments with one-dimensional inputs and outputs. """
        assert self.output_size == self.input_size == 1
        x_train, y_train = jnp.array(sorted(zip(x_train, y_train)))[..., 0].split(2, 1)

        # determine plotting domain
        x_min, x_max = jnp.min(x_train, axis=0), jnp.max(x_train, axis=0)
        width = x_max - x_min
        if domain_l is None:
            domain_l = x_min - 0.3 * width
        if domain_u is None:
            domain_u = x_max + 0.3 * width
        x_plot = jnp.linspace(domain_l, domain_u, 200).reshape((-1, 1))

        # make predictions
        pred_mean, pred_std = map(lambda arr: arr.flatten(), self.predict(x_plot))

        # draw the plot
        fig, ax = plt.subplots()

        if plot_data is not None:
            x, y = plot_data
            x, y = zip(*sorted(zip(x, y)))
            ax.plot(x, y, label='true fun', linestyle='--', color='#4958AD')
        ax.plot(x_plot, pred_mean, label='pred mean', color='#42529C')
        ax.fill_between(x_plot.flatten(), pred_mean - pred_std, pred_mean + pred_std, alpha=0.15, color='blue',
                        label=r"$\pm$ 1 std. dev.", )

        if hasattr(self, 'predict_post_samples'):
            y_post_samples = self.predict_post_samples(x_plot)
            for y in y_post_samples:
                ax.plot(x_plot, y, linewidth=0.5, alpha=0.3, color='#3169A4')

        ax.plot(x_train, y_train, 'o', linewidth=2, label='train points', alpha=0.8, markeredgecolor='blue',
                color='white', markeredgewidth=0.5, markersize=8)

        if title is not None:
            ax.set_title(title)
        ax.legend(loc='upper left')

        if show:
            fig.show()

        return fig

    def _add_checktypes_logging_filter(self):
        logger = logging.getLogger()

        class CheckTypesFilter(logging.Filter):
            def filter(self, record):
                return "check_types" not in record.getMessage()

        logger.addFilter(CheckTypesFilter())

    def step(self, x_batch: jnp.ndarray, y_batch: jnp.ndarray, num_train_points: Union[float, int]) -> Dict[str, float]:
        pass


class BatchedNeuralNetworkModel(AbstactRegressionModel):

    def __init__(self,
                 *args,
                 data_batch_size: int = 16,
                 num_train_steps: int = 10000,
                 hidden_layer_sizes: List[int] = (32, 32, 32),
                 hidden_activation: Optional[Callable] = jax.nn.leaky_relu,
                 last_activation: Optional[Callable] = None,
                 num_batched_nns: int = 10,
                 initializer_b: Callable = jax.nn.initializers.constant(0.01),
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.data_batch_size = data_batch_size
        self.num_train_steps = num_train_steps
        self.num_batched_nns = num_batched_nns
        self.trained_nn = None
        self.params = None

        # setup batched mlp
        self.batched_model = BatchedMLP(input_size=self.input_size, output_size=self.output_size,
                                        num_batched_modules=num_batched_nns,
                                        hidden_layer_sizes=hidden_layer_sizes,
                                        hidden_activation=hidden_activation,
                                        last_activation=last_activation,
                                        rng_key=self.rng_key,
                                        initializer_b=initializer_b)

    def fit(self, x_train: jnp.ndarray, y_train: jnp.ndarray, x_eval: Optional[jnp.ndarray] = None,
            y_eval: Optional[jnp.ndarray] = None, num_steps: Optional[int] = None, log_period: int = 500):
        """ Function for fitting the BNN to the data. """
        # check whether eval data has been passed
        evaluate = x_eval is not None or y_eval is not None
        assert not evaluate or x_eval.shape[0] == y_eval.shape[0]
        # prepare data and data loader
        x_train, y_train = self._preprocess_train_data(x_train, y_train)
        train_loader = self._create_data_loader(x_train, y_train,
                                                batch_size=self.data_batch_size)
        num_train_points = x_train.shape[0]

        # initialize attributes to keep track of during training
        num_steps = self.num_train_steps if num_steps is None else num_steps
        samples_cum_period = 0.0
        stats_list, eval_list = [], []
        t_start_period = time.time()

        # training loop
        for step, (x_batch, y_batch) in enumerate(train_loader, 1):
            samples_cum_period += x_batch.shape[0]

            # perform the train step
            stats = self.step(x_batch, y_batch, num_train_points)
            stats_list.append(stats)

            if step % log_period == 0 or step == 1:
                duration_sec = time.time() - t_start_period
                duration_per_sample_ms = duration_sec / samples_cum_period * 1000
                stats_agg = aggregate_stats(stats_list)

                if evaluate:
                    eval_stats = self.eval(x_eval, y_eval, prefix='eval_')
                    stats_agg.update(eval_stats)
                    eval_list.append(eval_stats)

                stats_msg = ' | '.join([f'{n}: {v:.4f}' for n, v in stats_agg.items()])
                msg = (f'Step {step}/{num_steps} | {stats_msg} | Duration {duration_sec:.2f} sec | '
                       f'Time per sample {duration_per_sample_ms:.2f} ms')
                print(msg)

                # reset the attributes we keep track of
                stats_list = []
                samples_cum_period = 0
                t_start_period = time.time()

            if step >= num_steps:
                break
        return eval_list

    def _construct_nn_param_prior(self, weight_prior_std: float, bias_prior_std: float) -> tfd.MultivariateNormalDiag:
        prior_stds = []
        params = self.batched_model.params_stacked
        for layer_params in params:
            for param_name, param_array in layer_params.items():
                flat_shape = param_array[0].flatten().shape
                if param_name == 'w':
                    prior_stds.append(weight_prior_std * jnp.ones(flat_shape))
                elif param_name == 'b':
                    prior_stds.append(bias_prior_std * jnp.ones(flat_shape))
                else:
                    raise ValueError(f'Unknown parameter {param_name}')
        prior_stds = jnp.concatenate(prior_stds)
        prior_dist = tfd.MultivariateNormalDiag(jnp.zeros_like(prior_stds), prior_stds)
        assert prior_dist.event_shape == self.batched_model.param_vectors_stacked.shape[-1:]
        return prior_dist

    def predict(self, x: jnp.ndarray, include_noise: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """ Returns the mean and std of the predictive distribution.

        Args:
            x (jnp.ndarray): points in which to make predictions
            include_noise (bool): whether to include alleatoric uncertainty in the predictive std. If False,
                                  only return the epistemic std.

        Returns: predictive mean and std
        """
        pred_dist = self.predict_dist(x=x, include_noise=include_noise)
        return pred_dist.mean, pred_dist.stddev
