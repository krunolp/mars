import jax
import jax.numpy as jnp
import haiku as hk
import matplotlib.pyplot as plt
import gpjax as gpx
import optax as ox

from jax import jit


class GP:
    """ A minimal implementation of the GaussianProcessRegressor with GPJax package,
        https://github.com/thomaspinder/GPJax.
    """

    def __init__(self, gp_kernel: gpx.kernels.Kernel, random_state=jax.random.PRNGKey(1234)):
        self.kernel = gp_kernel
        self.random_state = random_state

        self.final_params = None
        self.prior = None
        self.likelihood = None
        self.posterior = None
        self.D = None

        self.posterior_fitted = False

    def fit(self, x: jnp.ndarray, y: jnp.ndarray):
        """ Fit Gaussian process regression model. """

        # defining the dataset of the training points
        y = y.reshape(-1, 1)
        self.D = gpx.Dataset(X=x, y=y)
        self.prior = gpx.Prior(kernel=self.kernel)

        self.likelihood = gpx.Gaussian(num_datapoints=self.D.n)
        self.posterior = self.prior * self.likelihood
        self.posterior_fitted = True
        params, trainable, constrainer, unconstrainer = gpx.initialise(self.posterior)

        #  transforming the parameters onto an unconstrained space
        params = gpx.transform(params, unconstrainer)

        # optimising the marginal log-likelihood of the posterior with respect to the hyperparameters
        mll = jit(self.posterior.marginal_log_likelihood(self.D, constrainer, negative=True))

        # declaring the optimiser
        opt = ox.adam(learning_rate=0.01)

        # learned final parameters
        final_params = gpx.fit(
            mll,
            params,
            trainable,
            opt,
            log_rate=500,
            n_iters=500,
        )

        # un-transfoming the trained unconstrained parameters back to their original, constrained space
        self.final_params = gpx.transform(final_params, constrainer)

        return self

    def sample_y(self, xtest: jnp.ndarray, n_samples: int, random_state: jax.random.PRNGKey = None):
        """ Draw samples from Gaussian process and evaluate at X. """
        # assert xtest.shape[-1] == 1
        random_state = self.random_state if random_state is None else random_state

        latent_dist = self.posterior(self.D, self.final_params)(xtest)
        predictive_dist = self.likelihood(latent_dist, self.final_params)

        samples = jnp.array([predictive_dist.sample(seed=random_state) for _ in range(n_samples)])

        return samples.T

    def predict(self, x: jnp.ndarray, return_std: bool = False, return_cov: bool = False):
        """ Predict using the Gaussian process regression model. """
        assert x.shape[-1] == 1

        latent_dist = self.posterior(self.D, self.final_params)(x)
        predictive_dist = self.likelihood(latent_dist, self.final_params)

        if return_std:
            return predictive_dist.mean(), predictive_dist.stddev()
        elif return_cov:
            return predictive_dist.mean(), predictive_dist.covariance()
        else:
            return predictive_dist.mean()

    def plot_prior(self, x: jnp.ndarray):
        """ Plot the predictions corresponding to the GP prior. """
        params, trainable, constrainer, unconstrainer = gpx.initialise(self.prior)
        prior_dist = self.prior(params)(x)

        prior_mean = prior_dist.mean()
        prior_std = jnp.sqrt(prior_dist.covariance().diagonal())
        samples = prior_dist.sample(seed=self.random_state, sample_shape=20).T

        plt.plot(x, samples, color='tab:blue', alpha=0.5)
        plt.plot(x, prior_mean, color='tab:orange')
        plt.fill_between(x.flatten(), prior_mean - prior_std, prior_mean + prior_std, color='tab:orange', alpha=0.3)
        plt.title("GP prior samples")
        plt.show()

    def plot_posterior(self, x_train: jnp.ndarray, y_train: jnp.ndarray, x_test: jnp.ndarray, y_test: jnp.ndarray):
        """ Plot the predictions corresponding to the GP posterior. """
        if not self.posterior_fitted:
            raise NotImplementedError("Posterior not fitted yet.")

        latent_dist = self.posterior(self.D, self.final_params)(x_test)
        predictive_dist = self.likelihood(latent_dist, self.final_params)

        predictive_mean = predictive_dist.mean()
        predictive_std = predictive_dist.stddev()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(x_train, y_train, "o", label="Observations", color="tab:red")
        ax.plot(x_test, predictive_mean, label="Predictive mean", color="tab:blue")
        ax.fill_between(
            x_test.squeeze(),
            predictive_mean - predictive_std,
            predictive_mean + predictive_std,
            alpha=0.2,
            color="tab:blue",
            label='Two sigma',
        )
        ax.plot(x_test, predictive_mean - predictive_std, color="tab:blue", linestyle="--", linewidth=1)
        ax.plot(x_test, predictive_mean + predictive_std, color="tab:blue", linestyle="--", linewidth=1)

        ax.plot(x_test, y_test, label="Latent function", color="black", linestyle="--", linewidth=1)

        ax.legend()
        fig.suptitle("Posterior prediction")
        fig.show()


if __name__ == '__main__':
    seed = 2
    key = hk.PRNGSequence(seed)
    x_range = [-3., 3.]

    # initialise training data
    x = jnp.sort(jax.random.uniform(next(key), shape=(100, 1), minval=x_range[0], maxval=x_range[1]), axis=0)
    noise = 1e-1
    f = lambda x_: jnp.sin(4 * x_) + jnp.cos(2 * x_)
    signal = f(x)

    y = signal + jax.random.normal(next(key), shape=signal.shape) * noise

    # set kernel hyperparameters
    kernel = gpx.RBF()
    kernel._params = {"lengthscale": 1., "variance": 1.}

    # fit the GP
    x_test = jnp.linspace(start=-3.5, stop=3.5, num=300)[..., None]
    y_test = f(x_test)

    gpr = GP(gp_kernel=kernel, random_state=key)
    gp = gpr.fit(x, y)  # (10, 1) and (10,)
    y_predict = gp.sample_y(x_test, 10)

    # plot the prior
    gp.plot_prior(x)

    # plot the posterior
    gp.plot_posterior(x, y, x_test, y_test)
