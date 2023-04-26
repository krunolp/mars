import jax
import optax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import haiku as hk
import warnings

from typing import Any
from tqdm import tqdm
from tensorflow_probability.substrates import jax as tfp
from jax import vmap
from MARS.modules.attention_modules.architectures_refactored import arch1
from MARS.modules.score_network.losses import score_net_loss
from MARS.modules.utils import numeric_integration
from MARS.modules.data_modules.rff import get_data

jax.config.update("jax_enable_x64", False)

tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels


def warn(*args, **kwargs):
    pass


warnings.warn = warn


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


class ScoreEstimator:
    def __init__(self,
                 num_meas_pts: int = 2,
                 x_dim: int = 1,
                 n_fn_samples: int = 4,
                 learning_rate: float = 1e-1,
                 weight_decay: float = 0.,

                 loss_type: str = "exact_sm",
                 spectr_penalty_multiplier: float = 1.,
                 rkhs_pen_coeff: float = 1e-4,
                 bandwidth: float = 1.,
                 spectr_norm_const: float = 1.,
                 grad_pen_const: float = 1.,

                 attn_num_layers: int = 2,
                 attn_architecture: Any = arch1,
                 attn_dim=64,
                 attn_key_size=32,
                 attn_num_heads=8,

                 transition_steps: int = 1000,
                 scheduler_type: int = 0,
                 eps_root: float = 1e-16,
                 decay_rate: float = 1.) -> None:

        self.n_fn_samples = n_fn_samples
        self.x_dim = x_dim
        self.num_meas_pts = num_meas_pts

        if scheduler_type == 0:
            scheduler = optax.exponential_decay(
                init_value=learning_rate,
                transition_steps=transition_steps,
                decay_rate=decay_rate)
        else:
            scheduler = optax.cosine_decay_schedule(
                init_value=learning_rate,
                decay_steps=transition_steps)

        # Combining gradient transforms using `optax.chain`.
        gradient_transform = optax.chain(
            optax.additive_weight_decay(weight_decay),
            optax.scale_by_adam(eps_root=eps_root),
            optax.scale_by_schedule(scheduler),
            optax.scale(-1.0)
        )

        self.optimizer = gradient_transform
        self.loss_type = loss_type
        self.spectr_norm_const = spectr_norm_const

        # initializing the score network architecture
        model_kwargs = {"num_meas_points": self.num_meas_pts,
                        "x_dim": self.x_dim,

                        "dim": attn_dim,
                        "layers": attn_num_layers,
                        "key_size": attn_key_size,
                        "num_heads": attn_num_heads,

                        "layer_norm": False if "exact_w_spectr_norm" in loss_type else True,
                        "widening_factor": 2,
                        "dropout": 0.0,
                        "ln_axis": {"last": -1, "lasttwo": (-2, -1)}["last"],
                        }
        self.nn = hk.transform(lambda *args: attn_architecture(**model_kwargs)(*args))

        # initializing the loss function
        loss_init = score_net_loss(loss_type=loss_type,
                                   nn=self.nn,
                                   x_dim=self.x_dim,
                                   spectr_penalty_multiplier=spectr_penalty_multiplier,
                                   rkhs_pen_coeff=rkhs_pen_coeff,
                                   bandwidth=bandwidth,
                                   spectr_norm_const=spectr_norm_const,
                                   grad_pen_const=grad_pen_const
                                   )
        self.loss = loss_init.apply
        self.spectr_norm = loss_init.spectr_norm_apply

    def train(self, n_fns: int = 20, n_iter: int = 1000, log_iter: int = 10, key: hk.PRNGSequence = None):
        """ Main training loop. """

        # using RFF to simulate the data
        x_iter, y_iter, score_iter = get_data(self.num_meas_pts, n_fns, num_iters=n_iter, key=next(key))
        x, y, score = next(x_iter), next(y_iter), next(score_iter)

        x_fx_init = vmap(lambda y_: jnp.stack((x[..., 0], y_), -1))(y)

        param = self.nn.init(next(key), x_fx_init[0, ...])
        opt_state = self.optimizer.init(param)

        # training loop
        for i in tqdm(range(n_iter), desc="Training score network"):
            x, y, true_score = next(x_iter), next(y_iter), next(score_iter)
            if self.loss_type == 'exact_w_spectr_norm': param = self.spectr_norm(param)
            x_fx = vmap(lambda y_: jnp.stack((x[..., 0], y_), -1))(y)

            l, grads = jax.value_and_grad(self.loss)(param, x_fx, next(key))
            updates, opt_state = self.optimizer.update(grads, opt_state, param)
            param = optax.apply_updates(param, updates)

            if i % log_iter == 0 and i > 0:
                est_score = jax.vmap(lambda xx: self.nn.apply(param, next(key), xx))(x_fx)
                cos_sim = jnp.mean(optax.cosine_similarity(true_score, est_score))
                mse = jnp.mean(jnp.square(true_score - est_score))

                print("Cos. sim. is : ", cos_sim, " MSE is: ", mse)
                self.plot_interp(param, i, next(key))

    def plot_interp(self, param: jnp.ndarray, it: int, rng_key: jax.random.PRNGKey, num_evals: int = 50):
        """ Plots the corresponding predicted score and the numerically integrated distribution. """
        key, subkey = jax.random.split(rng_key)
        fig, axs = plt.subplots(2, 2)

        x1 = jnp.ones(num_evals) * (-0.5)
        x2 = jnp.ones(num_evals) * 0.5
        f_interp = jnp.linspace(-2, 2, num_evals)
        f_zeros = jnp.zeros(num_evals)

        x_fx = jnp.stack((jnp.stack((x1, x2), -1), jnp.stack((f_zeros, f_interp), -1)), -1)
        preds1 = vmap(lambda xfx: self.nn.apply(param, key, xfx))(x_fx)[..., 1]
        axs[0, 0].plot(f_interp, preds1, label='predicted', color='#42529C')
        axs[0, 0].plot(f_interp, -f_interp, label='true', linestyle='--', color='#4958AD')
        axs[0, 0].set_title('Score: x=-0.5')
        axs[0, 0].legend()

        x_distr1, y_distr1 = numeric_integration(f_interp, preds1)

        axs[0, 1].plot(x_distr1, y_distr1, label='predicted', color='#42529C')
        axs[0, 1].plot(x_distr1, jnp.exp(-x_distr1 ** 2) / jnp.sum(jnp.exp(-x_distr1 ** 2)), label='true',
                       linestyle='--', color='#4958AD')
        axs[0, 1].set_title('Distr:  x=-0.5')

        x_fx = jnp.stack((jnp.stack((x1, x2), -1), jnp.stack((f_interp, f_zeros), -1)), -1)
        preds2 = vmap(lambda xfx: self.nn.apply(param, subkey, xfx))(x_fx)[..., 0]
        axs[1, 0].plot(f_interp, preds2, label='predicted', color='#42529C')
        axs[1, 0].plot(f_interp, -f_interp, label='true', linestyle='--', color='#4958AD')
        axs[1, 0].set_title('Score: x=0.5')

        x_distr2, y_distr2 = numeric_integration(f_interp, preds2)
        axs[1, 1].plot(x_distr2, y_distr2, label='predicted', color='#42529C')
        axs[1, 1].plot(x_distr2, jnp.exp(-x_distr2 ** 2) / jnp.sum(jnp.exp(-x_distr2 ** 2)), label='True',
                       linestyle='--', color='#4958AD')
        axs[1, 1].set_title('Distr:  x=-0.5')

        fig.suptitle('Iteration: ' + str(it))
        fig.tight_layout()
        fig.show()


if __name__ == '__main__':
    seed = 10
    hk_key = hk.PRNGSequence(seed)

    # initializer score network
    score_net = ScoreEstimator(learning_rate=1e-3, loss_type="exact_w_spectr_norm")
    # train score network
    score_net.train(n_iter=200, n_fns=10, key=hk_key)
