import optax
import haiku as hk
from tensorflow_probability.substrates import jax as tfp
import jax
import jax.numpy as jnp
import warnings

from typing import Any
from typing import Union
from tqdm import tqdm
from MARS.modules.data_modules.simulator_base import GPMetaDataset, GPMetaDatasetJax
from MARS.modules.attention_modules.architectures_refactored import arch1
from MARS.modules.score_network.losses import score_net_loss

tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels

KeyArray = Union[Any, jax.random.PRNGKey]


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
    return hk.Transformed(init_stateless, apply_stateless)


class FunctionSimulator:

    def __init__(self, input_size: int, output_size: int, finite_fns: bool, num_fns: Union[int, None],
                 init_seed: jax.random.PRNGKey):
        self.input_size = input_size
        self.output_size = output_size

        self.finite_fns = finite_fns
        if self.finite_fns:
            assert num_fns is not None and init_seed is not None
            self.init_seed = init_seed
            self.num_fns = num_fns
            self.seeds = iter(jax.random.split(init_seed, num_fns))

    def sample_function_vals(self, x: jnp.ndarray, num_samples: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        raise NotImplementedError

    def get_rng_key(self) -> jax.random.PRNGKey:
        try:
            rng_key = next(self.seeds)
        except StopIteration:
            self.seeds = iter(jax.random.split(self.init_seed, self.num_fns))
            rng_key = next(self.seeds)
        return rng_key


class ScoreEstimator:
    def __init__(self,
                 function_sim: Any,
                 sample_from_gp: bool,
                 x_dim: int = None,
                 n_fn_samples: int = 10,
                 learning_rate: float = 1e-2,
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
                 decay_rate: float = 1.):

        self.function_sim = function_sim
        self.sample_from_gp = sample_from_gp
        self.n_fn_samples = n_fn_samples
        self.x_dim = x_dim if x_dim is not None else function_sim.input_size

        self.num_x_pts = self.function_sim.num_pts

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

        self.param = None
        self.opt_state = None

        model_kwargs = {"num_meas_points": self.num_x_pts,
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

    def get_x_fx(self, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        """ Obtains the concatenation of inputs & outputs from the GP interpolation. """
        if self.sample_from_gp:
            x_fx = self.function_sim.sample_x_fx(self.n_fn_samples, rng_key)
        else:
            x_fx = self.function_sim.sample_x_fx_mean(self.n_fn_samples, rng_key)
        return x_fx

    def train(self, hk_key: hk.PRNGSequence = hk.PRNGSequence(1234), n_iter: int = 20000):
        """ Main training loop. """
        x_fx_init = self.get_x_fx(next(hk_key))
        if self.param is None and self.opt_state is None:
            param = self.nn.init(next(hk_key), x_fx_init[0, ...])
            opt_state = self.optimizer.init(param)
        else:
            param = self.param
            opt_state = self.opt_state

        # training loop
        pbar = tqdm(range(n_iter))
        for _ in pbar:
            x_fx = self.get_x_fx(next(hk_key))
            l, grads = jax.value_and_grad(self.loss)(param, x_fx, next(hk_key))
            updates, opt_state = self.optimizer.update(grads, opt_state, param)
            param = optax.apply_updates(param, updates)

            pbar.set_description("Training SNN., loss is %s" % l)

        self.param = param
        self.opt_state = opt_state


if __name__ == '__main__':
    seed = 2
    key = hk.PRNGSequence(seed)
    sim = GPMetaDataset(dataset="sin_2", init_seed=next(key), num_pts=10)

    est = ScoreEstimator(function_sim=sim,
                         sample_from_gp=True,
                         x_dim=1,
                         loss_type="exact_w_grad_pen",
                         learning_rate=0.001,
                         )

    est.train(key, n_iter=20000)
