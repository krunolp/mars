import jax
import jax.numpy as jnp
import haiku as hk

from MARS.modules.attention_modules.attention_custom import MultiHeadAttentionWithDistance


def hk_layer_norm(*, axis, name=None):
    return hk.LayerNorm(axis=axis, create_scale=True, create_offset=True, name=name)


class BaseModel(hk.Module):
    def __init__(self,
                 n_mixtures=1,
                 dim=128,
                 layers=8,
                 dropout=0.1,
                 ln_axis=-1,
                 widening_factor=4,
                 num_heads=8,
                 key_size=32,
                 logit_bias_init=-3.0,
                 x_dim=None,
                 y_dim=None,
                 mixture_drop=None,
                 name="BaseModel",
                 ):
        super().__init__(name=name)
        self.n_mixtures = n_mixtures
        self.dim = dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.layers = 2 * layers
        self.dropout = dropout
        self.mixture_drop = mixture_drop or dropout
        self.ln_axis = ln_axis
        self.widening_factor = widening_factor
        self.num_heads = num_heads
        self.key_size = key_size
        self.logit_bias_init = logit_bias_init
        self.w_init = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")

    def __call__(self, x):
        dropout_rate = 0.0
        z = hk.Linear(self.dim)(x)

        for _ in range(self.layers):
            # mha
            q_in = hk_layer_norm(axis=self.ln_axis)(z)
            k_in = hk_layer_norm(axis=self.ln_axis)(z)
            v_in = hk_layer_norm(axis=self.ln_axis)(z)
            z_attn = hk.MultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.key_size,
                w_init_scale=2.0,
                model_size=self.dim,
            )(q_in, k_in, v_in)
            z = z + hk.dropout(hk.next_rng_key(), dropout_rate, z_attn)

            # ffn
            z_in = hk_layer_norm(axis=self.ln_axis)(z)

            z_ffn = hk.Sequential([
                hk.Linear(self.widening_factor * self.dim, w_init=self.w_init),
                jax.nn.relu,
                hk.Linear(self.dim, w_init=self.w_init),
            ])(z_in)

            z = z + hk.dropout(hk.next_rng_key(), dropout_rate, z_ffn)

        z = hk_layer_norm(axis=self.ln_axis)(z)

        out_dim = int(self.x_dim * self.y_dim)
        score = jnp.squeeze(hk.Sequential([
            hk_layer_norm(axis=self.ln_axis),
            hk.Linear(out_dim, w_init=self.w_init),
        ])(z))
        return score


class arch1(hk.Module):
    """ Attention-based architecture which uses a combined embedding on the input-output concatenation. """
    def __init__(self,
                 num_meas_points=1,
                 dim=128,
                 layers=8,
                 layer_norm=True,
                 x_dim=12,
                 dropout=0.1,
                 ln_axis=-1,
                 widening_factor=4,
                 num_heads=8,
                 key_size=32,
                 logit_bias_init=-3.0,
                 out_dim=None,
                 mixture_drop=None,
                 name="BaseModel",
                 add_pos_to_values=False
                 ):
        super().__init__(name=name)
        self.num_meas_points = num_meas_points
        self.dim = dim
        self.x_dim = x_dim
        self.layer_norm = layer_norm
        self.out_dim = out_dim or dim
        self.layers = 2 * layers
        self.dropout = dropout
        self.mixture_drop = mixture_drop or dropout
        self.ln_axis = ln_axis
        self.widening_factor = widening_factor
        self.num_heads = num_heads
        self.key_size = key_size
        self.logit_bias_init = logit_bias_init
        self.w_init = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")
        self.add_pos_to_values = add_pos_to_values  # kaiming uniform

    def __call__(self, x, is_training: bool = False):
        dropout_rate = self.dropout if is_training else 0.0
        z = hk.Linear(self.dim)(x)
        for _ in range(self.layers):
            # mha
            q_in = hk_layer_norm(axis=self.ln_axis)(z) if self.layer_norm else z
            k_in = hk_layer_norm(axis=self.ln_axis)(z) if self.layer_norm else z
            v_in = hk_layer_norm(axis=self.ln_axis)(z) if self.layer_norm else z
            z_attn = MultiHeadAttentionWithDistance(
                num_heads=self.num_heads,
                key_size=self.key_size,
                w_init_scale=2.0,
                model_size=self.dim,
                add_pos_to_values=self.add_pos_to_values
            )(q_in, k_in, v_in)
            z = z + hk.dropout(hk.next_rng_key(), dropout_rate, z_attn)

            # ffn
            z_in = hk_layer_norm(axis=self.ln_axis)(z) if self.layer_norm else z

            z_ffn = hk.Sequential([
                hk.Linear(self.widening_factor * self.dim, w_init=self.w_init),
                jax.nn.gelu,
                hk.Linear(self.dim, w_init=self.w_init),
            ])(z_in)

            z = z + hk.dropout(hk.next_rng_key(), dropout_rate, z_ffn)

        z = hk_layer_norm(axis=self.ln_axis)(z) if self.layer_norm else z

        # estimated score
        if self.layer_norm:
            score = jnp.squeeze(hk.Sequential([
                hk_layer_norm(axis=self.ln_axis),
                hk.Linear(1, w_init=self.w_init),
            ])(z))
        else:
            score = jnp.squeeze(hk.Sequential([
                hk.Linear(1, w_init=self.w_init),
            ])(z))
        return score


class arch2(hk.Module):
    """ Attention-based architecture which uses a separate embedding on the inputs & outputs. """

    def __init__(self,
                 num_meas_points=1,
                 dim=128,
                 layers=8,
                 x_dim=12,
                 layer_norm=True,
                 dropout=0.1,
                 ln_axis=-1,
                 widening_factor=4,
                 num_heads=8,
                 key_size=32,
                 logit_bias_init=-3.0,
                 out_dim=None,
                 mixture_drop=None,
                 name="BaseModel",
                 add_pos_to_values=False
                 ):
        super().__init__(name=name)
        self.num_meas_points = num_meas_points
        self.dim = dim
        self.x_dim = x_dim
        self.layer_norm = layer_norm
        self.out_dim = out_dim or dim
        self.layers = 2 * layers
        self.dropout = dropout
        self.mixture_drop = mixture_drop or dropout
        self.ln_axis = ln_axis
        self.widening_factor = widening_factor
        self.num_heads = num_heads
        self.key_size = key_size
        self.logit_bias_init = logit_bias_init
        self.w_init = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")
        self.add_pos_to_values = add_pos_to_values  # kaiming uniform

    def __call__(self, x, is_training: bool = False):
        dropout_rate = self.dropout if is_training else 0.0
        z1 = hk.Linear(int(self.dim / 2))(x[..., :self.x_dim])
        z2 = hk.Linear(int(self.dim / 2))(x[..., self.x_dim:])
        z = jnp.hstack((z1, z2))

        for _ in range(self.layers):
            # mha
            q_in = hk_layer_norm(axis=self.ln_axis)(z) if self.layer_norm else z
            k_in = hk_layer_norm(axis=self.ln_axis)(z) if self.layer_norm else z
            v_in = hk_layer_norm(axis=self.ln_axis)(z) if self.layer_norm else z
            z_attn = MultiHeadAttentionWithDistance(
                num_heads=self.num_heads,
                key_size=self.key_size,
                w_init_scale=2.0,
                model_size=self.dim,
                add_pos_to_values=self.add_pos_to_values
            )(q_in, k_in, v_in)
            z = z + hk.dropout(hk.next_rng_key(), dropout_rate, z_attn)

            # ffn
            z_in = hk_layer_norm(axis=self.ln_axis)(z) if self.layer_norm else z

            z_ffn = hk.Sequential([
                hk.Linear(self.widening_factor * self.dim, w_init=self.w_init),
                jax.nn.gelu,
                hk.Linear(self.dim, w_init=self.w_init),
            ])(z_in)

            z = z + hk.dropout(hk.next_rng_key(), dropout_rate, z_ffn)

        z = hk_layer_norm(axis=self.ln_axis)(z) if self.layer_norm else z

        if self.layer_norm:
            score = jnp.squeeze(hk.Sequential([
                hk_layer_norm(axis=self.ln_axis),
                hk.Linear(1, w_init=self.w_init),
            ])(z))
        else:
            score = jnp.squeeze(hk.Sequential([
                hk.Linear(1, w_init=self.w_init),
            ])(z))
        return score


class arch3(hk.Module):
    """ Attention-based architecture which uses inputs as keys & queries, outputs as values. """

    def __init__(self,
                 num_meas_points=1,
                 dim=128,
                 layers=8,
                 x_dim=12,
                 dropout=0.1,
                 ln_axis=-1,
                 layer_norm=True,
                 widening_factor=4,
                 num_heads=8,
                 key_size=32,
                 logit_bias_init=-3.0,
                 out_dim=None,
                 mixture_drop=None,
                 name="BaseModel",
                 add_pos_to_values=False
                 ):
        super().__init__(name=name)
        self.num_meas_points = num_meas_points
        self.dim = dim
        self.layer_norm = layer_norm
        self.x_dim = x_dim
        self.out_dim = out_dim or dim
        self.layers = 2 * layers
        self.dropout = dropout
        self.mixture_drop = mixture_drop or dropout
        self.ln_axis = ln_axis
        self.widening_factor = widening_factor
        self.num_heads = num_heads
        self.key_size = key_size
        self.logit_bias_init = logit_bias_init
        self.w_init = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")
        self.add_pos_to_values = add_pos_to_values  # kaiming uniform

    def __call__(self, x, is_training: bool = False):
        dropout_rate = self.dropout if is_training else 0.0
        z1 = hk.Linear(int(self.dim))(x[..., :self.x_dim])
        z2 = hk.Linear(int(self.dim))(x[..., self.x_dim:])

        for _ in range(self.layers):
            # mha
            q_in = hk_layer_norm(axis=self.ln_axis)(z1) if self.layer_norm else z1
            k_in = hk_layer_norm(axis=self.ln_axis)(z1) if self.layer_norm else z1
            v_in = hk_layer_norm(axis=self.ln_axis)(z2) if self.layer_norm else z2
            z_attn = MultiHeadAttentionWithDistance(
                num_heads=self.num_heads,
                key_size=self.key_size,
                w_init_scale=2.0,
                model_size=self.dim,
                add_pos_to_values=self.add_pos_to_values
            )(q_in, k_in, v_in)
            z2 = z2 + hk.dropout(hk.next_rng_key(), dropout_rate, z_attn)

            # ffn
            z_in = hk_layer_norm(axis=self.ln_axis)(z2) if self.layer_norm else z2

            z_ffn = hk.Sequential([
                hk.Linear(self.widening_factor * self.dim, w_init=self.w_init),
                jax.nn.elu,
                hk.Linear(self.dim, w_init=self.w_init),
            ])(z_in)

            z2 = z2 + hk.dropout(hk.next_rng_key(), dropout_rate, z_ffn)

        z2 = hk_layer_norm(axis=self.ln_axis)(z2) if self.layer_norm else z2

        if self.layer_norm:
            score = jnp.squeeze(hk.Sequential([
                hk_layer_norm(axis=self.ln_axis),
                hk.Linear(1, w_init=self.w_init),
            ])(z2))
        else:
            score = jnp.squeeze(hk.Sequential([
                hk.Linear(1, w_init=self.w_init),
            ])(z2))
        return score
