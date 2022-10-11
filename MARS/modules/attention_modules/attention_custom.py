import haiku as hk
from jax import vmap
from MARS.modules.attention_modules.attention import MultiHeadAttention


class MultiHeadSelfAttention(hk.Module):
    """
    Same as `MultiHeadAttention` but directly doing self attention
    For unified number of arguments with EModule
    """

    def __init__(self, *, w_init, key_size, value_size=None, model_size=None, num_heads=8, name=None):
        super().__init__(name=name)
        self.mha = MultiHeadAttention(w_init=w_init, key_size=key_size, value_size=value_size, model_size=model_size,
                                      num_heads=num_heads)

    def __call__(self, x):
        return self.mha(x, x, x)


# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""(Multi-Head) Attention module to be used in a Transformer architecture."""

import types
from typing import Optional

from haiku._src import basic
from haiku._src import initializers
from haiku._src import module
import jax
import jax.numpy as jnp
import numpy as np

# If you are forking replace this with `import haiku as hk`.
hk = types.ModuleType("haiku")
hk.Module = module.Module
hk.Linear = basic.Linear
hk.transparent = module.transparent
hk.initializers = initializers
del basic, module, initializers
import haiku as hk


class MultiHeadAttentionLearnedDistance(hk.Module):
    """ Multi-headed attention mechanism with learned relative encodings, inspired by:
        "Music transformer" https://arxiv.org/pdf/1809.04281, and
    """

    def __init__(
            self,
            num_heads: int,
            key_size: int,
            w_init_scale: float,
            value_size: Optional[int] = None,
            model_size: Optional[int] = None,
            name: Optional[str] = None,
            add_pos_to_values: Optional[bool] = False,
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size or key_size
        self.model_size = model_size or key_size * num_heads
        self.w_init = hk.initializers.VarianceScaling(w_init_scale)
        self.add_pos_to_values = add_pos_to_values

    def __call__(
            self,
            query: jnp.ndarray,
            key: jnp.ndarray,
            value: jnp.ndarray,
            dist: Optional[jnp.ndarray] = None,
            mask: Optional[jnp.ndarray] = None,

    ) -> jnp.ndarray:

        """Compute (optionally masked) MHA with queries, keys & values."""
        query_heads = self._linear_projection(query, self.key_size, "query")
        key_heads = self._linear_projection(key, self.key_size, "key")
        value_heads = self._linear_projection(value, self.value_size, "value")

        attn_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
        sqrt_key_size = np.sqrt(self.key_size).astype(key.dtype)

        attn_logits = hk.Bias(bias_dims=[])(attn_logits) / sqrt_key_size
        if mask is not None:
            if mask.ndim != attn_logits.ndim:
                raise ValueError(f"Mask dimensionality {mask.ndim} must match logits "
                                 f"{attn_logits.ndim}.")
            attn_logits = jnp.where(mask, attn_logits, -1e30)

        attn_weights = jax.nn.softmax(attn_logits)
        if self.add_pos_to_values:
            attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
        else:
            attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, hk.Bias(bias_dims=[])(value_heads))

        # Concatenate attention matrix of all heads into a single vector.
        attn_vec = jnp.reshape(attn, (*query.shape[:-1], -1))

        return hk.Linear(self.model_size, w_init=self.w_init)(attn_vec)

    @hk.transparent
    def _linear_projection(
            self,
            x: jnp.ndarray,
            head_size: int,
            name: Optional[str] = None
    ) -> jnp.ndarray:
        y = hk.Linear(self.num_heads * head_size, w_init=self.w_init, name=name)(x)
        return y.reshape((*x.shape[:-1], self.num_heads, head_size))


class MultiHeadAttentionWithDistance(hk.Module):
    """ Multi-headed attention mechanism with positional distance added as relative encodings, inspired by:
        "Self-Attention with Relative Position Representations" https://arxiv.org/abs/1803.02155.
    """
    def __init__(
            self,
            num_heads: int,
            key_size: int,
            w_init_scale: float,
            value_size: Optional[int] = None,
            model_size: Optional[int] = None,
            name: Optional[str] = None,
            add_pos_to_values: Optional[bool] = False,
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size or key_size
        self.model_size = model_size or key_size * num_heads
        self.w_init = hk.initializers.VarianceScaling(w_init_scale)
        self.add_pos_to_values = add_pos_to_values

    def __call__(
            self,
            query: jnp.ndarray,
            key: jnp.ndarray,
            value: jnp.ndarray,
            dist: Optional[jnp.ndarray] = None,
            mask: Optional[jnp.ndarray] = None,

    ) -> jnp.ndarray:
        """Compute (optionally masked) MHA with queries, keys & values."""
        query_heads = self._linear_projection(query, self.key_size, "query")
        key_heads = self._linear_projection(key, self.key_size, "key")
        value_heads = self._linear_projection(value, self.value_size, "value")

        attn_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
        sqrt_key_size = np.sqrt(self.key_size).astype(key.dtype)

        if dist is not None:
            attn_logits = (attn_logits + dist) / sqrt_key_size
        else:
            attn_logits = attn_logits / sqrt_key_size

        if mask is not None:
            if mask.ndim != attn_logits.ndim:
                raise ValueError(f"Mask dimensionality {mask.ndim} must match logits "
                                 f"{attn_logits.ndim}.")
            attn_logits = jnp.where(mask, attn_logits, -1e30)

        attn_weights = jax.nn.softmax(attn_logits)
        if self.add_pos_to_values and dist is not None:
            attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads + dist.reshape(value_heads.shape))
        else:
            attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads)

        # Concatenate attention matrix of all heads into a single vector.
        attn_vec = jnp.reshape(attn, (*query.shape[:-1], -1))

        return hk.Linear(self.model_size, w_init=self.w_init)(attn_vec)

    @hk.transparent
    def _linear_projection(
            self,
            x: jnp.ndarray,
            head_size: int,
            name: Optional[str] = None
    ) -> jnp.ndarray:
        y = hk.Linear(self.num_heads * head_size, w_init=self.w_init, name=name)(x)
        return y.reshape((*x.shape[:-1], self.num_heads, head_size))


class MultiHeadAttentionWithDistanceAndNorm(hk.Module):
    """Multi-headed attention mechanism with normalized self-attention layers, inspired by:
        "Lipschitz normalization for self-attention layers with application to graph neural network"
        https://arxiv.org/abs/2103.04886
    """

    def __init__(
            self,
            num_heads: int,
            key_size: int,
            w_init_scale: float,
            value_size: Optional[int] = None,
            model_size: Optional[int] = None,
            name: Optional[str] = None,
            add_pos_to_values: Optional[bool] = False,
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size or key_size
        self.model_size = model_size or key_size * num_heads
        self.w_init = hk.initializers.VarianceScaling(w_init_scale)
        self.add_pos_to_values = add_pos_to_values
        self.norm = quadratic_lipschitz_norm

    def __call__(
            self,
            query: jnp.ndarray,
            key: jnp.ndarray,
            value: jnp.ndarray,
            dist: Optional[jnp.ndarray] = None,
            mask: Optional[jnp.ndarray] = None,

    ) -> jnp.ndarray:
        """Compute (optionally masked) MHA with queries, keys & values."""

        query_heads = self._linear_projection(query, self.key_size, "query")
        key_heads = self._linear_projection(key, self.key_size, "key")
        value_heads = self._linear_projection(value, self.value_size, "value")

        attn_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
        sqrt_key_size = np.sqrt(self.key_size).astype(key.dtype)

        if dist is not None:
            attn_logits = (attn_logits + dist) / sqrt_key_size
        else:
            attn_logits = attn_logits / sqrt_key_size

        if mask is not None:
            if mask.ndim != attn_logits.ndim:
                raise ValueError(f"Mask dimensionality {mask.ndim} must match logits "
                                 f"{attn_logits.ndim}.")
            attn_logits = jnp.where(mask, attn_logits, -1e30)

        attn_logits = self.norm(query_heads, key_heads, value_heads, attn_logits)

        attn_weights = jax.nn.softmax(attn_logits)
        if self.add_pos_to_values and dist is not None:
            attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads + dist.reshape(value_heads.shape))
        else:
            attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads)

        # Concatenate attention matrix of all heads into a single vector.
        attn_vec = jnp.reshape(attn, (*query.shape[:-1], -1))

        return hk.Linear(self.model_size, w_init=self.w_init)(attn_vec)

    @hk.transparent
    def _linear_projection(
            self,
            x: jnp.ndarray,
            head_size: int,
            name: Optional[str] = None
    ) -> jnp.ndarray:
        y = hk.Linear(self.num_heads * head_size, w_init=self.w_init, name=name)(x)
        return y.reshape((*x.shape[:-1], self.num_heads, head_size))


def quadratic_lipschitz_norm(query_heads: jnp.ndarray, key_heads: jnp.ndarray, value_heads: jnp.ndarray,
                             attn_logits: jnp.ndarray, eps: float = 1e-7):
    """ Quadratis Lipschitz norm calculation. """
    Q_F = jnp.linalg.norm(query_heads, axis=(0, 2))
    K_2 = jnp.linalg.norm(key_heads, axis=-1)
    V_2 = jnp.linalg.norm(value_heads, axis=-1)
    K_inf_2 = vmap(jnp.max, 1)(K_2)
    V_inf_2 = vmap(jnp.max, 1)(V_2)

    uv = Q_F * K_inf_2
    uw = Q_F * V_inf_2
    vw = K_inf_2 * V_inf_2

    max_over_norms = jnp.max(jnp.stack((uv, uw, vw)), axis=0)
    attn_logits = vmap(lambda x, y: x / (y + eps))(attn_logits, max_over_norms)
    return attn_logits
