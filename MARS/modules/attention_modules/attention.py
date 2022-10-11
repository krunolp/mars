import haiku as hk
import jax
import jax.numpy as jnp


class Identity(hk.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self, x):
        return x


class MultiHeadAttention(hk.Module):
    """
    Multi-headed attention mechanism.
    Code adapted from https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/attention.py

    Default:
        key_size: 64
        num_heasds: 8
        model_size: 512

    """

    def __init__(self, *, w_init, key_size, value_size=None, model_size=None, num_heads=8, name=None):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size or key_size
        self.model_size = model_size or key_size * num_heads
        self.w_init = w_init

    def __call__(self, q, k, v, mask=None):
        """Compute (optionally masked) multi-head attention
        Args:
            q:  [..., N, k_in]  usually: k_in = model_size
            k:  [..., M, k_in]  usually: k_in = model_size
            v:  [..., M, v_in]  usually: v_in = model_size

            number of keys and values must be the same ("soft lookup")
            queries are compared with keys and extracts values by similarity with keys

        Returns:
            [..., N, model_size]

        """
        # [..., N, q_in] -> [..., N, h, key_size]
        query_heads = self._linear_projection(q, self.key_size, "query")

        # [..., M, k_in] -> [..., M, h, key_size]
        key_heads = self._linear_projection(k, self.key_size, "key")

        # [..., M, v_in] -> [..., M, h, value_size]
        value_heads = self._linear_projection(v, self.value_size, "value")

        # [..., h, N, M]
        attn_logits = jnp.einsum("...Nhd,...Mhd->...hNM", query_heads, key_heads)
        attn_logits = attn_logits / jnp.sqrt(self.key_size).astype(k.dtype)
        if mask is not None:
            assert len(mask.shape) == len(attn_logits.shape)
            attn_logits = jnp.where(mask, attn_logits, -1e30)

        # [..., h, N, M]
        # sums to 1 across last dim (over M) for "soft look-up" of values
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)

        # [..., N, h, value_size]
        # h "soft looked-up" embeddings for each input obj N
        attn = jnp.einsum("...hNM,...Mhd->...Nhd", attn_weights, value_heads)

        # [..., N, h * value_size]
        # concatenate attention matrix of all heads into a single vector
        attn_vec = jnp.reshape(attn, (*q.shape[:-1], -1))

        # [..., N, model_size]
        # linear transformation to get desired output representation dim
        return hk.Linear(self.model_size, w_init=self.w_init)(attn_vec)

    def _linear_projection(self, x, head_size, name):
        """Linear projection
        Args:
            x:  [..., dim]
            name: necessary to identify unique linear module
        Returns:
            [..., num_heads, head_size]

        """
        y = hk.Linear(self.num_heads * head_size, w_init=self.w_init, name=name)(x)
        return y.reshape((*x.shape[:-1], self.num_heads, head_size))


class MAB(hk.Module):
    """
    Multi-headed attention block
    (Plain-vanilla transformer encoder block)

    Defined as in https://arxiv.org/pdf/1810.00825.pdf
    """

    def __init__(self, *, w_init, key_size, value_size=None, model_size=None, num_heads=8, name=None,
                 use_ln=True, widening_factor=4, dropout_rate=0.0, nonlinearity=jax.nn.relu):
        super().__init__(name=name)

        value_size = value_size or key_size
        model_size = model_size or key_size * num_heads
        self._dropout_rate = dropout_rate

        # submodules
        self.mha = MultiHeadAttention(w_init=w_init, key_size=key_size, value_size=value_size, model_size=model_size,
                                      num_heads=num_heads, name="mha")
        self.ffn = hk.Sequential([
            hk.Linear(widening_factor * model_size, w_init=w_init),  # here original transformer has model_size * 4
            nonlinearity,
            hk.Linear(model_size, w_init=w_init),
        ], name="ffn")

        # norm
        self.ln0 = hk.LayerNorm(create_scale=True, create_offset=True, axis=-1) if use_ln else Identity()
        self.ln1 = hk.LayerNorm(create_scale=True, create_offset=True, axis=-1) if use_ln else Identity()

    def __call__(self, q, k, is_training: bool):
        """Compute residual multi-head block (transformer encoder)

        Args:
            q:  [..., N, model_size]
            k:  [..., M, model_size]

        Returns:
            [..., N, model_size]

        """
        assert q.shape[-1] == self.mha.model_size
        dropout_rate = self._dropout_rate if is_training else 0.0

        # [..., N, model_size]
        h = self.ln0(q + hk.dropout(hk.next_rng_key(), dropout_rate, self.mha(q, k, k)))

        # [..., N, model_size]
        o = self.ln1(h + hk.dropout(hk.next_rng_key(), dropout_rate, self.ffn(h)))

        return o


class SAB(hk.Module):
    """
    Multi-headed self-attention block
    = MAB(X, X)
    """

    def __init__(self, *, w_init, key_size, value_size=None, model_size=None, num_heads=8, name=None,
                 widening_factor=4, use_ln=True, dropout_rate=0.0, nonlinearity=jax.nn.relu):
        super().__init__(name=name)
        self.mab = MAB(w_init=w_init, key_size=key_size, value_size=value_size, model_size=model_size,
                       num_heads=num_heads,
                       use_ln=use_ln, widening_factor=widening_factor, nonlinearity=nonlinearity,
                       dropout_rate=dropout_rate)

    def __call__(self, x, is_training: bool):
        """Compute residual multi-head self-attention block
        Args:
            x:  [..., N, model_size]

        Returns:
            [..., N, model_size]

        """
        return self.mab(x, x, is_training=is_training)


class PMA(hk.Module):
    """
    Pooled multi-headed attention
    = MAB(S, X) for a set of learned seed vectors S of shape [n_features, model_size]

    When truly pooling, i.e. getting rid of a dimension, we want n_features = 1.
    The set transformer paper states ffN(X) instead of X, but their code only uses X, which we do as well
    """

    def __init__(self, *, w_init, key_size, value_size=None, model_size=None, num_heads=8, name=None,
                 widening_factor=4, n_features=1, use_ln=True, dropout_rate=0.0, nonlinearity=jax.nn.relu):
        super().__init__(name=name)

        self.w_init = w_init
        self.model_size = model_size
        self.n_features = n_features
        self.mab = MAB(w_init=w_init, key_size=key_size, value_size=value_size, model_size=model_size,
                       num_heads=num_heads,
                       use_ln=use_ln, widening_factor=widening_factor, nonlinearity=nonlinearity,
                       dropout_rate=dropout_rate)

    def __call__(self, x, is_training: bool):
        """
        Args:
            x:  [..., N, model_size]

        Returns:
            [..., n_features, model_size]

        """
        s = hk.get_parameter("seed_vectors", (1, self.n_features, self.model_size), x.dtype,
                             init=self.w_init)

        return self.mab(jnp.tile(s, (x.shape[0], 1, 1)), x, is_training=is_training)
