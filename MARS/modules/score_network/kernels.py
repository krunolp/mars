"""""
The following code directly follows from the source code by:
Nonparametric Score Estimators
Yuhao Zhou, Jiaxin Shi, Jun Zhu. https://arxiv.org/abs/2005.10099
Source code: https://github.com/miskcoo/kscore
"""""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import jax
import jax.numpy as jnp


class Base:

    def __init__(self, kernel_type, kernel_hyperparams, heuristic_hyperparams):
        if kernel_hyperparams is not None:
            heuristic_hyperparams = lambda x, y: kernel_hyperparams
        self._kernel_type = kernel_type
        self._heuristic_hyperparams = heuristic_hyperparams

    def kernel_type(self):
        return self._kernel_type

    def heuristic_hyperparams(self, x, y):
        return self._heuristic_hyperparams(x, y)

    def kernel_operator(self, x, y, kernel_hyperparams, **kwargs):
        raise NotImplementedError

    def kernel_matrix(self, x, y, kernel_hyperparams=None, flatten=True, compute_divergence=True):
        if compute_divergence:
            op, divergence = self.kernel_operator(x, y,
                                                  compute_divergence=True,
                                                  kernel_hyperparams=kernel_hyperparams)
            return op.kernel_matrix(flatten), divergence
        op = self.kernel_operator(x, y, compute_divergence=False,
                                  kernel_hyperparams=kernel_hyperparams)
        return op.kernel_matrix(flatten)


def median_heuristic(x, y):
    # x: [..., n, d]
    # y: [..., m, d]
    # return: []
    n = jnp.shape(x)[-2]
    m = jnp.shape(y)[-2]
    x_expand = jnp.expand_dims(x, -2)
    y_expand = jnp.expand_dims(y, -3)
    pairwise_dist = jnp.sqrt(jnp.sum(jnp.square(x_expand - y_expand), axis=-1))
    k = n * m // 2
    top_k_values = jax.lax.top_k(
        jnp.reshape(pairwise_dist, [-1, n * m]),
        k=k)[0]
    kernel_width = jnp.reshape(top_k_values[:, -1], jnp.shape(x)[:-2])
    return jax.lax.stop_gradient(kernel_width)


class SquareCurlFree(Base):
    def __init__(self, kernel_hyperparams=None, heuristic_hyperparams=median_heuristic):
        super().__init__('curl-free', kernel_hyperparams, heuristic_hyperparams)

    def _gram_derivatives(self, x, y, kernel_hyperparams):
        if kernel_hyperparams is None:
            kernel_width = self.heuristic_hyperparams(x, y)
        else:
            kernel_width = kernel_hyperparams
        x_m = jnp.expand_dims(x, -2)  # [M, 1, d]
        y_m = jnp.expand_dims(y, -3)  # [1, N, d]
        r = x_m - y_m  # [M, N, d]
        norm_rr = jnp.sum(r * r, -1)  # [M, N]
        return self._gram_derivatives_impl(r, norm_rr, kernel_width)

    def _gram_derivatives_impl(self, r, norm_rr, sigma):
        raise NotImplementedError('`_gram_derivatives` not implemented.')

    def kernel_energy(self, x, y, kernel_hyperparams=None, compute_divergence=True):
        d, M, N = jnp.shape(x)[-1], jnp.shape(x)[-2], jnp.shape(y)[-2]
        r, norm_rr, G_1st, G_2nd, _ = self._gram_derivatives(x, y, kernel_hyperparams)

        energy_k = -2. * jnp.expand_dims(G_1st, -1) * r

        if compute_divergence:
            divergence = jnp.array(2 * d).astype(G_1st.dtype) * G_1st \
                         + 4. * norm_rr * G_2nd
            return energy_k, divergence
        return energy_k

    def kernel_operator(self, x, y, kernel_hyperparams=None, compute_divergence=True, return_matr=False,
                        flatten_matr=True):
        d, M, N = jnp.shape(x)[-1], jnp.shape(x)[-2], jnp.shape(y)[-2]  # x: (2,1), y: (2,1), d=1, M=N=2
        r, norm_rr, G_1st, G_2nd, G_3rd = self._gram_derivatives(x, y, kernel_hyperparams)
        G_1st = jnp.expand_dims(G_1st, -1)  # [M, N, 1]
        G_2nd = jnp.expand_dims(G_2nd, -1)  # [M, N, 1]

        if compute_divergence:
            coeff = (jnp.array(d).astype(G_1st.dtype) + 2) * G_2nd \
                    + 2. * jnp.expand_dims(norm_rr * G_3rd, -1)
            divergence = 4. * coeff * r

        def kernel_op(z):
            # z: [N * d, L]
            L = None
            if jnp.shape(z) is not None:
                L = jnp.shape(z)[-1]
            if L is None:
                L = jnp.shape(z)[-1]
            z = jnp.reshape(z, [1, N, d, L])  # [1, N, d, L]
            hat_r = jnp.expand_dims(r, -1)  # [M, N, d, 1]
            dot_rz = jnp.sum(z * hat_r, axis=-2)  # [M, N,    L]
            coeff = -4. * G_2nd * dot_rz  # [M, N,    L]
            ret = jnp.expand_dims(coeff, -2) * hat_r \
                  - 2. * jnp.expand_dims(G_1st, -1) * z
            return jnp.reshape(jnp.sum(ret, axis=-3), [M * d, L])

        def kernel_adjoint_op(z):
            # z: [M * d, L]
            L = None
            if jnp.shape(z) is not None:
                L = jnp.shape(z)[-1]
            if L is None:
                L = jnp.shape(z)[-1]
            z = jnp.reshape(z, [M, 1, d, L])  # [M, 1, d, L]
            hat_r = jnp.expand_dims(r, -1)  # [M, N, d, 1]
            dot_rz = jnp.sum(z * hat_r, axis=-2)  # [M, N,    L]
            coeff = -4. * G_2nd * dot_rz  # [M, N,    L]
            ret = jnp.expand_dims(coeff, -2) * hat_r \
                  - 2. * jnp.expand_dims(G_1st, -1) * z
            return jnp.reshape(jnp.sum(ret, axis=-4), [N * d, L])

        def kernel_mat(flatten):
            Km = jnp.expand_dims(r, -1) * jnp.expand_dims(r, -2)
            K = -2. * jnp.expand_dims(G_1st, -1) * jnp.eye(d) \
                - 4. * jnp.expand_dims(G_2nd, -1) * Km
            if flatten:
                K = jnp.reshape(jnp.transpose(K, [0, 2, 1, 3]), [M * d, N * d])
            return K

        linear_operator = collections.namedtuple(
            "KernelOperator", ["shape", "dtype", "apply", "apply_adjoint", "kernel_matrix"])

        op = linear_operator(
            shape=[M * d, N * d],
            dtype=x.dtype,
            apply=kernel_op,
            apply_adjoint=kernel_adjoint_op,
            kernel_matrix=kernel_mat,
        )

        if not return_matr:
            if compute_divergence:
                return op, divergence
            return op
        else:
            return op.kernel_matrix(flatten_matr)


class CurlFreeIMQ(SquareCurlFree):
    # Inverse Multi-Quadratic curl-free kernel
    def __init__(self, kernel_hyperparams=None, heuristic_hyperparams=median_heuristic):
        super().__init__(kernel_hyperparams, heuristic_hyperparams)

    def _gram_derivatives_impl(self, r, norm_rr, sigma):
        inv_sqr_sigma = 1.0 / jnp.square(sigma)
        imq = jax.lax.rsqrt(1.0 + norm_rr * inv_sqr_sigma)  # [M, N]
        imq_2 = 1.0 / (1.0 + norm_rr * inv_sqr_sigma)
        G_1st = -0.5 * imq_2 * inv_sqr_sigma * imq
        G_2nd = -1.5 * imq_2 * inv_sqr_sigma * G_1st
        G_3rd = -2.5 * imq_2 * inv_sqr_sigma * G_2nd
        return r, norm_rr, G_1st, G_2nd, G_3rd


class CurlFreeIMQp(SquareCurlFree):
    def __init__(self, p=0.5, kernel_hyperparams=None, heuristic_hyperparams=median_heuristic):
        super().__init__(kernel_hyperparams, heuristic_hyperparams)
        self._p = p

    def _gram_derivatives_impl(self, r, norm_rr, sigma):
        inv_sqr_sigma = 1.0 / jnp.square(sigma)
        imq = 1.0 / (1.0 + norm_rr * inv_sqr_sigma)
        imq_p = jax.lax.pow(imq, self._p)  # [M, N]
        G_1st = -(0. + self._p) * imq * inv_sqr_sigma * imq_p
        G_2nd = -(1. + self._p) * imq * inv_sqr_sigma * G_1st
        G_3rd = -(2. + self._p) * imq * inv_sqr_sigma * G_2nd
        return r, norm_rr, G_1st, G_2nd, G_3rd


class CurlFreeGaussian(SquareCurlFree):
    def __init__(self, kernel_hyperparams=None, heuristic_hyperparams=median_heuristic):
        super().__init__(kernel_hyperparams, heuristic_hyperparams)

    def _gram_derivatives_impl(self, r, norm_rr, sigma):
        """""
        Construct the curl-free kernel $-\nabla^2 \psi(\|x - y\|^2)$.
        You need to provide the first, second and third derivatives of $\psi$.
        See eq. (21) and eq. (22). 
        """""
        inv_sqr_sigma = 0.5 / jnp.square(sigma)
        rbf = jnp.exp(-norm_rr * inv_sqr_sigma)
        G_1st = -rbf * inv_sqr_sigma
        G_2nd = -G_1st * inv_sqr_sigma
        G_3rd = -G_2nd * inv_sqr_sigma
        return r, norm_rr, G_1st, G_2nd, G_3rd


class SquareCurlFree(Base):

    def __init__(self, kernel_hyperparams=None, heuristic_hyperparams=median_heuristic):
        super().__init__('curl-free', kernel_hyperparams, heuristic_hyperparams)

    def _gram_derivatives(self, x, y, kernel_hyperparams):
        if kernel_hyperparams is None:
            kernel_width = self.heuristic_hyperparams(x, y)
        else:
            kernel_width = kernel_hyperparams
        x_m = jnp.expand_dims(x, -2)  # [M, 1, d]
        y_m = jnp.expand_dims(y, -3)  # [1, N, d]
        r = x_m - y_m  # [M, N, d]
        norm_rr = jnp.sum(r * r, -1)  # [M, N]
        return self._gram_derivatives_impl(r, norm_rr, kernel_width)

    def _gram_derivatives_impl(self, r, norm_rr, sigma):
        raise NotImplementedError('`_gram_derivatives` not implemented.')

    def kernel_energy(self, x, y, kernel_hyperparams=None, compute_divergence=True):
        d, M, N = jnp.shape(x)[-1], jnp.shape(x)[-2], jnp.shape(y)[-2]
        r, norm_rr, G_1st, G_2nd, _ = self._gram_derivatives(x, y, kernel_hyperparams)

        energy_k = -2. * jnp.expand_dims(G_1st, -1) * r

        if compute_divergence:
            divergence = jnp.array(2 * d).astype(G_1st.dtype) * G_1st \
                    + 4. * norm_rr * G_2nd
            return energy_k, divergence
        return energy_k

    def kernel_operator(self, x, y, kernel_hyperparams=None, compute_divergence=True, return_matr=False, flatten_matr=True):
        d, M, N = jnp.shape(x)[-1], jnp.shape(x)[-2], jnp.shape(y)[-2] # x: (2,1), y: (2,1), d=1, M=N=2
        r, norm_rr, G_1st, G_2nd, G_3rd = self._gram_derivatives(x, y, kernel_hyperparams)
        G_1st = jnp.expand_dims(G_1st, -1)   # [M, N, 1]
        G_2nd = jnp.expand_dims(G_2nd, -1)   # [M, N, 1]

        if compute_divergence:
            coeff = (jnp.array(d).astype(G_1st.dtype) + 2) * G_2nd \
                    + 2. * jnp.expand_dims(norm_rr * G_3rd, -1)
            divergence = 4. * coeff * r

        def kernel_op(z):
            # z: [N * d, L]
            L = None
            if jnp.shape(z) is not None:
                L = jnp.shape(z)[-1]
            if L is None:
                L = jnp.shape(z)[-1]
            z = jnp.reshape(z, [1, N, d, L])            # [1, N, d, L]
            hat_r = jnp.expand_dims(r, -1)              # [M, N, d, 1]
            dot_rz = jnp.sum(z * hat_r, axis=-2) # [M, N,    L]
            coeff = -4. * G_2nd * dot_rz               # [M, N,    L]
            ret = jnp.expand_dims(coeff, -2) * hat_r \
                    - 2. * jnp.expand_dims(G_1st, -1) * z
            return jnp.reshape(jnp.sum(ret, axis=-3), [M * d, L])

        def kernel_adjoint_op(z):
            # z: [M * d, L]
            L = None
            if jnp.shape(z) is not None:
                L = jnp.shape(z)[-1]
            if L is None:
                L = jnp.shape(z)[-1]
            z = jnp.reshape(z, [M, 1, d, L])            # [M, 1, d, L]
            hat_r = jnp.expand_dims(r, -1)              # [M, N, d, 1]
            dot_rz = jnp.sum(z * hat_r, axis=-2) # [M, N,    L]
            coeff = -4. * G_2nd * dot_rz               # [M, N,    L]
            ret = jnp.expand_dims(coeff, -2) * hat_r \
                    - 2. * jnp.expand_dims(G_1st, -1) * z
            return jnp.reshape(jnp.sum(ret, axis=-4), [N * d, L])

        def kernel_mat(flatten):
            Km = jnp.expand_dims(r, -1) * jnp.expand_dims(r, -2)
            K = -2. * jnp.expand_dims(G_1st, -1) * jnp.eye(d) \
                    - 4. * jnp.expand_dims(G_2nd, -1) * Km
            if flatten:
                K = jnp.reshape(jnp.transpose(K, [0, 2, 1, 3]), [M * d, N * d])
            return K

        linear_operator = collections.namedtuple(
            "KernelOperator", ["shape", "dtype", "apply", "apply_adjoint", "kernel_matrix"])

        op = linear_operator(
            shape=[M * d, N * d],
            dtype=x.dtype,
            apply=kernel_op,
            apply_adjoint=kernel_adjoint_op,
            kernel_matrix=kernel_mat,
        )

        if not return_matr:
            if compute_divergence:
                return op, divergence
            return op
        else:
            return op.kernel_matrix(flatten_matr)
