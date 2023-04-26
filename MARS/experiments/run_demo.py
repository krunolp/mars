import jax
import jax.numpy as jnp
import haiku as hk
import warnings

from MARS.modules.score_network.score_network import ScoreEstimator
from MARS.modules.data_modules.simulator_base import GPMetaDatasetExample
from MARS.models.bnn_fsvgd_score_nn import BNN_fSVGD_MARS

jax.config.update("jax_enable_x64", False)


def warn(*args, **kwargs):
    pass


warnings.warn = warn

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    seed = 1234
    hk_key = hk.PRNGSequence(seed)

    # fit the GP
    sim = GPMetaDatasetExample(dataset="gp_sin_20", init_seed=next(hk_key))
    x_train, y_train, x_test, y_test = sim.example_data

    # initializer score network
    score_net = ScoreEstimator(function_sim=sim,
                               x_dim=sim.input_size,
                               sample_from_gp=True,
                               learning_rate=0.001,
                               loss_type="exact_w_spectr_norm",
                               n_fn_samples=20)
    # train score network
    score_net.train(hk_key, n_iter=250)


    # initialize fSVGD
    bnn = BNN_fSVGD_MARS(score_network=score_net,
                         rng_key=next(hk_key),
                         lr=0.0075,
                         num_train_steps=20000,
                         likelihood_std=1.25,
                         bandwidth_svgd=0.125)

    # train fSVGD
    for i in range(25):
        eval_stats = bnn.fit(x_train, y_train, x_eval=x_test, y_eval=y_test, num_steps=100)
        bnn.plot_1d(x_train, y_train, domain_l=jnp.array([-4]), domain_u=jnp.array([4]),
                    title='fSVGD predictions at iteration: ' + str(i * 50),
                    plot_data=(jnp.linspace(-4, 4, 50), sim.dataset.mean_fn(np.linspace(-4, 4, 50))))
