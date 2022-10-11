import jax
import haiku as hk
import warnings

from MARS.modules.score_network.score_network import ScoreEstimator
from MARS.modules.data_modules.simulator_base import GPMetaDataset
from MARS.models.bnn_fsvgd_score_nn import BNN_fSVGD_MARS

jax.config.update("jax_enable_x64", False)


def warn(*args, **kwargs):
    pass


warnings.warn = warn

if __name__ == '__main__':
    seed = 10
    hk_key = hk.PRNGSequence(seed)
    num_train_pts = 10
    num_meas_pts = 10

    # fit the GP
    sim = GPMetaDataset(dataset="argus", init_seed=next(hk_key), num_pts=num_train_pts + num_meas_pts)
    x_train, y_train, x_test, y_test = sim.meta_test_data[0]

    # initializer score network
    score_net = ScoreEstimator(function_sim=sim,
                               x_dim=sim.input_size,
                               sample_from_gp=True,
                               learning_rate=1e-2,
                               loss_type="exact_w_spectr_norm",
                               n_fn_samples=10)
    # train score network
    score_net.train(hk_key, n_iter=10000)

    # initialize fSVGD
    bnn = BNN_fSVGD_MARS(sim.input_size, score_network=score_net, rng_key=next(hk_key), lr=1e-3,
                              data_batch_size=num_train_pts,
                              num_measurement_points=num_meas_pts,
                              num_train_steps=20000,
                              likelihood_std=0.5,
                              bandwidth_svgd=0.2)

    # train fSVGD
    eval_stats = bnn.fit(x_train, y_train, x_eval=x_test, y_eval=y_test, num_steps=10000)
