import jax
import haiku as hk
import warnings
import jax.numpy as jnp

from MARS.modules.score_network.score_network import ScoreEstimator
from MARS.modules.data_modules.simulator_base import NNDropoutMetaDataset
from MARS.models.bnn_fsvgd_score_nn import BNN_fSVGD_MARS

jax.config.update("jax_enable_x64", False)


def warn(*args, **kwargs):
    pass


warnings.warn = warn

if __name__ == '__main__':
    seed = 10
    hk_key = hk.PRNGSequence(seed)
    num_train_pts = 5
    num_meas_pts = 10

    # available datasets: physionet_0, physionet_2, argus, swissfel, berkeley_10

    # To fit MARS-GP:
    # sim = GPMetaDataset(dataset="argus", init_seed=next(hk_key), num_input_pts=num_train_pts + num_meas_pts)

    # To fit MARS-BNN:
    sim = NNDropoutMetaDataset(dataset="argus", init_seed=next(hk_key),
                               dropout=0.1,
                               num_pts=num_train_pts + num_meas_pts,
                               start_nn_lr=0.0005,
                               start_nn_batch_size=1,
                               start_nn_num_epochs=50,
                               )

    # initializer score network
    score_net = ScoreEstimator(function_sim=sim,
                               x_dim=sim.input_size,
                               sample_from_gp=True,
                               learning_rate=0.0001,
                               loss_type="exact_w_spectr_norm",
                               n_fn_samples=40,
                               weight_decay=0.)

    temp_list = []

    # train score network
    score_net.train(hk_key, n_iter=10000)

    # perform testing for every meta-test dataset
    for meta_test_data in sim.meta_test_data:
        x_context, y_context, x_test, y_test = meta_test_data
        bnn = BNN_fSVGD_MARS(score_network=score_net,
                             rng_key=next(hk_key),
                             lr=0.01,
                             bandwidth_svgd=.001,
                             data_batch_size=num_train_pts,
                             num_particles=10,
                             likelihood_std=1.5,
                             num_measurement_points=num_meas_pts,
                             clip_value=.1,
                             hidden_activation=jax.nn.elu,

                             num_train_steps=500,
                             )
        eval_stats = bnn.fit(x_context, y_context, x_eval=x_test, y_eval=y_test, num_steps=50)
        temp_list.append(eval_stats)

    total_eval_nll, total_eval_rmse, total_eval_calib = [], [], []
    for tasks in temp_list:
        eval_nll_temp, eval_rmse_temp, eval_calib_temp = [], [], []
        for task_data in tasks:
            eval_nll_temp.append(task_data['eval_nll'])
            eval_rmse_temp.append(task_data['eval_rmse'])
            eval_calib_temp.append(task_data['eval_calib'])
        total_eval_nll.append(eval_nll_temp)
        total_eval_rmse.append(eval_rmse_temp)
        total_eval_calib.append(eval_calib_temp)

    # obtain final results
    nll_mean = jnp.mean(jnp.array(total_eval_nll), axis=0)
    nll_std = jnp.std(jnp.array(total_eval_nll), axis=0)
    rmse_mean = jnp.mean(jnp.array(total_eval_rmse), axis=0)
    rmse_std = jnp.std(jnp.array(total_eval_rmse), axis=0)
    calib_mean = jnp.mean(jnp.array(total_eval_calib), axis=0)
    calib_std = jnp.std(jnp.array(total_eval_calib), axis=0)

    print("NLL mean: ", nll_mean, " and NLL std: ", nll_std)
    print("RMSE mean: ", nll_mean, " and RMSE std: ", nll_std)
    print("Calibration mean: ", calib_mean, " and Calibration std: ", calib_std)
