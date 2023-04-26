# MARS: Meta-Learning as Score Matching in the Function Space

This is a repository provides source code for MARS, introduced in the
paper [MARS: Meta-Learning as Score Matching in the Function Space](www.arxiv.com).

## Dependencies

To install the minimal dependencies needed to use the algorithms, run in the
main directory of this repository

```commandline
pip install .
```

Alternatively, you can install the required packages using the following
command:

```commandline
pip install -r requirements.txt
```

## Usage

The following code snippet presents the core functionality of this repository. Furthermore, run_demo.py provides a short example how to run MARS-BNN/MARS-GP with the Sinusoidal meta-learning environment, and run_interpolation.py file provides a concise example on how the training dynamics of score estimator network. run_baseline.py can be used for benchmarking and hyperparameter tuning.


```python
from MARS.modules.score_network.score_network import ScoreEstimator
from MARS.modules.data_modules.simulator_base import GPMetaDataset
from MARS.models.bnn_fsvgd_score_nn import BNN_fSVGD_MARS

# fit the GP
sim = GPMetaDataset(dataset="sin_20", num_input_pts=8)
x_train, y_train, x_test, y_test = sim.meta_test_data[0]

# initialize score network
score_net = ScoreEstimator(function_sim=sim,
                           sample_from_gp=True,
                           loss_type="exact_w_spectr_norm")
# train score network
score_net.train(n_iter=10000)

# initialize fSVGD
bnn = BNN_fSVGD_MARS(score_network=score_net, num_train_steps=20000)

# train fSVGD
eval_stats = bnn.fit(x_train, y_train, x_eval=x_test, y_eval=y_test, num_steps=10000)

```

Finally, demo_MARS.ipynb contains a detailed demo code with plots and discussions regarding the overall MARS framework, whereas demo_MARS_interp_1d_exp.ipynb contains plots and discussions regarding the score estimation network. These two notebooks are probably the best point to familiarize yourself with the code.

## Loading the meta-learning datasets / environments
The meta-learning regression environments that were used in the paper can be loaded using
 the ```provide_data``` function:
```python
from MARS.modules.data_modules.regression_datasets import provide_data
meta_train_data, meta_valid_data, meta_test_data = provide_data(dataset=DATASET_NAME)
```
Once you call provide_data, the necessary datasets are automatically downloaded. The following table maps the environment name in the paper to the the DATASET_NAME which needs to be used in the code:

| Name in the paper | DATASET_NAME |
|-------------------|:------------:|
| Sinosoids         |    sin_20    | 
| Swissfel          |   swissfel   |
| Physionet-GCS     | physionet_0  | 
| Physionet-HCT     | physionet_2  | 
| Berkeley-Sensor   |   berkeley   | 
| Argus-Control     |    argus     | 



## Citing
If you use the MARS implementation or the meta-learning environments in your research, please cite it as follows:

```
@article{pavasovic2022mars,
  title={MARS: Meta-learning as score matching in the function space},
  author={Pavasovic, Krunoslav Lehman and Rothfuss, Jonas and Krause, Andreas},
  journal={arXiv preprint arXiv:2210.13319},
  year={2022}
}
```




