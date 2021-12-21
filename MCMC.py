# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse, os, time 
import utils 

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from jax import vmap
import jax.numpy as jnp
import jax.random as random

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import pandas as pd

from constants import *
import os
from sklearn.preprocessing import MinMaxScaler

matplotlib.use("Agg")  # noqa: E402

dir_name = os.path.dirname(os.path.abspath(__file__))

# the non-linearity we use in our neural network
def nonlin(x):
    return jnp.tanh(x)


# a two-layer bayesian neural network with computational flow
# given by D_X => D_H => D_H => D_Y where D_H is the number of
# hidden units. (note we indicate tensor dimensions in the comments)
def model(X, Y, D_H, D_Y, i_prior,o_prior,ps_prior):
    N, D_X = X.shape

    # sample first layer (we put unit normal priors on all weights)
    w1 = numpyro.sample("w1", dist.Normal(jnp.array(i_prior), jnp.ones((D_X, D_H))))
    assert w1.shape == (D_X, D_H)
    z1 = nonlin(jnp.matmul(X, w1))  # <= first layer of activations
    assert z1.shape == (N, D_H)

    # sample second layer
    w2 = numpyro.sample("w2", dist.Normal(jnp.array(o_prior), jnp.ones((D_H, D_H))))
    assert w2.shape == (D_H, D_H)
    z2 = nonlin(jnp.matmul(z1, w2))  # <= second layer of activations
    assert z2.shape == (N, D_H)

    # sample final layer of weights and neural network output
    w3 = numpyro.sample("w3", dist.Normal(jnp.array(ps_prior), jnp.ones((D_H, D_Y))))
    assert w3.shape == (D_H, D_Y)
    z3 = jnp.matmul(z2, w3)  # <= output of the neural network
    assert z3.shape == (N, D_Y)

    if Y is not None:
        assert z3.shape == Y.shape

    # we put a prior on the observation noise
    prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    sigma_obs = 1.0 / jnp.sqrt(prec_obs)

    # observe data
    with numpyro.plate("data", N):
        # note we use to_event(1) because each observation has shape (1,)
        r = numpyro.sample("Y", dist.Normal(z3, sigma_obs).to_event(1), obs=Y)


# helper function for HMC inference
def run_inference(model, config, rng_key, X, Y, D_H, D_Y, i_prior,o_prior, ps_prior):
    start = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(kernel,num_warmup=config['num_warmup'],num_samples=config['num_samples'],
        num_chains=config['num_chains'], progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,)
    mcmc.run(rng_key, X, Y, D_H, D_Y, i_prior,o_prior, ps_prior)
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()

# helper function for prediction
def predict(model, rng_key, samples, X, D_H, D_Y, i_prior,o_prior, ps_prior):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = handlers.trace(model).get_trace(X=X, Y=None, D_H=D_H, D_Y=D_Y,i_prior=i_prior,o_prior=o_prior,ps_prior=ps_prior)
    return model_trace["Y"]["value"]

def run_sampler(config, data, N, i_prior, o_prior, ps_prior):

    numpyro.set_platform(config['mcmc']['device'])
    numpyro.set_host_device_count(config['mcmc']['num_chains'])

    D_X, D_H = 20, 50 

    values = data
    D_Y = 18
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    reframed = utils.series_to_supervised(scaled, 1, 1)
    reframed.drop(reframed.columns[[20, 21]], axis=1, inplace=True)

    train_X, train_y, test_X, test_y = utils.prep_data_mcmc(reframed, 18)

    # do inference
    rng_key, rng_key_predict = random.split(random.PRNGKey(0))
    samples = run_inference(model, config['mcmc'], rng_key, train_X, train_y, D_H, D_Y, i_prior,o_prior, ps_prior)

    # predict Y_test at inputs X_test
    vmap_args = (
        samples,
        random.split(rng_key_predict, config['mcmc']['num_samples'] * config['mcmc']['num_chains']),
    )
    predictions = vmap(
        lambda samples, rng_key: predict(model, rng_key, samples, test_X, D_H, D_Y, i_prior,o_prior, ps_prior)
    )(*vmap_args)

    mean_prediction = jnp.mean(predictions, axis=0)

    with open("results/mean_prediction.json", "w") as f:
        import json
        json.dump(np.array(mean_prediction).tolist(), f)

    with open("results/groundTruth.json", "w") as f:
        json.dump(np.array(test_y).tolist(), f)


    prefix = "{}data_{}hidden_".format(N, D_H)
    # Plot trajectory comparison figure
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

    # plot ground truth
    ax.plot(test_y[:,-7], test_y[:,-6], "kx", label='ground truth')

    # plot prediction
    ax.plot(mean_prediction[:,-7], mean_prediction[:,-6], '.', label='prediction', color='firebrick')

    ax.set(xlabel='X position', ylabel='Y position', title="BNN Ground Truth vs Prediction")
    ax.legend()

    plt.savefig("{}/figures/BNN_comparison.png".format(dir_name))
    plt.cla()

    return samples, predictions, mean_prediction
