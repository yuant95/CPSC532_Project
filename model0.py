# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Bayesian Neural Network
================================
We demonstrate how to use NUTS to do inference on a simple (small)
Bayesian neural network with two hidden layers.
.. image:: ../_static/img/examples/bnn.png
    :align: center
"""

import argparse
import os
import time

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

scalerTest = MinMaxScaler()
# the non-linearity we use in our neural network
def nonlin(x):
    return jnp.tanh(x)


# a two-layer bayesian neural network with computational flow
# given by D_X => D_H => D_H => D_Y where D_H is the number of
# hidden units. (note we indicate tensor dimensions in the comments)
def model(X, Y, D_H, D_Y):
    N, D_X = X.shape

    # sample first layer (we put unit normal priors on all weights)
    w1 = numpyro.sample("w1", dist.Normal(jnp.zeros((D_X, D_H)), jnp.ones((D_X, D_H))))
    assert w1.shape == (D_X, D_H)
    z1 = nonlin(jnp.matmul(X, w1))  # <= first layer of activations
    assert z1.shape == (N, D_H)

    # sample second layer
    w2 = numpyro.sample("w2", dist.Normal(jnp.zeros((D_H, D_H)), jnp.ones((D_H, D_H))))
    assert w2.shape == (D_H, D_H)
    z2 = nonlin(jnp.matmul(z1, w2))  # <= second layer of activations
    assert z2.shape == (N, D_H)

    # sample final layer of weights and neural network output
    w3 = numpyro.sample("w3", dist.Normal(jnp.zeros((D_H, D_Y)), jnp.ones((D_H, D_Y))))
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
def run_inference(model, args, rng_key, X, Y, D_H, D_Y):
    start = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(rng_key, X, Y, D_H, D_Y)
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()

# helper function for prediction
def predict(model, rng_key, samples, X, D_H, D_Y):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = handlers.trace(model).get_trace(X=X, Y=None, D_H=D_H, D_Y = D_Y)
    return model_trace["Y"]["value"]

def predict_seires(model, rng_key, samples, X, D_H, D_Y):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    X0 = X[0]
    X0 = X0[np.newaxis, :]
    model_trace0 = handlers.trace(model).get_trace(X=X0, Y=None, D_H=D_H, D_Y = D_Y)
    Y_t = jnp.mean(model_trace0["Y"]["value"].val, axis=0)
    ret = Y_t

    for i in range(1, len(X)):
        control = X[i, 4:6]
        X_t = jnp.append(Y_t, control)[np.newaxis, :]
        model_trace = handlers.trace(model).get_trace(X=X_t, Y=None, D_H=D_H, D_Y = D_Y)
        Y_t = jnp.mean(model_trace["Y"]["value"].val, axis=0)
        ret = jnp.concatenate([ret, Y_t])


    return ret

# Note: Current it's using test 1 as training, and test 2 as test data
# Only using ROBOT_POSE_DATA_ITEMS and CONTROLLER_DATA_ITEMS
def read_data(N=100, N_test=10):
    # To get rid of some noisy set up phase data
    offset = 200
    N += offset
    finalizedDataPath = os.path.join(DRIVE_TEST_1_FOLDER, "finalizedData2.csv")
    testDataPath = os.path.join(DRIVE_TEST_2_FOLDER, "finalizedData.csv")
    data = pd.read_csv(finalizedDataPath)
    testData = pd.read_csv(testDataPath)

    # Normalize the data
    scaler = MinMaxScaler()

    data[data.columns] = scaler.fit_transform(data)
    testData[testData.columns] = scalerTest.fit_transform(testData)

    X = jnp.array(data.loc[offset:N-1, ROBOT_POSE_DATA_ITEMS+CONTROLLER_DATA_ITEMS].to_numpy())
    Y = jnp.array(data.loc[offset+1:N,ROBOT_POSE_DATA_ITEMS].to_numpy())

    X_test = jnp.array(testData.loc[offset:offset+N_test, ROBOT_POSE_DATA_ITEMS+CONTROLLER_DATA_ITEMS].to_numpy())
    Y_test_truth = jnp.array(testData.loc[offset+1:offset+N_test+1, ROBOT_POSE_DATA_ITEMS].to_numpy())

    return X, Y, X_test, Y_test_truth

def plot_result(X,Y,X_test,predictions, xDim, yDim, prefix = ""):
    xName = (ROBOT_POSE_DATA_ITEMS+CONTROLLER_DATA_ITEMS)[xDim]
    yName = ROBOT_POSE_DATA_ITEMS[yDim]

    mean_prediction = jnp.mean(predictions, axis=0)
    percentiles = np.percentile(predictions, [5.0, 95.0], axis=0)

    # make plots
    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    # plot training data
    ax.plot(X[:, xDim], Y[:, yDim], "kx")
    # plot 90% confidence level of predictions
    # ax.fill_between(
    #     X_test[:, xDim], percentiles[0, :, yDim], percentiles[1, :, yDim], color="lightblue"
    # )
    # plot mean prediction
    ax.plot(X_test[:, xDim], mean_prediction[:,yDim], "bo")#, ls="solid", lw=2.0)
    ax.set(xlabel="X_{}".format(xName), ylabel="Y_{}".format(yName), title="Mean predictions with 90% CI")

    plt.savefig("{}_bnn_plot_x{}_y{}.png".format(prefix, xName, yName))

    plt.close(fig)
    plt.clf()

    # No prediction
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    # plot training data
    ax.plot(X[:, xDim], Y[:, yDim], "kx")

    ax.set(xlabel="X_{}".format(xName), ylabel="Y_{}".format(yName), title="{} Mean predictions with 90% CI".format(prefix))

    plt.savefig("{}_data_x{}_y{}.png".format(prefix, xName, yName))
    plt.close(fig)

def main(args):
    D_Y = 4
    N_test = 1000
    N, D_X, D_H = args.num_data, D_Y+2, args.num_hidden
    
    X, Y, X_test, Y_test_truth = read_data(N, N_test)

    # do inference
    rng_key, rng_key_predict = random.split(random.PRNGKey(0))
    samples = run_inference(model, args, rng_key, X, Y, D_H, D_Y)

    # predict Y_test at inputs X_test
    vmap_args = (
        samples,
        random.split(rng_key_predict, args.num_samples * args.num_chains),
    )
    predictions = vmap(
        lambda samples, rng_key: predict(model, rng_key, samples, X_test, D_H, D_Y)
    )(*vmap_args)

    mean_prediction = jnp.mean(predictions, axis=0)

    # series_predictions = vmap(
    #     lambda samples, rng_key: predict_seires(model, rng_key, samples, X_test, D_H, D_Y)
    # )(*vmap_args)

    # series_predictions = jnp.mean(series_predictions, axis=0)
    mean_prediction = jnp.mean(predictions, axis=0)

    with open("mean_prediction.json", "w") as f:
        import json
        json.dump(np.array(mean_prediction).tolist(), f)

    with open("groundTruth.json", "w") as f:
        json.dump(np.array(Y_test_truth).tolist(), f)

    for i in range(D_X):
        for j in range(D_Y):
            prefix = "{}data_{}hidden_".format(N, D_H)
            plot_result(X,Y,X_test,predictions,i, j, prefix)

    # Plot trajectory comparison figure
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    # plot ground truth
    ax.plot(Y_test_truth[:,-4], Y_test_truth[:,-3], "kx")

    # plot prediction
    ax.plot(mean_prediction[:,-4], mean_prediction[:,-3], "bo")

    ax.set(xlabel="X_{}".format("pose.position.x"), ylabel="Y_{}".format("pose.position.y"), title="Mean predictions with 90% CI")

    plt.savefig("{}_comparison.png".format(prefix))


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.8.0")
    parser = argparse.ArgumentParser(description="Bayesian neural network example")
    parser.add_argument("-n", "--num-samples", nargs="?", default=2000, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=1000, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument("--num-data", nargs="?", default=3000, type=int)
    parser.add_argument("--num-hidden", nargs="?", default=7, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)