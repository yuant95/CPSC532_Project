# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
An implementation of a Deep Markov Model in Pyro based on reference [1].
This is essentially the DKS variant outlined in the paper. The primary difference
between this implementation and theirs is that in our version any KL divergence terms
in the ELBO are estimated via sampling, while they make use of the analytic formulae.
We also illustrate the use of normalizing flows in the variational distribution (in which
case analytic formulae for the KL divergences are in any case unavailable).
Reference:
[1] Structured Inference Networks for Nonlinear State Space Models [arXiv:1609.09869]
    Rahul G. Krishnan, Uri Shalit, David Sontag
"""

import argparse
import logging
import time
from os.path import exists

import numpy as np
import torch
import torch.nn as nn

import pyro
import pyro.contrib.examples.polyphonic_data_loader as poly
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import TransformedDistribution
from pyro.distributions.transforms import affine_autoregressive
from pyro.infer import (
    SVI,
    JitTrace_ELBO,
    Trace_ELBO,
    TraceEnum_ELBO,
    TraceTMC_ELBO,
    config_enumerate,
)
from pyro.optim import ClippedAdam
import BNN
import utils
from constants import *
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class Emitter(nn.Module):
    """
    Parameterizes the bernoulli observation likelihood `p(x_t | z_t)`
    """

    def __init__(self, input_dim, z_dim, emission_dim):
        # super().__init__()
        # # initialize the three linear transformations used in the neural network
        # self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        # self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        # self.lin_hidden_to_input = nn.Linear(emission_dim, input_dim)
        # # initialize the two non-linearities used in the neural network
        # self.relu = nn.ReLU()

        super().__init__()
        # initialize the six linear transformations used in the neural network
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_gate_hidden_to_z = nn.Linear(emission_dim, z_dim)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(emission_dim, z_dim)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)
        # modify the default initialization of lin_z_to_loc
        # so that it's starts out as the identity function
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)
        # initialize the three non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    # def forward(self, z_t):
    #     """
    #     Given the latent z at a particular time step t we return the vector of
    #     probabilities `ps` that parameterizes the bernoulli distribution `p(x_t|z_t)`
    #     """
    #     h1 = self.relu(self.lin_z_to_hidden(z_t))
    #     h2 = self.relu(self.lin_hidden_to_hidden(h1))
    #     ps = torch.sigmoid(self.lin_hidden_to_input(h2))
    #     return ps

    def forward(self, z_t):
        """
        Given the latent z at a particular time step t we return the vector of
        probabilities `ps` that parameterizes the bernoulli distribution `p(x_t|z_t)`
        """
        # compute the gating function
        _gate = self.relu(self.lin_gate_z_to_hidden(z_t))
        gate = torch.sigmoid(self.lin_gate_hidden_to_z(_gate))
        # compute the 'proposed mean'
        _proposed_mean = self.relu(self.lin_proposed_mean_z_to_hidden(z_t))
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)
        # assemble the actual mean used to sample z_t, which mixes a linear transformation
        # of z_{t-1} with the proposed mean modulated by the gating function
        loc = (1 - gate) * self.lin_z_to_loc(z_t) + gate * proposed_mean
        # compute the scale used to sample z_t, using the proposed mean from
        # above as input the softplus ensures that scale is positive
        scale = self.softplus(self.lin_sig(self.relu(proposed_mean)))
        # return loc, scale which can be fed into Normal
        return loc, scale


class GatedTransition(nn.Module):
    """
    Parameterizes the gaussian latent transition probability `p(z_t | z_{t-1})`
    See section 5 in the reference for comparison.
    """

    def __init__(self, z_dim, transition_dim):
        super().__init__()
        # initialize the six linear transformations used in the neural network
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)
        # modify the default initialization of lin_z_to_loc
        # so that it's starts out as the identity function
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)
        # initialize the three non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1):
        """
        Given the latent `z_{t-1}` corresponding to the time step t-1
        we return the mean and scale vectors that parameterize the
        (diagonal) gaussian distribution `p(z_t | z_{t-1})`
        """
        # compute the gating function
        _gate = self.relu(self.lin_gate_z_to_hidden(z_t_1))
        gate = torch.sigmoid(self.lin_gate_hidden_to_z(_gate))
        # compute the 'proposed mean'
        _proposed_mean = self.relu(self.lin_proposed_mean_z_to_hidden(z_t_1))
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)
        # assemble the actual mean used to sample z_t, which mixes a linear transformation
        # of z_{t-1} with the proposed mean modulated by the gating function
        loc = (1 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean
        # compute the scale used to sample z_t, using the proposed mean from
        # above as input the softplus ensures that scale is positive
        scale = self.softplus(self.lin_sig(self.relu(proposed_mean)))
        # return loc, scale which can be fed into Normal
        return loc, scale


class Combiner(nn.Module):
    """
    Parameterizes `q(z_t | z_{t-1}, x_{t:T})`, which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on `x_{t:T}` is
    through the hidden state of the RNN (see the PyTorch module `rnn` below)
    """

    def __init__(self, z_dim, rnn_dim):
        super().__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)
        # initialize the two non-linearities used in the neural network
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, h_rnn):
        """
        Given the latent z at at a particular time step t-1 as well as the hidden
        state of the RNN `h(x_{t:T})` we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution `q(z_t | z_{t-1}, x_{t:T})`
        """
        # combine the rnn hidden state with a transformed version of z_t_1
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)
        # use the combined hidden state to compute the mean used to sample z_t
        loc = self.lin_hidden_to_loc(h_combined)
        # use the combined hidden state to compute the scale used to sample z_t
        scale = self.softplus(self.lin_hidden_to_scale(h_combined))
        # return loc, scale which can be fed into Normal
        return loc, scale


class DMM(nn.Module):
    """
    This PyTorch Module encapsulates the model as well as the
    variational distribution (the guide) for the Deep Markov Model
    """

    def __init__(
        self,
        input_dim=12,
        z_dim=12,
        emission_dim=12,
        transition_dim=6,
        rnn_dim=20,
        num_layers=1,
        rnn_dropout_rate=0.0,
        num_iafs=0,
        iaf_dim=50,
        use_cuda=False,
    ):
        super().__init__()
        # instantiate PyTorch modules used in the model and guide below
        self.emitter = Emitter(input_dim, z_dim, emission_dim)
        self.trans = GatedTransition(z_dim, transition_dim)
        self.combiner = Combiner(z_dim, rnn_dim)
        # dropout just takes effect on inner layers of rnn
        rnn_dropout_rate = 0.0 if num_layers == 1 else rnn_dropout_rate
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=rnn_dim,
            nonlinearity="relu",
            batch_first=True,
            bidirectional=False,
            num_layers=num_layers,
            dropout=rnn_dropout_rate,
        )

        # if we're using normalizing flows, instantiate those too
        self.iafs = [
            affine_autoregressive(z_dim, hidden_dims=[iaf_dim]) for _ in range(num_iafs)
        ]
        self.iafs_modules = nn.ModuleList(self.iafs)

        # define a (trainable) parameters z_0 and z_q_0 that help define the probability
        # distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim))
        # define a (trainable) parameter for the initial hidden state of the rnn
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))

        self.use_cuda = use_cuda
        # if on gpu cuda-ize all PyTorch (sub)modules
        if use_cuda:
            self.cuda()

    # the model p(x_{1:T} | z_{1:T}) p(z_{1:T})
    def model(
        self,
        mini_batch,
        mini_batch_reversed,
        mini_batch_mask,
        mini_batch_seq_lengths,
        annealing_factor=1.0,
    ):

        # this is the number of time steps we need to process in the mini-batch
        T_max = mini_batch.size(1)

        # register all PyTorch (sub)modules with pyro
        # this needs to happen in both the model and guide
        pyro.module("dmm", self)

        # set z_prev = z_0 to setup the recursive conditioning in p(z_t | z_{t-1})
        z_prev = self.z_0.expand(mini_batch.size(0), self.z_0.size(0))

        # we enclose all the sample statements in the model in a plate.
        # this marks that each datapoint is conditionally independent of the others
        with pyro.plate("z_minibatch", len(mini_batch)):
            # sample the latents z and observed x's one time step at a time
            # we wrap this loop in pyro.markov so that TraceEnum_ELBO can use multiple samples from the guide at each z
            for t in pyro.markov(range(1, T_max + 1)):
                # the next chunk of code samples z_t ~ p(z_t | z_{t-1})
                # note that (both here and elsewhere) we use poutine.scale to take care
                # of KL annealing. we use the mask() method to deal with raggedness
                # in the observed data (i.e. different sequences in the mini-batch
                # have different lengths)

                # first compute the parameters of the diagonal gaussian distribution p(z_t | z_{t-1})
                z_loc, z_scale = self.trans(z_prev)

                # then sample z_t according to dist.Normal(z_loc, z_scale)
                # note that we use the reshape method so that the univariate Normal distribution
                # is treated as a multivariate Normal distribution with a diagonal covariance.
                with poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample(
                        "z_%d" % t,
                        dist.Normal(z_loc, z_scale)
                        .mask(mini_batch_mask[:, t - 1 : t])
                        .to_event(1),
                    )

                # compute the probabilities that parameterize the bernoulli likelihood
                emission_probs_t_loc,  emission_probs_t_scale = self.emitter(z_t)
                # emission_probs_t = self.emitter(z_t)
                # the next statement instructs pyro to observe x_t according to the
                # bernoulli distribution p(x_t|z_t)
                pyro.sample(
                    "obs_x_%d" % t,
                    dist.Normal(emission_probs_t_loc, emission_probs_t_scale)
                    .mask(mini_batch_mask[:, t - 1 : t])
                    .to_event(1),
                    obs=mini_batch[:, t - 1, :],
                )
                # the latent sampled at this time step will be conditioned upon
                # in the next time step so keep track of it
                z_prev = z_t

    # the guide q(z_{1:T} | x_{1:T}) (i.e. the variational distribution)
    def guide(
        self,
        mini_batch,
        mini_batch_reversed,
        mini_batch_mask,
        mini_batch_seq_lengths,
        annealing_factor=1.0,
    ):

        # this is the number of time steps we need to process in the mini-batch
        T_max = mini_batch.size(1)
        # register all PyTorch (sub)modules with pyro
        pyro.module("dmm", self)

        # if on gpu we need the fully broadcast view of the rnn initial state
        # to be in contiguous gpu memory
        h_0_contig = self.h_0.expand(
            1, mini_batch.size(0), self.rnn.hidden_size
        ).contiguous()
        # push the observed x's through the rnn;
        # rnn_output contains the hidden state at each time step
        rnn_output, _ = self.rnn(mini_batch_reversed.float(), h_0_contig)
        # reverse the time-ordering in the hidden state and un-pack it
        rnn_output = poly.pad_and_reverse(rnn_output, mini_batch_seq_lengths.int())
        # set z_prev = z_q_0 to setup the recursive conditioning in q(z_t |...)
        z_prev = self.z_q_0.expand(mini_batch.size(0), self.z_q_0.size(0))

        # we enclose all the sample statements in the guide in a plate.
        # this marks that each datapoint is conditionally independent of the others.
        with pyro.plate("z_minibatch", len(mini_batch)):
            # sample the latents z one time step at a time
            # we wrap this loop in pyro.markov so that TraceEnum_ELBO can use multiple samples from the guide at each z
            for t in pyro.markov(range(1, T_max + 1)):
                # the next two lines assemble the distribution q(z_t | z_{t-1}, x_{t:T})
                z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])

                # if we are using normalizing flows, we apply the sequence of transformations
                # parameterized by self.iafs to the base distribution defined in the previous line
                # to yield a transformed distribution that we use for q(z_t|...)
                if len(self.iafs) > 0:
                    z_dist = TransformedDistribution(
                        dist.Normal(z_loc, z_scale), self.iafs
                    )
                    assert z_dist.event_shape == (self.z_q_0.size(0),)
                    assert z_dist.batch_shape[-1:] == (len(mini_batch),)
                else:
                    z_dist = dist.Normal(z_loc, z_scale)
                    assert z_dist.event_shape == ()
                    assert z_dist.batch_shape[-2:] == (
                        len(mini_batch),
                        self.z_q_0.size(0),
                    )

                # sample z_t from the distribution z_dist
                with pyro.poutine.scale(scale=annealing_factor):
                    if len(self.iafs) > 0:
                        # in output of normalizing flow, all dimensions are correlated (event shape is not empty)
                        z_t = pyro.sample(
                            "z_%d" % t, z_dist.mask(mini_batch_mask[:, t - 1])
                        )
                    else:
                        # when no normalizing flow used, ".to_event(1)" indicates latent dimensions are independent
                        z_t = pyro.sample(
                            "z_%d" % t,
                            z_dist.mask(mini_batch_mask[:, t - 1 : t]).to_event(1),
                        )
                # the latent sampled at this time step will be conditioned upon in the next time step
                # so keep track of it
                z_prev = z_t


def read_data(N, N_test, seqLength):
    # To get rid of some noisy set up phase data
    offset = 200
    N += offset
    finalizedDataPath = os.path.join(DRIVE_TEST_1_FOLDER, "finalizedData.csv")
    testDataPath = os.path.join(DRIVE_TEST_2_FOLDER, "finalizedData.csv")
    valDataPath = os.path.join(DRIVE_TEST_3_FOLDER, "finalizedData.csv")

    data = pd.read_csv(finalizedDataPath)
    testData = pd.read_csv(testDataPath)
    valData = pd.read_csv(valDataPath)

    # Normalize the data
    scaler = MinMaxScaler()

    data[data.columns] = scaler.fit_transform(data)
    testData[testData.columns] = scaler.fit_transform(testData)
    valData[valData.columns] = scaler.fit_transform(valData)

    X = data.loc[offset:N-1, STATE_ITEMS+CONTROLLER_DATA_ITEMS].to_numpy()
    X_test = testData.loc[offset+N_test:offset+N_test+N_test-1, STATE_ITEMS+CONTROLLER_DATA_ITEMS].to_numpy()
    X_val = testData.loc[offset:offset+N_test-1, STATE_ITEMS+CONTROLLER_DATA_ITEMS].to_numpy()

    X = X.reshape((int(X.shape[0]/seqLength), seqLength, X.shape[1]))
    X_test = X_test.reshape((int(X_test.shape[0]/seqLength), seqLength, X_test.shape[1]))
    X_val = X_val.reshape((int(X_val.shape[0]/seqLength), seqLength, X_val.shape[1]))

    return X, X_val, X_test

# setup, training, and evaluation
def main(args):
    # setup logging
    logging.basicConfig(
        level=logging.DEBUG, format="%(message)s", filename=args.log, filemode="w"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)
    logging.info(args) 

    # data = poly.load_data(poly.JSB_CHORALES)
    # training_seq_lengths = data["train"]["sequence_lengths"]
    # training_data_sequences = data["train"]["sequences"]
    # test_seq_lengths = data["test"]["sequence_lengths"]
    # test_data_sequences = data["test"]["sequences"]
    # val_seq_lengths = data["valid"]["sequence_lengths"]
    # val_data_sequences = data["valid"]["sequences"]
    # N_train_data = len(training_seq_lengths)
    # N_train_time_slices = float(torch.sum(training_seq_lengths))
    # N_mini_batches = int(
    #     N_train_data / args.mini_batch_size
    #     + int(N_train_data % args.mini_batch_size > 0)
    # )
    seqLength = 2
    train_X, val_X, test_X = read_data(2000, 1000, seqLength)
    train_X = val_X
    config = utils.load_exp_config('test.json')

    training_seq_lengths = seqLength*torch.ones(train_X.shape[0])    
    training_data_sequences = torch.tensor(train_X)        
    test_seq_lengths = seqLength*torch.ones(test_X.shape[0]) 
    test_data_sequences = torch.tensor(test_X)
    val_seq_lengths = seqLength*torch.ones(val_X.shape[0]) 
    val_data_sequences = torch.tensor(val_X)
    N_train_data = len(training_seq_lengths)
    N_train_time_slices = float(torch.sum(training_seq_lengths))
    N_mini_batches = int(N_train_data / config['config']['mini_batch_size'] + int(N_train_data % config['config']['mini_batch_size'] > 0))

    logging.info(
        "N_train_data: %d     avg. training seq. length: %.2f    N_mini_batches: %d"
        % (N_train_data, training_seq_lengths.float().mean(), N_mini_batches)
    )

    # how often we do validation/test evaluation during training
    val_test_frequency = 50
    # the number of samples we use to do the evaluation
    n_eval_samples = 1

    
    # package repeated copies of val/test data for faster evaluation
    # (i.e. set us up for vectorization)
    def rep(x):
        rep_shape = torch.Size([x.size(0) * n_eval_samples]) + x.size()[1:]
        repeat_dims = [1] * len(x.size())
        repeat_dims[0] = n_eval_samples
        return (
            x.repeat(repeat_dims)
            .reshape(n_eval_samples, -1)
            .transpose(1, 0)
            .reshape(rep_shape)
        )

        # package repeated copies of val/test data for faster evaluation


    (val_batch,val_batch_reversed,val_batch_mask,val_seq_lengths,) = utils.get_mini_batch(
        torch.arange(n_eval_samples * val_data_sequences.shape[0], dtype=torch.int64), rep(val_data_sequences), val_seq_lengths)
    (test_batch,test_batch_reversed,test_batch_mask,test_seq_lengths) = utils.get_mini_batch(
        torch.arange(n_eval_samples * test_data_sequences.shape[0],dtype=torch.int64), rep(test_data_sequences), test_seq_lengths)
    # get the validation/test data ready for the dmm: pack into sequences, etc.
    val_seq_lengths = rep(val_seq_lengths)
    test_seq_lengths = rep(test_seq_lengths)
    # (
    #     val_batch,
    #     val_batch_reversed,
    #     val_batch_mask,
    #     val_seq_lengths,
    # ) = poly.get_mini_batch(
    #     torch.arange(n_eval_samples * val_data_sequences.shape[0]),
    #     rep(val_data_sequences),
    #     val_seq_lengths,
    #     cuda=args.cuda,
    # )
    # (
    #     test_batch,
    #     test_batch_reversed,
    #     test_batch_mask,
    #     test_seq_lengths,
    # ) = poly.get_mini_batch(
    #     torch.arange(n_eval_samples * test_data_sequences.shape[0]),
    #     rep(test_data_sequences),
    #     test_seq_lengths,
    #     cuda=args.cuda,
    # )

    # instantiate the dmm
    dmm = DMM(
        rnn_dropout_rate=args.rnn_dropout_rate,
        num_iafs=args.num_iafs,
        iaf_dim=args.iaf_dim,
        use_cuda=args.cuda,
    )

    # setup optimizer
    adam_params = {
        "lr": args.learning_rate,
        "betas": (args.beta1, args.beta2),
        "clip_norm": args.clip_norm,
        "lrd": args.lr_decay,
        "weight_decay": args.weight_decay,
    }
    adam = ClippedAdam(adam_params)

    # setup inference algorithm
    if args.tmc:
        if args.jit:
            raise NotImplementedError("no JIT support yet for TMC")
        tmc_loss = TraceTMC_ELBO()
        dmm_guide = config_enumerate(
            dmm.guide,
            default="parallel",
            num_samples=args.tmc_num_samples,
            expand=False,
        )
        svi = SVI(dmm.model, dmm_guide, adam, loss=tmc_loss)
    elif args.tmcelbo:
        if args.jit:
            raise NotImplementedError("no JIT support yet for TMC ELBO")
        elbo = TraceEnum_ELBO()
        dmm_guide = config_enumerate(
            dmm.guide,
            default="parallel",
            num_samples=args.tmc_num_samples,
            expand=False,
        )
        svi = SVI(dmm.model, dmm_guide, adam, loss=elbo)
    else:
        elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
        svi = SVI(dmm.model, dmm.guide, adam, loss=elbo)

    # now we're going to define some functions we need to form the main training loop

    # saves the model and optimizer states to disk
    def save_checkpoint():
        logging.info("saving model to %s..." % args.save_model)
        torch.save(dmm.state_dict(), args.save_model)
        logging.info("saving optimizer states to %s..." % args.save_opt)
        adam.save(args.save_opt)
        logging.info("done saving model and optimizer checkpoints to disk.")

    # loads the model and optimizer states from disk
    def load_checkpoint():
        assert exists(args.load_opt) and exists(
            args.load_model
        ), "--load-model and/or --load-opt misspecified"
        logging.info("loading model from %s..." % args.load_model)
        dmm.load_state_dict(torch.load(args.load_model))
        logging.info("loading optimizer states from %s..." % args.load_opt)
        adam.load(args.load_opt)
        logging.info("done loading model and optimizer states.")

    # prepare a mini-batch and take a gradient step to minimize -elbo
    # prepare a mini-batch and take a gradient step to minimize -elbo
    def process_minibatch(epoch, which_mini_batch, shuffled_indices):
        if config['VRNN']['annealing_epochs'] > 0 and epoch < config['VRNN']['annealing_epochs']:
            # compute the KL annealing factor approriate for the current mini-batch in the current epoch
            min_af = config['VRNN']['minimum_annealing_factor']
            annealing_factor = min_af + (1.0 - min_af) * (
                float(which_mini_batch + epoch * N_mini_batches + 1)
                / float(config['VRNN']['annealing_epochs'] * N_mini_batches))
        else:
            # by default the KL annealing factor is unity
            annealing_factor = 1.0

        # compute which sequences in the training set we should grab
        mini_batch_start = which_mini_batch * config['config']['mini_batch_size']
        mini_batch_end = np.min(
            [(which_mini_batch + 1) * config['config']['mini_batch_size'], N_train_data])
        mini_batch_indices = shuffled_indices[mini_batch_start:mini_batch_end]
        # grab a fully prepped mini-batch using the helper function in the data loader
        (
            mini_batch,
            mini_batch_reversed,
            mini_batch_mask,
            mini_batch_seq_lengths,
        ) = utils.get_mini_batch(
            mini_batch_indices,
            training_data_sequences,
            training_seq_lengths,
            cuda=False,
        )
        # do an actual gradient step
        loss = svi.step(
            mini_batch,
            mini_batch_reversed,
            mini_batch_mask,
            mini_batch_seq_lengths,
            annealing_factor,
        )
        # keep track of the training loss
        return loss

    # helper function for doing evaluation
    def do_evaluation():
        # put the RNN into evaluation mode (i.e. turn off drop-out if applicable)
        dmm.rnn.eval()

        # compute the validation and test loss n_samples many times
        val_nll = svi.evaluate_loss(
            val_batch, val_batch_reversed, val_batch_mask, val_seq_lengths
        ) / float(torch.sum(val_seq_lengths))
        test_nll = svi.evaluate_loss(
            test_batch, test_batch_reversed, test_batch_mask, test_seq_lengths
        ) / float(torch.sum(test_seq_lengths))

        # put the RNN back into training mode (i.e. turn on drop-out if applicable)
        dmm.rnn.train()
        return val_nll, test_nll

    # if checkpoint files provided, load model and optimizer states from disk before we start training
    if args.load_opt != "" and args.load_model != "":
        load_checkpoint()

    #################
    # TRAINING LOOP #
    #################
    times = [time.time()]

    nll = []
    valNll = []
    testNll = []
    for epoch in range(args.num_epochs):
        # if specified, save model and optimizer states to disk every checkpoint_freq epochs
        if args.checkpoint_freq > 0 and epoch > 0 and epoch % args.checkpoint_freq == 0:
            save_checkpoint()

        # accumulator for our estimate of the negative log likelihood (or rather -elbo) for this epoch
        epoch_nll = 0.0
        # prepare mini-batch subsampling indices for this epoch
        shuffled_indices = torch.randperm(N_train_data)

        # process each mini-batch; this is where we take gradient steps
        for which_mini_batch in range(N_mini_batches):
            epoch_nll += process_minibatch(epoch, which_mini_batch, shuffled_indices)

        # report training diagnostics
        times.append(time.time())
        epoch_time = times[-1] - times[-2]
        logging.info(
            "[training epoch %04d]  %.4f \t\t\t\t(dt = %.3f sec)"
            % (epoch, epoch_nll / N_train_time_slices, epoch_time)
        )

        nll.append(epoch_nll / N_train_time_slices)


        # do evaluation on test and validation data and report results
        if val_test_frequency > 0 and epoch > 0 and epoch % val_test_frequency == 0:
            val_nll, test_nll = do_evaluation()
            logging.info(
                "[val/test epoch %04d]  %.4f  %.4f" % (epoch, val_nll, test_nll)
            )
            valNll.append(val_nll)
            testNll.append(test_nll)

    plot(nll, "Train")
    plot(valNll, "Train")
    plot(testNll, "Test")

    nlls = {
        "train": nll,
        "val": valNll,
        "test": testNll,

    }

    with open("nll.json", "w") as f:
        import json
        json.dump(nlls, f)



def plot(data,name):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    # plot ground truth
    ax.plot(data)

    ax.set(xlabel="Steps", ylabel="{} NLL".format(name))

    plt.savefig("dmm_{}_nll.png".format(name))

# parse command-line arguments and execute the main method
if __name__ == "__main__":
    assert pyro.__version__.startswith("1.7.0")

    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument("-n", "--num-epochs", type=int, default=5000)
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.0003)
    parser.add_argument("-b1", "--beta1", type=float, default=0.96)
    parser.add_argument("-b2", "--beta2", type=float, default=0.999)
    parser.add_argument("-cn", "--clip-norm", type=float, default=10.0)
    parser.add_argument("-lrd", "--lr-decay", type=float, default=0.99996)
    parser.add_argument("-wd", "--weight-decay", type=float, default=2.0)
    parser.add_argument("-mbs", "--mini-batch-size", type=int, default=20)
    parser.add_argument("-ae", "--annealing-epochs", type=int, default=1000)
    parser.add_argument("-maf", "--minimum-annealing-factor", type=float, default=0.2)
    parser.add_argument("-rdr", "--rnn-dropout-rate", type=float, default=0.1)
    parser.add_argument("-iafs", "--num-iafs", type=int, default=0)
    parser.add_argument("-id", "--iaf-dim", type=int, default=100)
    parser.add_argument("-cf", "--checkpoint-freq", type=int, default=0)
    parser.add_argument("-lopt", "--load-opt", type=str, default="")
    parser.add_argument("-lmod", "--load-model", type=str, default="")
    parser.add_argument("-sopt", "--save-opt", type=str, default="")
    parser.add_argument("-smod", "--save-model", type=str, default="")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--tmc", action="store_true")
    parser.add_argument("--tmcelbo", action="store_true")
    parser.add_argument("--tmc-num-samples", default=10, type=int)
    parser.add_argument("-l", "--log", type=str, default="dmm.log")
    args = parser.parse_args()

    main(args)