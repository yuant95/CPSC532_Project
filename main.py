import argparse, logging, time

import os, torch, copy
import pyro
import LSTM, MCMC, utils
import torch.nn as nn
from matplotlib import pyplot
import jax.numpy as jnp
import numpy as np

from os.path import exists
from pyro.optim import ClippedAdam

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
from VRNN import DMM


dir_name = os.path.dirname(os.path.abspath(__file__))

 
def main(idx,data,config):
    '''setup, training, and evaluation'''
    # setup logging
    logging.basicConfig(level=logging.DEBUG, format="%(message)s", filename='{}.log'.format(config["id"]), filemode="w")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)
    logging.info(config)

    values = data.data
    # plot features as time series
    utils.plot_features(idx, values)
    # set up model 0 LSTM s.t (s_t|s_{t-1},a_{t-1})
    model_0, train_X, train_y, val_X, val_y, test_X, test_y = LSTM.get_model(values)
    # extract weights, H (internal state), bias  
    W,H,b = LSTM.get_LSTMweights(model_0)
    ps = LSTM.get_dist_parm(model_0)

    for key in W:
        print('Key? ', key)
        print('Weights shape? ', W[key].shape)

    # set up model 1 BNN s.t (x_t | x_{t-1},a_{t-1}) and (s_t|x_t)
    samples, predictions, mean_prediction = MCMC.run_sampler(config, data, len(values), W['o'], H['o'], ps[0])

    

    training_seq_lengths = torch.ones(train_X.shape[0])    
    training_data_sequences = torch.tensor(train_X)        
    test_seq_lengths = torch.ones(test_X.shape[0]) 
    test_data_sequences = torch.tensor(test_X)
    val_seq_lengths = torch.ones(val_X.shape[0]) 
    val_data_sequences = torch.tensor(val_X)
    N_train_data = len(training_seq_lengths)
    N_train_time_slices = float(torch.sum(training_seq_lengths))
    N_mini_batches = int(N_train_data / config['config']['mini_batch_size'] + int(N_train_data % config['config']['mini_batch_size'] > 0))

    logging.info("N_train_data: %d     avg. training seq. length: %.2f    N_mini_batches: %d" % (N_train_data, training_seq_lengths.float().mean(), N_mini_batches))

    # how often we do validation/test evaluation during training
    val_test_frequency = config['config']['val_test_frequency']
    # the number of samples we use to do the evaluation
    n_eval_samples = config['config']['n_eval_samples']

    print('test X? ', test_X.shape, 'Val X? ', val_data_sequences.size())

    # package repeated copies of val/test data for faster evaluation
    # (i.e. set us up for vectorization)
    def rep(x):
        rep_shape = torch.Size([x.size(0) * n_eval_samples]) + x.size()[1:]
        repeat_dims = [1] * len(x.size())
        repeat_dims[0] = n_eval_samples
        return (x.repeat(repeat_dims).reshape(n_eval_samples, -1).transpose(1, 0).reshape(rep_shape))

    # get the validation/test data ready for the dmm: pack into sequences, etc.
    val_seq_lengths = rep(val_seq_lengths)
    test_seq_lengths = rep(test_seq_lengths)
    (val_batch,val_batch_reversed,val_batch_mask,val_seq_lengths,) = utils.get_mini_batch(
        torch.arange(n_eval_samples * val_data_sequences.shape[0], dtype=torch.int64), rep(val_data_sequences), val_seq_lengths)
    (test_batch,test_batch_reversed,test_batch_mask,test_seq_lengths) = utils.get_mini_batch(
        torch.arange(n_eval_samples * test_data_sequences.shape[0],dtype=torch.int64), rep(test_data_sequences), test_seq_lengths)

    # instantiate the dmm with prior from BNN
    dmm = DMM(
        rnn_dropout_rate=config['config']['drop_out_rate'],
        num_iafs=config['config']['num_iafs'],
        iaf_dim=config['config']['iaf_dim'],
        use_cuda=config['config']['cuda'],
        emitter=torch.from_numpy(copy.deepcopy(np.asarray(samples['w1']))),
        trans=torch.from_numpy(copy.deepcopy(np.asarray(samples['w1']))),
        comb=torch.from_numpy(copy.deepcopy(np.asarray(samples['w3']))))

    adam = ClippedAdam(config["optimizer"])

    # setup inference algorithm
    if config['alg'] == 'tmc':
        tmc_loss = TraceTMC_ELBO()
        dmm_guide = config_enumerate(
            dmm.guide,
            default="parallel",
            num_samples=config['tmc']['num_samples'],
            expand=False,
        )
        svi = SVI(dmm.model, dmm_guide, adam, loss=tmc_loss)
    elif args.tmcelbo:
        elbo = TraceEnum_ELBO()
        dmm_guide = config_enumerate(
            dmm.guide,
            default="parallel",
            num_samples=config['tmc']['num_samples'],
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
            cuda=True,
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
    if config['load_opt'] != "" and config['load_model'] != "":
        load_checkpoint()

    #################
    # TRAINING LOOP #
    #################
    times = [time.time()]
    for epoch in range(config['VRNN']['num_epochs']):
        # if specified, save model and optimizer states to disk every checkpoint_freq epochs
        if config['VRNN']['checkpoint_freq'] > 0 and epoch > 0 and epoch % config['VRNN']['checkpoint_freq'] == 0:
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
        logging.info( "[training epoch %04d]  %.4f \t\t\t\t(dt = %.3f sec)"
            % (epoch, epoch_nll / N_train_time_slices, epoch_time))

        # do evaluation on test and validation data and report results
        if val_test_frequency > 0 and epoch > 0 and epoch % val_test_frequency == 0:
            val_nll, test_nll = do_evaluation()
            logging.info(
                "[val/test epoch %04d]  %.4f  %.4f" % (epoch, val_nll, test_nll))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Running Inference")
    parser.add_argument("-i", "--id", default=0, type=int)
    parser.add_argument("-dfn", "--data-fn", default='', type=str)
    parser.add_argument("-cnfg", "--config", default='test.json', type=str)
    parser.add_argument("-l", "--log", type=str, default="dmm.log")
    args = parser.parse_args()

    print('Loading Data from exp no. {}'.format(args.id)) 

    
    idx, data = utils.load_data(args.data_fn)
    config = utils.load_exp_config(args.config)

    main(idx,torch.tensor(data),config)

    

    