import os, csv, torch, json, copy

import torch.nn as nn
import numpy as np
import pandas as pd
from matplotlib import pyplot

dir_name = os.path.dirname(os.path.abspath(__file__))

# this function returns a 0/1 mask that can be used to mask out a mini-batch
# composed of sequences of length `seq_lengths`
def get_mini_batch_mask(mini_batch, seq_lengths):
    mask = torch.zeros(mini_batch.shape[0:2])
    for b in range(mini_batch.shape[0]):
        mask[b, 0 : seq_lengths[b]] = torch.ones(seq_lengths[b])
    return mask

# this function takes a torch mini-batch and reverses each sequence
# (w.r.t. the temporal axis, i.e. axis=1).
def reverse_sequences(mini_batch, seq_lengths):
    reversed_mini_batch = torch.zeros_like(mini_batch)
    for b in range(mini_batch.size(0)):
        T = seq_lengths[b]
        time_slice = torch.arange(T - 1, -1, -1, device=mini_batch.device)
        reversed_sequence = torch.index_select(mini_batch[b, :, :], 0, time_slice)
        reversed_mini_batch[b, 0:T, :] = reversed_sequence
    return reversed_mini_batch

def get_mini_batch(mini_batch_indices, sequences, seq_lengths, cuda=False):
	# get the sequence lengths of the mini-batch
    seq_lengths = seq_lengths[mini_batch_indices]
    # sort the sequence lengths
    _, sorted_seq_length_indices = torch.sort(seq_lengths)
    sorted_seq_length_indices = sorted_seq_length_indices.flip(0)
    sorted_seq_lengths = seq_lengths[sorted_seq_length_indices]
    sorted_mini_batch_indices = mini_batch_indices[sorted_seq_length_indices]

    # compute the length of the longest sequence in the mini-batch
    T_max = torch.max(seq_lengths)
    # this is the sorted mini-batch
    mini_batch = sequences[sorted_mini_batch_indices, 0:T_max, :]
    # this is the sorted mini-batch in reverse temporal order
    mini_batch_reversed = reverse_sequences(mini_batch, sorted_seq_lengths)
    # get mask for mini-batch
    mini_batch_mask = get_mini_batch_mask(mini_batch, sorted_seq_lengths)

    # cuda() here because need to cuda() before packing
    if cuda:
        mini_batch = mini_batch.cuda()
        mini_batch_mask = mini_batch_mask.cuda()
        mini_batch_reversed = mini_batch_reversed.cuda()

    # do sequence packing
    mini_batch_reversed = nn.utils.rnn.pack_padded_sequence(
        mini_batch_reversed, sorted_seq_lengths, batch_first=True)

    return mini_batch, mini_batch_reversed, mini_batch_mask, sorted_seq_lengths

def plot_features(idx,values):
	# specify columns to plot
	i = 1
    # plot each column
	pyplot.figure()
	axis = None
	for feature in idx:    
		if axis is None:
			axis = pyplot.subplot(len(idx), 1, i)
			pyplot.setp(axis.get_yticklabels(), fontsize=5)
		else: 
			subaxis = pyplot.subplot(len(idx), 1, i, sharex=axis) 
			pyplot.setp(subaxis.get_xticklabels(), visible=False)
			pyplot.setp(subaxis.get_yticklabels(), fontsize=5)
		pyplot.plot(values[:, idx[feature]])
		pyplot.title(feature, y=0.5, loc='right', fontsize=6)
		i += 1
		pyplot.savefig('{}/figures/features_{}'.format(dir_name,feature.replace('.','_')))
		pyplot.clf()

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def prep_data(data, output_dim):
	# split into train, val and test sets
	values = data.to_numpy()
	split_date_val, split_date_test = get_split_idx(data.index)
	train = values[:split_date_val, :]
	val = values[split_date_val:split_date_test,:]
	test = values[split_date_test:, :]
	# split into input and outputs
	train_X, train_y = train[:, :20], train[:, -output_dim:]
	val_X, val_y = val[:,:20], val[:,-output_dim:]
	test_X, test_y = test[:, :20], test[:, -output_dim:]
	# reshape input to be 3D [samples, timesteps, features]
	train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
	val_X = val_X.reshape((val_X.shape[0], 1, val_X.shape[1]))
	test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
	return (train_X,train_y,val_X,val_y,test_X,test_y)


def get_split_idx(data):
	n = len(data)
	return (round(n * 0.7), round(n * 0.85))

def load_data(file_name):

	with open(dir_name + '/data/driveTest/finalizedData{}.csv'.format(file_name), newline='') as csvfile:
		f_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')

		labels = next(f_reader)
		labels = labels[0].split(',')
		labels.pop(0)
		labels_dict = dict(zip(labels,range(len(labels))))

		data = []
 
		for row in f_reader:
			d = row[0].split(',')
			d.pop(0)
			data.append([float(x) for x in d])

		return labels_dict, data 

def load_exp_config(file_name):

    with open(dir_name + '/config/' + file_name) as json_file:
        return copy.deepcopy(json.load(json_file))
            