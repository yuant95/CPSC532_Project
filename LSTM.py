import utils
import os, copy
import numpy as np
	
from math import sqrt
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

dir_name = os.path.dirname(os.path.abspath(__file__))


def get_LSTM_UWb(weight):
    '''
    weight must be output of LSTM's layer.get_weights()
    W: weights for input
    U: weights for hidden states
    b: bias
    '''
    warr,uarr, barr = weight
    gates = ["i","f","c","o"]
    hunit = uarr.shape[0]
    U, W, b = {},{},{}
    for i1,i2 in enumerate(range(0,len(barr),hunit)):
        
        W[gates[i1]] = warr[:,i2:i2+hunit]
        U[gates[i1]] = uarr[:,i2:i2+hunit]
        b[gates[i1]] = barr[i2:i2+hunit].reshape(hunit,1)
    return(W,U,b)

def get_LSTMweights(model1):
    for layer in model1.layers:
        if "LSTM" in str(layer):
            w = layer.get_weights()
            W,U,b = get_LSTM_UWb(w)
            break
    return W,U,b

def get_dist_parm(model):
    for layer in model.layers:
        if "Dense" in str(layer):
            w = layer.get_weights()
            print(w[0].shape)
            break
    return w


def get_model(data):
    values = data
    output_dim = 18
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    reframed = utils.series_to_supervised(scaled, 1, 1)
    reframed.drop(reframed.columns[[20, 21]], axis=1, inplace=True)
    
    train_X, train_y, val_X, val_y, test_X, test_y = utils.prep_data(reframed, 18)

    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(18))

    # distribution_outputs = Lambda(negative_binomial_layer)(outputs)
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=75, validation_data=(val_X, val_y), verbose=2, shuffle=True)
    #plot loss
    pyplot.figure(figsize=(8,8))
    pyplot.plot(history.history['loss'], label='E-train')
    pyplot.plot(history.history['val_loss'], label='E-val')
    pyplot.xlabel("Dataset: test")
    pyplot.legend()
    pyplot.savefig('{}/figures/loss_model_0'.format(dir_name))
    pyplot.clf()
    # make a prediction
    yhat = model.predict(test_X)
    test_X_shape = copy.deepcopy(test_X)
    print('yhat shape? ', yhat.shape)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    print('test x shape', test_X.shape)
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, output_dim:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0:output_dim]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), output_dim))
    inv_y = np.concatenate((test_y, test_X[:, output_dim:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0:output_dim]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Dataset test ')
    print('Test RMSE: %.3f' % rmse)
    return model, train_X, train_y, val_X, val_y, test_X_shape, test_y


            