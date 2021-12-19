import utils
import os
import numpy as np
	
from math import sqrt
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

dir_name = os.path.dirname(os.path.abspath(__file__))


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
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=75, validation_data=(val_X, val_y), verbose=2, shuffle=True)
    #plot loss
    pyplot.plot(history.history['loss'], label='E-train')
    pyplot.plot(history.history['val_loss'], label='E-val')
    pyplot.xlabel("Dataset: test")
    pyplot.legend()
    pyplot.savefig('{}/figures/loss_model_0'.format(dir_name))
    pyplot.clf()
    # make a prediction
    yhat = model.predict(test_X)
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
    return model


            