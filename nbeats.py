import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K

from sklearn.preprocessing import StandardScaler


class ModelNBEATS():
    def __init__(self, y, params, **kwargs):
            """Creates a TCNN model object.
        Params: y - pandas DataFrame/Series with all time series required for modeling purposes.
            **kwargs:
                xreg - pandas DataFrame with all external regressors (for prediction purposes, additional periods must be included)
                scale_data - bool, use standard scaler to scale data to 0-1 range; default True
                num_layers - int, number of dense layers to use in NBEATS block; default 4
                num_neurons - int, number of neurons within each dense layer default 128
                stacks - list/str, list or comma separated str of values for stacks; default: 'generic, trend, seasonality, generic'
                activation - str; default 'relu'
                optimizer - str/keras optimizer object; default 'adam'
                num_epoch - int, number of epochs for training; default 100
                loss - str/keras optimizer object, used for loss evaluation in training; default 'mse'
        """

    def __init__(self, y, **kwargs):
        if isinstance(y, pd.Series):
            self.id = y.name
            self.out = 1
        else:
            self.id = y.columns
            self.out = len(y.columns)

        self.xreg = kwargs.get('xreg', pd.DataFrame())
        if self.xreg.shape[0]>0:
            self.y = y.join(self.xreg).replace([np.inf, -np.inf], 0).fillna(0)
        else:
            self.y = y.replace([np.inf, -np.inf], 0).fillna(0)

        self.y_norm = self.y.copy()
        self.scale_data = kwargs.get('scale_data', True)
        if self.scale_data:
            self.scaler = StandardScaler().fit(y)
            self.y_norm[self.id] = self.scaler.transform(y)

        self.num_layers = kwargs.get('num_layers',4)
        self.num_neurons = kwargs.get('num_neurons',128)
        self.stacks = kwargs.get('stacks', 'generic, trend, seasonality, generic')
        if type(self.stacks)==str:
            self.stacks=self.stacks.split(',')
        self.activation = kwargs.get('activation','relu')
        self.optimizer = kwargs.get('optimizer','adam')
        self.num_epoch = kwargs.get('num_epoch',100)
        self.loss = kwargs.get('loss','mse')
        self.model = None

    class NBeatsBlock(tf.keras.layers.Layer):
        def __init__(self, input_size: int, theta_size: int, horizon: int, n_neurons: int, n_layers: int, stack_type: str, **kwargs):
            super().__init__(**kwargs)
            self.input_size = input_size
            self.theta_size = theta_size
            self.horizon = horizon
            self.n_neurons = n_neurons
            self.n_layers = n_layers
            self.stack_type = stack_type 

            # by default block contains stack of 4 fully connected layers each has ReLU activation
            self.hidden = [tf.keras.layers.Dense(n_neurons, activation="relu") for _ in range(n_layers)]
            # Output of block is a theta layer with linear activation
            self.theta_layer = tf.keras.layers.Dense(theta_size, activation="linear", name="theta")

        def linear_space(self, backcast_length, forecast_length, is_forecast=True):
            ls = K.arange(-float(backcast_length), float(forecast_length), 1) / forecast_length
            return ls[backcast_length:] if is_forecast else K.abs(K.reverse(ls[:backcast_length], axes=0))

        def seasonality_model(self, thetas, backcast_length, forecast_length, is_forecast):
            p = thetas.get_shape().as_list()[-1]
            p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
            t = self.linear_space(backcast_length, forecast_length, is_forecast=is_forecast)
            s1 = K.stack([K.cos(2 * np.pi * i * t) for i in range(p1)])
            s2 = K.stack([K.sin(2 * np.pi * i * t) for i in range(p2)])
            if p == 1:
                s = s2
            else:
                s = K.concatenate([s1, s2], axis=0)
            s = K.cast(s, np.float32)
            return K.dot(thetas, s)

        def trend_model(self, thetas, backcast_length, forecast_length, is_forecast):
            p = thetas.shape[-1]           # take time dimension
            t = self.linear_space(backcast_length, forecast_length, is_forecast=is_forecast)
            t = K.transpose(K.stack([t for i in range(p)]))
            t = K.cast(t, np.float32)
            return K.dot(thetas, K.transpose(t)) 
            
        def call(self, inputs): 
            x = inputs 
            for layer in self.hidden:
                x = layer(x)
                theta = self.theta_layer(x)
                
            if self.stack_type == 'generic':
                backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]
            elif self.stack_type == 'seasonal':
                backcast = tf.keras.layers.Lambda(self.seasonality_model, arguments={'is_forecast': False, 'backcast_length': self.theta_size, 'forecast_length': self.horizon}, name='seasonal')(theta[:, :self.input_size])
                forecast = tf.keras.layers.Lambda(self.seasonality_model, arguments={'is_forecast': True, 'backcast_length': self.theta_size, 'forecast_length': self.horizon}, name='seasonal')(theta[:, -self.horizon:])
            else:
                backcast = tf.keras.layers.Lambda(self.trend_model, arguments={'is_forecast': False, 'backcast_length': self.theta_size, 'forecast_length': self.horizon}, name='trend')(theta[:, :self.input_size])
                forecast = tf.keras.layers.Lambda(self.trend_model, arguments={'is_forecast': True, 'backcast_length': self.theta_size, 'forecast_length': self.horizon}, name='trend')(theta[:, -self.horizon:])
            return backcast, forecast

    def _define_model_object(self):
        """
        Build NBEATS model arhitecture
        """
        if self.model is not None:
            return self.model
        else: 
            shape_t, shape_f = len(self.y.index)//2, self.y_norm.shape[1]

            inputs = tf.keras.layers.Input(shape=(shape_t, shape_f))
            
            initial_block = self.NBeatsBlock(input_size=shape_t, theta_size=shape_f, horizon=1, n_neurons=self.num_neurons, n_layers=self.num_layers, stack_type=self.stacks)
            residuals, forecast = initial_block(inputs)
            for i in range(1, len(self.stacks)):
                backcast, block_forecast = self.NBeatsBlock(input_size=inputs.shape[1], theta_size=inputs.shape[2], horizon=1, n_neurons=self.num_neurons, n_layers=self.num_layers, stack_type=self.stacks)(residuals) 
                residuals = tf.keras.layers.subtract([residuals, backcast], name=f"subtract_{i}")
                forecast = tf.keras.layers.add([forecast, block_forecast], name=f"add_{i}")

            model = tf.keras.Model(inputs=inputs, outputs=forecast[0])

        return model

    def fit(self):
        """
        Fit model to the provided data
        """
        model = self._define_model_object()

        generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(self.y_norm.values, self.y_norm[self.id].values, length=model.input.get_shape()[1], batch_size=1)  
        model.compile(optimizer=self.optimizer, loss=self.loss)

        model.fit(generator, steps_per_epoch=1, epochs=self.num_epoch, shuffle=False, verbose=0)
        self.model = model

        return self.model

    def save(self, path):
        """
        Save model object - provide full path, for example: '~/usr/models/mymodel.h5'
        """
        self.model.save(path)
    
    def load(self, path):
        """
        Load model object - provide full path, for example: '~/usr/models/mymodel.h5'
        """
        self.model = tf.keras.models.load_model(path)

    def predict(self, h):
        """
        Generate predictions for h steps ahead
        Params: h - number of steps to forecast
        If xreg data was used during the training, it must be included for next h periods in the future
        """
        periods=pd.date_range(start=max(self.y.index), periods=h+1, freq=self.y.index.freq)[1:]
        pred = pd.DataFrame(data=[], columns=self.y.columns, index=periods)
        if self.xreg.shape[0]>0:
            pred[self.xreg.columns] = self.xreg[self.xreg.index.isin(pred.index)].values
        tmp_pred = self.y_norm[-self.model.input.get_shape()[1]:]
        for i in range(h):
            inp = np.asarray(tmp_pred[-self.model.input.get_shape()[1]:].values.reshape((1, self.model.input.get_shape()[1], self.y_norm.shape[1]))).astype(np.float32)
            p = self.model.predict(inp, verbose=0)
            pred.loc[pred.index[i], self.id] = p
            tmp_pred = pd.concat([tmp_pred, pred.iloc[[i]]])

        if self.scale_data:
            res = self.scaler.inverse_transform(pred)
        else:
            res = pred.values
        res = pd.DataFrame(data=np.where(res<0, 0, res.astype(int)), columns=pred.columns, index=pred.index)
        return res
