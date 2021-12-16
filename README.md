# nbeats-tf2
NBEATS model implementation in TF2/Keras for time series prediction

## API

The only dependency is tensorflow 2.x and ususal way to build model is to import model class and create the object instance by passing the time series dataframe.

```python
import pandas as pd
import tensorflow as tf
from tcnn import ModelNBEATS

train = pd.read_csv('train.csv')
train.index = pd.date_range(start='2010-01-01',periods=146, freq='D')

model = ModelNBEATS(y=train)            # use default values
model.fit()
model.predict(10)
...

###### you can access tf model object via model.model_
model.model_.summary()

model.save('save_path')
######
model.load('load_path')
```

### Arguments

```python
model = ModelNBEATS(y=train, {'xreg': 0, 
                            'scale_data':True,
                            'num_layers':4,
                            'num_neurons':128,
                            'stacks':'generic, trend, seasonality, generic':,
                            'activation':'relu',
                            'optimizer':'adam',
                            'num_epoch':100,
                            'loss':'mse'
})
```
- `xreg`: pd.DataFrame. DataFrame with all external regressors (for prediction purposes, additional periods must be included)
- `scale_data`: bool, whether to use standard scaler to scale data to 0-1 range; default True
- `num_layers`: int, number of dense layers to use in NBEATS block; default 4
- `num_neurons`: int, number of neurons within each dense layer; default 128
- `stacks` : list or str, list or comma separated str of values for stacks; default: 'generic, trend, seasonality, generic'
- `activation`: str. The activation used in the residual blocks o = activation(x + F(x)).
- `optimizer`: str/keras. Optimizer to use when training the model, either str or keras optimizer object - default 'adam'.
- `num_epoch`: int. Number of epochs to train the model on - default 100
- `loss`: str/keras. Loss function to use when training the model - deault 'mse'

### Input shape

Pandas DataFrame with time dimension in the rows and unique time-series in columns. Row index sould be date/date-time based.

### Output shape

Pandas DataFrame. By calling model.predict(num_steps), pandas dataframe will be generated with num_steps rows and same number of columns as the initial training dataset.

## Installation from the source

```bash
git clone git@github.com:apantovic/nbeats-tf2.git && cd nbeats-tf2
virtualenv -p python3 venv
source venv/bin/activate
pip install tensorflow==2.5.0
```

