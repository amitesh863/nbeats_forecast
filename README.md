# nbeats_forecast

##### Neural Beats implementation library

nbeats_forecast is end to end library for univariate time series forecasting using N-BEATS (https://arxiv.org/pdf/1905.10437v3.pdf - Published as conference paper in ICLR).
This library uses nbeats-pytorch (https://github.com/philipperemy/n-beats) as base and simplifies the task of forecasting using N-BEATS by providing a interface similar to scikit-learn and keras.

### Installation

```sh
$ pip install nbeats_forecast
```

#### Import
```sh
from nbeats_forecast import NBeats
```

#### Input
numpy array of size nx1 

#### Output
Forecasted values as numpy array of size mx1 

Mandatory Parameters for the model:
- data
- period_to_forecast

Basic model with only mandatory parameters can be used to get forecasted values as shown below:
```sh
import pandas as pd
from nbeats_forecast import NBeats

data = pd.read_csv('data.csv')   
data = data.values        #univariate time series data of shape nx1 (numpy array)

model = NBeats(data=data, period_to_forecast=12)
model.fit()
forecast = model.predict()
```

Other optional parameters for the object of the model  (as described in the paper) can be tweaked for better performance. If these parameters are not passed, default values as mentioned in the table below are considered.

#### Optional parameters
| Parameter | Type | Default Value| Description|
| ------ | ------ | --------------|------------|
| backcast_length | integer | 3* period_to_forecast |Explained in paper|
| path | string | '  ' |path to save intermediate training checkpoint |
| checkpoint_file_name | string | 'nbeats-training-checkpoint.th'| name for checkpoint file ending in format  .th |
|mode| string| 'cpu'| Any of the torch.device modes|
| batch_size | integer | len(data)/15 | size of batch|
|  thetas_dims | list of integers | [7, 8] | Explained in paper|
| nb_blocks_per_stack | integer | 3| Explained in paper|
|  share_weights_in_stack | boolean | False | Explained in paper|
| train_percent | float(below 1)  | 0.8 | Percentage of data to be used for training |
| save_checkpoint| boolean | False | save intermediate checkpoint files|
| hidden_layer_units | integer | 128 | number of hidden layer
| stack | list of integers | [1,1] | adding stacks in the model as per the paper passed in list as integer. Mapping is as follows -- 1: GENERIC_BLOCK,  2: TREND_BLOCK , 3: SEASONALITY_BLOCK|


#### Methods

#### fit(epoch,optimiser,plot,verbose):
This method is used to train the model for number of gradient steps passed in the object of the model.

###### Parameter - epoch : integer

epoch is 100 * gradient steps. 25 epoch means 2500 weight updation steps.
If optimizer is not passed, default value is 25.

###### Parameter - optimiser

optimizer from torch.optim can be passed as a parameter by including model.parameters as the variable.

Example: 

model.fit(epoch=5,optimiser=torch.optim.AdamW(model.parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False))

If optimizer is not passed, default optimizer used is Adam.

###### Parameter- plot : boolean
Default value - False

If True, plots during training are shown.

###### Parameter- verbose : boolean

Default value - True

If True, training details are printed.

#### predict():
Returns forecasted values.

#### save(file):
Saves the current step model after training. File needs to be passed a string. Format of the model to be saved is .th

Example: model.save('result/model.th')

#### load(file):
Load the saved model with format .th

Example: model.load('result/model.th')

### Example 
Model with a different optimiser and different stack is shown here.

Here  2: TREND_BLOCK and 3: SEASONALITY_BLOCK stacks are used.
```sh
import pandas as pd
from nbeats_forecast import NBeats
from torch import optim

data = pd.read_csv('data.csv')   
data = data.values #univariate time series data of shape nx1(numpy array)

model=NBeats(data=data,period_to_forecast=12,stack=[2,3],thetas_dims=[2,8])

model.fit(epoch=5,optimiser=optim.AdamW(model.parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False))

forecast=model.predict()
```

