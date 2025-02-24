# Number of Sun Spot Analysis

### Business Goal: 
Predict the number of sunspots based on historical data.

### Data Preparation: 
Creating the input-output pair by defining the window size.

Shuffle the window size data. This can help the model to be more general. 

### ML model: Neural model because we want to capture the non-linear and long-term dependency relationship inside the dataset. LSTM, RNN, CNN, DNN. 
CNN + LSTM + DNN shows best result: 

CNN: Extract the local feature and trend within the window. (It also helps to smooth out the noise to help LSTM less affected by the local extreme value and focus on long term trend).

LSTM: Capture the long term dependency.

DNN:  Non-linear transformation and information compression.

### Loss function: 
Huber Loss. It is better ability to handle outlier because it uses absolute error for large error and squared error for small error.

### Offline metrics: Number of Sun Spot is a regression problem.
MAE: calculate average absolute difference

MSE: penalize more to the large error.

### Optimizer: SGD + momentum 
SGD + momentum has advantage more general but more sensitive to initial learning rate choice. 

Why not adam optimizer?
Adam optimizer has advantage: less sensitive to the initial learning rate. But it is less general because it tries to adjust learning rate for every hyperparameter.

