### Electric Demand and Power Price Forecasting 

#### Business Goal: 

By combining the weather and energy consumption data to derive accurate predictions for electricity demand and pricing in a time-dependent context.

#### Data Preparation: 

Input:

1. Weather: Temp and Rain amount 

2. Energy: Various power generation amounts, Total load actual, Price day ahead, Price actual.

Preprocess: 

1. Drop features with high correlations to avoid  multicollinearity causing the coefficient varies significantly. For example, weather description, weather main.

#### Feature Engineering: 

1. Add hour, weekday, weekend, month, year to the data. Concatenate weather and energy data frame together.

2. Time-based features: Hour, day, month, weekday, weekend.

3. Lagged demand & price features: Previous 24-hour, 7-day trends.

4. Rolling statistics: 7-day moving average of demand and prices.

5. Cyclical encoding: Convert hour and weekday into sine/cosine features.

#### Model Development: 

##### 1. XGBoost: 

i. Capture non-linear relationships: Electricity demand and pricing often depends on complex, non-linear interactions such as weekend, weekday, temperature.

ii. Robustness: XGBoost is robust because it includes regularization to prevent overfitting. 

XGBoost is not designed for time series of forecasting, but we adapt it by converting the time series data into a supervised learning problem. 

- Time-Based Feature: Extract features such as hours of the day, day of the week, month, year. This helps model to capture time-electricity consumption-price relationships.

- Sliding Window Technique: in our case, we use 24 hours as window size and current hour’s price as target value.

Loss Function: MAE

##### 2. LSTM+Attention

LSTM networks excel at capturing long-term dependencies in sequential data, which is crucial for time series forecasting. Adding an attention mechanism further enhances this by allowing the model to focus on specific, critical time steps—such as peak hours or extreme weather events—that have a disproportionate impact on energy prices. For instance, when there's a sudden drop in temperature or a surge in consumption, the attention layer can assign greater weight to those moments, ensuring the model pays closer attention to the most influential factors.

###### Architecture:

Input Layer —> Masking —> LSTM —> Attention (Dot Product) —> Dense Layer —> Context (Dot Product) —> Flatten —> Dense

###### Input Layer: 

Input Layer defines input data shape (batch_size, hist_size, n_features). For example, hist_size = 24, every step has 16 features. Every data shape is (24, 16).

###### Masking:

This layer is used to ignore the padding in the input data—specifically, the positions where the values are all zeros. Its purpose is to ensure that the model only focuses on the valid data and avoids interference from the meaningless zero values in subsequent computations.

###### LSTM:

LSTM is used to capture long-term dependencies. It returns hidden states at every time step, so the output shape is (batch_size, 24, 132).

###### Attention:

This operation computes the similarity between the hidden state of each time step and the hidden states of all other time steps.

The result is attention score matrix with the shape (batch_size, 24, 24) which represents the similarity between every pair of time steps in the sequence.

The attention scores helps the model focus on the most relevant parts of historical sequence when predicting the price. For example, 6pm, 7pm, 8pm and 9pm the electricity demand is high. 2am, 3am, 4am, and 5am the demand might be low. 

How the attention matrix gives larger number for those time periods having higher impact to the price?

1. Hidden State Similarity: 

The  dot product is higher when hidden state representations are similar.

2. Training Process: 

If the features during the 6pm to 9pm consistently contribute more to accurate predictions. The network learns to “focus” on those periods. The learned hidden states for these time periods become similar, resulting in larger attention score.

###### Dense + Softmax:

The Dense layer and softmax activation together transform the raw attention scores into a normalized weight distribution for each time step. 

###### Context:

Dot product attention matrix and LSTM output to generate a context vector consolidate information from the entire sequence and emphasizes the time step that the model considers to be more important.

###### Flatten:

Convert multiple-dimensional tensor into a one-dimensional vector for each sample.

###### Dense Layer:

1.Feature Integration:

Combines input features by computing a weighted sum of inputs. This allows to integrate information from different from different features.

2. Learning Non-Linear Relationships:

A dense layer typically applies a non-linear activation function such as Softmax (ensuring that the weights for each time step sum to 1)

3. Dimensionality Transformation: Change the dimensionality of the data. For example, it can reduce a high-dimensional feature vector to a low-dimensional space.

##### 3.LSTM+Attention+XGBoost

The idea is that LSTM-Attention model captures the sequential patterns, but there might be additional information (e.g. non-sequential feature interaction) that does not capture perfectly. These missed aspects show up as residuals.

The XGBoost model learns to capture the residuals: non-sequential interactions.

In our case

1. Non-sequential data is like hour, weekday, month, year, since their value not vary with time.

For example, hour 3 (3am) is low demand, hour 18 (18pm) is high demand. Model can directly use this feature to better predict the price.

2. Sequential data is like generation xxx energy, total load actual, total load forecast, price actual, temp_city. Those values change with time.

#### Result

The results between XGBoost, LSTM+Attention, LSTM+Attention+XGBoost are very similar. 

1. Data Characteristics: If the time-related information is already well represented by non-sequential features (such as hour, weekday, etc.) — then all different models can access the crucial information. This means regardless of the model architecture, they may achieve similar predictive accuracy. For example, hour, weekday, those non-sequential feature can capture pattern of sequential features.
