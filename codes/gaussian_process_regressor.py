# -*- coding: utf-8 -*-
"""

Gaussian Process Regressor

"""

import pandas as pd, numpy as np, time
from scipy.stats import pearsonr, spearmanr

from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import ExpSineSquared, WhiteKernel, RBF, RationalQuadratic
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error


# Load the dataset
df = pd.read_excel('C:/Users/Yash Dahima/PhD/Course Work/ML/Project/AQI/Datasets/data4.xlsx')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')

#df = df['2011-01-01':'2013-12-31']
# Separate the target variable from the features
X = df.drop('pm2p5', axis=1)
y = df['pm2p5']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Define the kernels
kernel_daily = ExpSineSquared(length_scale=1.0, periodicity=1.0)
kernel_seasonal = ExpSineSquared(length_scale=1.0, periodicity=365.0)
kernel_trend = RationalQuadratic(length_scale=1.0, alpha=1.0)

# Combine the kernels using addition and multiplication
kernel = RBF(length_scale=1.0) * (kernel_daily + kernel_seasonal) + kernel_trend

# Initialize the GaussianProcessRegressor model
gaussian_process = GaussianProcessRegressor(kernel=kernel)

# Fit the model to the training data
start_time = time.time()
gaussian_process.fit(X_train, y_train)
print(f"Time for GaussianProcessRegressor fitting: {time.time() - start_time:.3f} seconds")

# Evaluate the model on the testing set
y_pred = gaussian_process.predict(X_test)

# Compute evaluation metrics
#coeff = model.coef_
evs = explained_variance_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)*100
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

#df['numerical_year'] = df['datetime'].apply(lambda x: x.year + (1/365.25)*((x.dayofyear - 1) + x.hour/24 + x.minute/24*60))
#X = df['numerical_year']

"""

------------------------------------------------------------------------

Using full data (all years, all features)

# Define the kernels
kernel_daily = ExpSineSquared(length_scale=1.0, periodicity=1.0)
kernel_seasonal = ExpSineSquared(length_scale=1.0, periodicity=365.0)
kernel_trend = RationalQuadratic(length_scale=1.0, alpha=1.0)

# Combine the kernels using addition and multiplication
kernel = RBF(length_scale=1.0) * (kernel_daily + kernel_seasonal) + kernel_trend

# Initialize the GaussianProcessRegressor model
gaussian_process = GaussianProcessRegressor(kernel=kernel)

evs = 0.616
mae = 32.404
mape = 44.531
rmse = 40.514
r2 = 0.101

------------------------------------------------------------------------



"""