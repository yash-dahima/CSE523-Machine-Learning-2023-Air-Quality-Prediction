# -*- coding: utf-8 -*-
"""

Kernel Ridge Regression

"""

import pandas as pd, numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.kernel_ridge import KernelRidge
from sklearn.utils.fixes import loguniform

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error


# Load the dataset
df = pd.read_excel('C:/Users/Yash Dahima/PhD/Course Work/ML/Project/AQI/Datasets/data4.xlsx')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')


#df = df['2011-01-01':'2012-12-31']

# Separate the target variable from the features
X = df.drop('pm2p5', axis=1)
y = df['pm2p5']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

kernel_ridge = KernelRidge(kernel=ExpSineSquared())

param_distributions = {
    "alpha": loguniform(1e-2, 1e3),
    "kernel__length_scale": loguniform(1e-2, 1e2),
    "kernel__periodicity": loguniform(1e-2, 1e2),
}

kernel_ridge_tuned = RandomizedSearchCV(
    kernel_ridge,
    param_distributions=param_distributions,
    n_iter=100,
    random_state=0,
)


# Train the model
kernel_ridge_tuned.fit(X_train, y_train)

# Evaluate the model on the testing set
y_pred = kernel_ridge_tuned.predict(X_test)

# Compute evaluation metrics
#coeff = model.coef_
evs = explained_variance_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)*100
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


"""

====================================================================================

kernel_ridge = KernelRidge(kernel=ExpSineSquared())

param_distributions = {
    "alpha": loguniform(1e0, 1e3),
    "kernel__length_scale": loguniform(1e0, 1e2),
    "kernel__periodicity": loguniform(1e0, 1e2),
}

kernel_ridge_tuned = RandomizedSearchCV(
    kernel_ridge,
    param_distributions=param_distributions,
    n_iter=10,
    random_state=0,
)

kernel_ridge_tuned.best_params_

{'alpha': 682.3493012435788,
 'kernel__length_scale': 1.222906594703436,
 'kernel__periodicity': 2.5981363554663415}


evs = 0.00049
mae = 41.56
mape = 45.69 %
mse = 3158.35
rmse = 56.199
r2 = -0.349
====================================================================================

kernel_ridge = KernelRidge(kernel=ExpSineSquared())

param_distributions = {
    "alpha": loguniform(1e-2, 1e3),
    "kernel__length_scale": loguniform(1e-2, 1e2),
    "kernel__periodicity": loguniform(1e-2, 1e2),
}

kernel_ridge_tuned = RandomizedSearchCV(
    kernel_ridge,
    param_distributions=param_distributions,
    n_iter=100,
    random_state=0,
)


kernel_ridge_tuned.best_params_

{'alpha': 771.7483306124897,
 'kernel__length_scale': 32.06424226809186,
 'kernel__periodicity': 0.2252349634321646}

evs = -2.71e-6
mae = 40.45
mape = 47.46 %
rmse = 53.91
r2 = -0.24


"""