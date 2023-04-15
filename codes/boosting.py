# -*- coding: utf-8 -*-
"""

Extra Trees Regressor

"""

import pandas as pd, numpy as np, time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error


# Load the dataset
df = pd.read_excel('C:/Users/Yash Dahima/PhD/Course Work/ML/Project/AQI/Datasets/data4.xlsx')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')
#df = df['2011-01-01':'2013-12-31']
df_scaled = StandardScaler().fit_transform(df)

# Separate the target variable from the features
#X, y = df_scaled[:, :-1], df_scaled[:, -1]
X = df.drop('pm2p5', axis=1)
y = df['pm2p5']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

#n_estimators, random_state, n_jobs =10000, 0, -1

# Create a list of models to evaluate
models = [ExtraTreesRegressor(n_estimators=10000, random_state=0, n_jobs=-1, criterion='friedman_mse', min_samples_split=20, min_samples_leaf=5, max_features=0.3),
          RandomForestRegressor(n_estimators=10000, random_state=0, n_jobs=-1, criterion='friedman_mse', min_samples_split=20, min_samples_leaf=5, max_features=0.3),
          GradientBoostingRegressor(n_estimators=10000, random_state=0, loss='huber', learning_rate=0.01, subsample=0.9, criterion='friedman_mse', min_samples_split=20, min_samples_leaf=5, max_depth=None, max_features=0.3),
          HistGradientBoostingRegressor(max_iter=10000, random_state=0, max_leaf_nodes=None, learning_rate=0.01, min_samples_leaf=5)
          ]

# Create an empty dataframe to store the evaluation metrics
metrics_df = pd.DataFrame(columns=["Model", "evs", "mae", "mape", "rmse", "r^2 Score"])


# Evaluate each model using a for loop
for model in models:
    
    start_time = time.time()
    
    # Train the model
    model.fit(X_train, y_train)
    
    print(f"Time for fitting: {(time.time() - start_time)/60:.3f} minutes")

    # Evaluate the model on the testing set
    y_pred = model.predict(X_test)

    # Compute evaluation metrics
    evs = explained_variance_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)*100
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Append evaluation metrics to the dataframe
    metrics_df = pd.concat([metrics_df, pd.DataFrame({"Model": [type(model).__name__],
                                                   "evs": [evs],
                                                   "mae": [mae],
                                                   "mape": [mape],
                                                   "rmse": [rmse],
                                                   "r^2 Score": [r2]})],
                           ignore_index=True)




"""

# Create a list of models to evaluate
models = [ExtraTreesRegressor(n_estimators=10000, random_state=0, n_jobs=-1),
          RandomForestRegressor(n_estimators=10000, random_state=0, n_jobs=-1),
          GradientBoostingRegressor(n_estimators=10000, random_state=0),
          HistGradientBoostingRegressor(max_iter=10000, random_state=0, max_leaf_nodes=None)
          ]

                           Model       evs        mae       mape       rmse  \
0            ExtraTreesRegressor  0.821801  13.644364  22.681243  18.037777   
1          RandomForestRegressor  0.815863  13.882158  22.913920  18.335334   
2      GradientBoostingRegressor  0.787091  14.853880  24.022811  19.714138   
3  HistGradientBoostingRegressor  0.809357  14.043517  22.831710  18.652113   

   r^2 Score  
0   0.821707  
1   0.815776  
2   0.787027  
3   0.809355  



# Create a list of models to evaluate
models = [ExtraTreesRegressor(n_estimators=10000, random_state=0, n_jobs=-1, criterion='friedman_mse', min_samples_split=20, min_samples_leaf=5, max_features=0.3),
          RandomForestRegressor(n_estimators=10000, random_state=0, n_jobs=-1, criterion='friedman_mse', min_samples_split=20, min_samples_leaf=5, max_features=0.3),
          GradientBoostingRegressor(n_estimators=10000, random_state=0, loss='huber', learning_rate=0.01, subsample=0.9, criterion='friedman_mse', min_samples_split=20, min_samples_leaf=5, max_depth=None, max_features=0.3),
          HistGradientBoostingRegressor(max_iter=10000, random_state=0, max_leaf_nodes=None, learning_rate=0.01, min_samples_leaf=5)
          ]


                           Model       evs        mae       mape       rmse  \
0            ExtraTreesRegressor  0.782994  15.098073  25.730924  19.901881   
1          RandomForestRegressor  0.814737  13.954160  23.220833  18.386972   
2      GradientBoostingRegressor  0.818324  13.633033  22.197581  18.209159   
3  HistGradientBoostingRegressor  0.799782  14.354958  23.540331  19.116580   

   r^2 Score  
0   0.782951  
1   0.814737  
2   0.818303  
3   0.799742 


"""











