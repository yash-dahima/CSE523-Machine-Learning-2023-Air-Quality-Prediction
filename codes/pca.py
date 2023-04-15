# -*- coding: utf-8 -*-
"""

Principal Component Analysis (PCA) is a popular technique for dimensionality reduction. PCA transforms the original features into a smaller set of features that capture most of the variance in the data. It can be used to preprocess the data before building a machine learning model, as well as for data visualization.

Here's how you can perform PCA on your dataset in Python, along with scaling the values:

"""

import matplotlib as mpl
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, FactorAnalysis
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel('C:/Users/Yash Dahima/PhD/Course Work/ML/Project/AQI/Datasets/data4.xlsx')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')

# Separate the features and the target variable
X = df.drop('pm2p5', axis=1)
y = df['pm2p5']

# Scale the data
std_scaler = StandardScaler()
#min_max_scaler = MinMaxScaler()
X_scaled_std = std_scaler.fit_transform(X)
#X_scaled_min_max = min_max_scaler.fit_transform(X)


# ----------------------------------- Perform FA ----------------------------------- #

fa = FactorAnalysis(n_components=10)

# Fit and transform the data using PCA
X_fa = fa.fit_transform(X_scaled_std)

fa_loadings = fa.components_
fa_loadings = pd.DataFrame(fa_loadings, columns=X.columns)

plt.scatter(X_fa[:,0], X_fa[:,1], c=y)





# ----------------------------------- Perform PCA ----------------------------------- #

pca = PCA(n_components=5)

# Fit and transform the data using PCA
X_pca = pca.fit_transform(X_scaled_std)
    
# Calculate the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)

# Get the contribution of each original feature to each principal component
feature_contributions_std = pd.DataFrame(pca.components_, columns=X.columns)

# Plot the cumulative explained variance ratio
mpl.rcParams.update({'font.size': 14, 'font.weight':'bold', 'lines.linewidth':2})
plt.plot(np.arange(1, X.shape[1] +1), cumulative_explained_variance_ratio)
plt.xlabel('Number of principal components', fontweight ="bold")
plt.ylabel('Cumulative explained variance ratio', fontweight ="bold")
plt.legend()
plt.grid()




"""

In the above code, we first load the dataset and separate the features and the target variable. Then, we scale the data using the StandardScaler class from scikit-learn to ensure that all features have the same scale.

Next, we perform PCA using the PCA class from scikit-learn. By default, PCA retains all principal components (i.e., it does not perform dimensionality reduction). We can access the transformed data using the fit_transform() method of the PCA object.

We also calculate the explained variance ratio for each principal component. The explained variance ratio represents the proportion of the total variance in the data that is explained by each principal component. We print the explained variance ratio for each principal component to see how much of the variance is captured by each principal component.

Finally, we plot the cumulative explained variance ratio to determine the optimal number of principal components to retain. The cumulative explained variance ratio shows how much of the total variance is explained by the first n principal components. In this case, we can see that the first 5 principal components explain more than 90% of the variance in the data, so we might consider retaining these 5 components.

By performing PCA and scaling the data, we can reduce the dimensionality of the data and improve the performance of machine learning models that rely on the data.

"""