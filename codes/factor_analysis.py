# -*- coding: utf-8 -*-
"""

Factor Analysis using Kaiser criterion and Scree plot

"""

import matplotlib as mpl, pandas as pd, numpy as np, matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer

# Load the dataset
df = pd.read_excel('C:/Users/Yash Dahima/PhD/Course Work/ML/Project/AQI/Datasets/data4.xlsx')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')

# Separate the features and the target variable
X = df.drop('pm2p5', axis=1)
y = df['pm2p5']

# Create a factor analyzer object and specify the number of factors to extract
n_factors = 3
fa = FactorAnalyzer(n_factors)

# Fit the factor analyzer on the data
fa.fit(X)

# Get the eigenvalues of the factors
ev, v = fa.get_eigenvalues()

# Create a scree plot to visualize the eigenvalues
plt.scatter(np.arange(1, X.shape[1]+1), ev)
plt.plot(range(1, X.shape[1]+1), ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()

# Apply the Kaiser criterion and select the number of factors to retain
n_factors_to_retain = len([i for i in ev if i > 1])

# Refit the factor analyzer with the selected number of factors
fa = FactorAnalyzer(n_factors_to_retain)
fa.fit(X)

# Get the factor loadings and the scores for each observation
loadings = fa.loadings_
scores = fa.transform(X)

# Add the target variable to the scores dataframe
scores_df = pd.DataFrame(scores, columns=['Factor '+str(i) for i in range(1, n_factors_to_retain+1)])
scores_df['PM2.5 concentration'] = df['pm2p5'].copy()

# Print the factor loadings
print('Factor Loadings:')
print(loadings)

# Print the scores for the first 5 observations
print('Factor Scores:')
print(scores_df.head())