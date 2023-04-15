# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 15:19:46 2023

@author: Yash Dahima
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns, numpy as np, os
from scipy.stats import pearsonr, spearmanr

# Load the dataset
df = pd.read_excel('C:/Users/Yash Dahima/PhD/Course Work/ML/Project/AQI/Datasets/data4.xlsx', parse_dates= True)
df.set_index('datetime', inplace=True)
# Check the first few rows of the dataset
print(df.head())

# Check the basic information about the dataset
print(df.info())

# Check the statistical summary of the dataset
print(df.describe())

# Check for missing values in the dataset
print(df.isnull().sum())

# Drop the data after 2019

# Visualize the distribution of each variable using histograms
df.hist(bins=50, figsize=(20,15))
plt.show()

#mask = np.zeros_like(df.corr(), dtype=np.bool)
#mask[np.triu_indices_from(mask)] = True

# Visualize the correlations between variables using a heatmap
sns.heatmap(df.corr(), cmap='coolwarm', vmin=-0.9, vmax=0.9, annot=True, fmt='.2f')
plt.show()

# Visualize the relationship between the target variable and the other variables using scatterplots
sns.pairplot(df, x_vars=['ws', 'wd', 'temp', 'dew_temp', 'pressure', 'wv', 'blh', 'bcaod550', 'duaod550', 'omaod550', 'ssaod550', 'suaod550', 'aod469', 'aod550', 'aod670', 'aod865', 'aod1240'], y_vars=['pm2p5'], height=7, aspect=0.7)
plt.show()


# , height=8, aspect=0.5


from sklearn.feature_selection import mutual_info_regression

X = df.copy()
y = X.pop("pm2p5")

def make_mi_scores(X, y):
    mi_scores = mutual_info_regression(X, y)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X, y)
mi_scores[::3]  # show a few features with their MI scores

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)


# Calculate Pearson correlation coefficient
corr_p, p_val_p = pearsonr(df['pm2p5'], df['pressure'])
print("Pearson correlation coefficient:", corr_p)
print("p-value:", p_val_p)

# Calculate Spearman correlation coefficient
corr_s, p_val_s = spearmanr(df['pm2p5'], df['pressure'])
print("Spearman correlation coefficient:", corr_s)
print("p-value:", p_val_s)









"""
CPCB

# Load the Excel file into a Pandas DataFrame
path = "E:/YASH/Meteorological/CPCB Maninagar/Maninagar/15 minutes/"
file_list = os.listdir(path)
os.chdir(path)

df = pd.DataFrame()

for i in range(len(file_list)):
    # Load the Excel file into a DataFrame, skipping the first 16 rows
    dff = pd.read_excel(file_list[i], header=16)
    # Append the DataFrame to the merged DataFrame
    df = pd.concat([df, dff], axis=1)

df.replace('None', np.nan, inplace=True)

# Check for missing values
print(df.isnull().sum())

df = df.loc[:, ['From Date','To Date','PM2.5','RH','WS','WD','BP','VWS']]

# Transpose the DataFrame
df_transposed = df.T

# Drop duplicate columns
df_transposed = df_transposed.drop_duplicates()

# Transpose the DataFrame back to the original orientation
df = df_transposed.T

"""