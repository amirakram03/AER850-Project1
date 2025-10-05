#Name: Amir Akram
#Title: AER850 Project 1

#Basic Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""2.1 Data Processing"""
df=pd.read_csv("Project 1 Data.csv")
# print(df.info())
# print(df.head())
#print(df.describe())

#Splitting the data set into train and test using the 80-20 split convention and stratifying the samples
#This is done before dta visualization to avoid data snooping bias

from sklearn.model_selection import train_test_split

#Identifying x and y variables
coord=df[['X','Y','Z']] #features
target=df['Step']; #Labels

coord_train, coord_test, target_train, target_test = train_test_split(coord,target,random_state = 42, test_size = 0.2,stratify=target);


"""2.2 Data Visualization"""

#3D Scatter Plot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
          '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf',
          '#a55194','#393b79','#637939']  # 13 distinct colors

cmap = ListedColormap(colors)
sc = ax.scatter(coord_train['X'], coord_train['Y'], coord_train['Z'], c=target_train, cmap=cmap)
cbar = plt.colorbar(sc, label='Step')
cbar.set_ticks(np.arange(1, 14))       # 13 steps
cbar.set_ticklabels(np.arange(1, 14))  # 1..13 labels

ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_zlabel("Z Coordinate")
plt.title("3D Scatter Plot (Training Data Only)")
ax.view_init(30,70);

#2D scattor plot for each feature combination and Histrograms
train_df = coord_train.copy()
train_df['Step'] = target_train.values

# Scatter matrix of training data
cmap = ListedColormap(colors)
pd.plotting.scatter_matrix(train_df, c=train_df['Step'], cmap=cmap)

"""2.3 Correlation Analysis"""
import seaborn as sns
plt.figure()
corr_matrix = train_df.corr(method='pearson') # Compute Pearson correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True) # Plot the correlation heatmap
plt.title("Correlation Between Features (X, Y, Z) and Step")





