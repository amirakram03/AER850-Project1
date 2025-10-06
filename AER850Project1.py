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

coord_train, coord_test, target_train, target_test = train_test_split(coord,target,random_state = 74, test_size = 0.2,stratify=target);


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

"""2.4 Classification Model Development/Engineering"""

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

# ---------- Model 1: Logistic Regression (GridSearchCV)
pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000, random_state=74))
])

param_grid_lr = {
    "lr__C": [0.01, 0.1, 1, 10],
    "lr__solver": ["lbfgs", "newton-cg", "saga"],
    "lr__penalty": ["none","l2"],
}

grid_lr = GridSearchCV(pipe_lr,param_grid_lr, cv=5, scoring="f1_weighted", n_jobs=-1)
grid_lr.fit(coord_train, target_train)
best_lr = grid_lr.best_estimator_

print("=== Logistic Regression (GridSearchCV) ===")
print("Best params:", grid_lr.best_params_)
print(f"Best CV mean accuracy: {grid_lr.best_score_:.4f}\n")


# ---------- Model 2: Random Forest (GridSearchCV)
rf = RandomForestClassifier(random_state=74)

param_grid_rf = {
    'n_estimators': [5, 10, 30, 50],
     'max_depth': [None, 5, 15, 45],
     'min_samples_split': [2, 5, 10],
     'min_samples_leaf': [1, 2, 4, 6],
     'max_features': [None,0.1,'sqrt', 'log2', 1, 2, 3],
     'criterion': ['gini', 'entropy']         
}
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring="f1_weighted", n_jobs=-1)

grid_rf.fit(coord_train, target_train)
best_rf = grid_rf.best_estimator_   
print("=== Random Forest (GridSearchCV) ===")
print("Best params:", grid_rf.best_params_)  
print(f"Best CV mean F1 (weighted): {grid_rf.best_score_:.4f}\n")




# ---------- Model 3: Support Vector Machine (GridSearchCV)
pipe_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(random_state=74))
])

param_grid_svm = {
    "svm__kernel": ["linear", "rbf", "poly"],
    "svm__C": [0.001,0.01,0.1, 1, 10],
    "svm__gamma": ['scale','auto',10,100],             
}

grid_svm = GridSearchCV(pipe_svm, param_grid_svm, cv=5, scoring="f1_weighted", n_jobs=-1,)

grid_svm.fit(coord_train, target_train)
best_svm = grid_svm.best_estimator_

print("=== SVM (GridSearchCV) ===")
print("Best params:", grid_svm.best_params_)
print(f"Best CV mean F1 (weighted): {grid_svm.best_score_:.4f}\n")





# ---------- Model 4: Random Forest (RandomizedSearchCV)

rf2 = RandomForestClassifier(random_state=74)

param_grid_rf2 = {
    'n_estimators': [5, 10, 15, 20, 25, 30, 50],
      'max_depth': [None, 5,10, 15, 25, 45],
      'min_samples_split': [2, 5, 10, 15],
      'min_samples_leaf': [1, 2, 4, 6, 12],
      'max_features': [None,0.1,'sqrt', 'log2', 1, 2, 3],
      'criterion': ['gini', 'entropy']
  }

grid_rf2 = RandomizedSearchCV(rf2, param_grid_rf2, cv=5, scoring="f1_weighted", n_jobs=-1)

grid_rf2.fit(coord_train, target_train)
best_rf2 = grid_rf2.best_estimator_

print("=== Random Forest (RandomizedSearchCV) ===")
print("Best params:", grid_rf2.best_params_)
print(f"Best CV mean F1 (weighted): {grid_rf2.best_score_:.4f}\n")
























