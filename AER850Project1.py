#Name: Amir Akram
#Title: AER850 Project 1

#Basic Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""2.1 Data Processing"""
df=pd.read_csv("Project 1 Data.csv") #Read CSV file into a Pandas DataFrame

print(df.head()) # Display first few rows of data to understand structure and sample values
print(df.describe()) # Show statistical summary (mean, std, min, max, quartiles)

#Splitting the data set into train and test using the 80-20 split convention and stratifying the samples
#This is done before dta visualization to avoid data snooping bias

from sklearn.model_selection import train_test_split

#Identifying x and y variables
coord=df[['X','Y','Z']] #features
target=df['Step']; #Labels

# Split dataset into training and testing subsets
coord_train, coord_test, target_train, target_test = train_test_split(coord,target,random_state = 42, test_size = 0.2,stratify=target);


"""2.2 Data Visualization"""

#3D Scatter Plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

# Create a 3D figure for visualizing spatial feature distribution
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# Define a set of 13 distinct colors for the 13 step classes
colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
          '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf',
          '#a55194','#393b79','#637939']  # 13 distinct colors

# Create a custom colormap using these colors
cmap = ListedColormap(colors)

# Scatter plot of training data in 3D space
sc = ax.scatter(coord_train['X'], coord_train['Y'], coord_train['Z'], c=target_train, cmap=cmap)
cbar = plt.colorbar(sc, label='Step') # Add colorbar showing corresponding step number
cbar.set_ticks(np.arange(1, 14))       # 13 steps
cbar.set_ticklabels(np.arange(1, 14))  # 1..13 labels

# Label axes
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_zlabel("Z Coordinate")

# Add title and set viewing angle
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
# Import libraries for model building, scaling, and hyperparameter tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ---------- Model 1: Logistic Regression (GridSearchCV)
# Pipeline used to include preprocessing (like scaling) since the model is sensitive to feature magnitudes.
pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000, random_state=42))
])

#parameter grid to be used for gridsearchcv
param_grid_lr = {
    "lr__C": [0.01, 0.1, 1, 10, 100], # Regularization strength
    "lr__solver": ["lbfgs", "newton-cg", "saga"], # Optimization algorithm
    "lr__penalty": [None,"l2"], # Regularization type
}

# Perform grid search cross-validation to find best parameters
grid_lr = GridSearchCV(pipe_lr,param_grid_lr, cv=5, scoring="f1_weighted", n_jobs=-1)
grid_lr.fit(coord_train, target_train)
best_lr = grid_lr.best_estimator_

print("=== Logistic Regression (GridSearchCV) ===")
print("Best params:", grid_lr.best_params_)


# ---------- Model 2: Random Forest (GridSearchCV)
rf = RandomForestClassifier(random_state=42)

# Define grid of hyperparameters for Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [2, 4, 6],
    'max_features': ['sqrt', 'log2'],
    'criterion': ['gini', 'entropy']         
}
# Perform grid search for Random Forest
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring="f1_weighted", n_jobs=-1)

grid_rf.fit(coord_train, target_train)
best_rf = grid_rf.best_estimator_   
print("=== Random Forest (GridSearchCV) ===")
print("Best params:", grid_rf.best_params_)  



# ---------- Model 3: Support Vector Machine (GridSearchCV)
# Pipeline used to include preprocessing (like scaling) since the model is sensitive to feature magnitudes.
pipe_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(random_state=42))
])
#parameter grid to be used for gridsearchcv
param_grid_svm = {
    "svm__kernel": ["rbf","sigmoid","poly"], # Kernel type
    "svm__C": [0.1, 1, 10, 100], # Regularization parameter
    "svm__gamma": ['scale','auto'],      # Kernel coefficient         
}

# Grid search for SVM
grid_svm = GridSearchCV(pipe_svm, param_grid_svm, cv=5, scoring="f1_weighted", n_jobs=-1,)

grid_svm.fit(coord_train, target_train)
best_svm = grid_svm.best_estimator_

print("=== SVM (GridSearchCV) ===")
print("Best params:", grid_svm.best_params_)





# ---------- Model 4: Decision Tree (RandomizedSearchCV)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)

# Parameter grid for Decision Tree using RandomizedSearch
param_grid_dt = {
     'max_depth': [None, 10, 20, 30],
     'min_samples_split': [2, 5, 10],
     'min_samples_leaf': [1, 2, 4],
     'max_features': ['sqrt', 'log2']
  }

# Randomized search to find optimal Decision Tree parameters
grid_dt = RandomizedSearchCV(dt, param_grid_dt, cv=5, scoring="f1_weighted", n_jobs=-1)

grid_dt.fit(coord_train, target_train)
best_dt = grid_dt.best_estimator_

print("=== Decision Tree (RandomizedSearchCV) ===")
print("Best params:", grid_dt.best_params_)






"""2.5 Model Performance Analysis"""
# ---------- Step 5: Model Performance Analysis ----------
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Store all trained models in a dictionary for evaluation
models = {
    "Logistic Regression": best_lr,
    "SVM": best_svm,
    "Random Forest (GridSearchCV)": best_rf,
    "Decision Tree (RandomizedSearchCV)": best_dt
}

# Evaluate each model
results = []

for name, model in models.items():
    y_pred = model.predict(coord_test)
    
    # Calculate key performance metrics
    acc = accuracy_score(target_test, y_pred)
    prec = precision_score(target_test, y_pred, average='weighted')
    rec = recall_score(target_test, y_pred, average='weighted')
    f1 = f1_score(target_test, y_pred, average='weighted')
    
    results.append([name, acc, prec, rec, f1])
    
    # Plot confusion matrix to visualize classification accuracy per class
    cm = confusion_matrix(target_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title(f"Confusion Matrix - {name}")
    plt.show()
    
# Summarize results for all models in a DataFrame
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
print("\n=== Model Performance Summary ===")
print(results_df)





"""2.6 Stacked Model Performance Analysis"""
from sklearn.ensemble import StackingClassifier
# Combine two strong models (SVM and Random Forest)
estimators = [
    ("svm", best_svm),
    ("rf", best_rf)
]

stacked_model = StackingClassifier(estimators, LogisticRegression(max_iter=1000, random_state=42), cv=5,n_jobs=-1)

# Train the stacked model
stacked_model.fit(coord_train, target_train)

# Test stacked model
y_pred_stack = stacked_model.predict(coord_test)

# Evaluate stacked model
acc = accuracy_score(target_test, y_pred_stack)
prec = precision_score(target_test, y_pred_stack, average="weighted")
rec = recall_score(target_test, y_pred_stack, average="weighted")
f1 = f1_score(target_test, y_pred_stack, average="weighted")

print("=== Stacked Model Performance ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}\n")

# Display confusion matrix
cm = confusion_matrix(target_test, y_pred_stack)
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues")
plt.title("Confusion Matrix – Stacked Model")
plt.tight_layout()
plt.show()





"""2.7 Model Evaluation"""
import joblib

# Save the final trained stacked model to disk
joblib.dump(stacked_model, "stacked_model.joblib")
print("Model saved as stacked_model.joblib")


loaded_model = joblib.load("stacked_model.joblib")

# Test model predictions on new unseen data points
new_points = np.array([
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0, 3.0625, 1.93],
    [9.4, 3, 1.8],
    [9.4, 3, 1.3]
])

# Display corresponding maintenance steps
predicted_steps = loaded_model.predict(new_points)
print("Predicted maintenance steps:", predicted_steps)


  






































