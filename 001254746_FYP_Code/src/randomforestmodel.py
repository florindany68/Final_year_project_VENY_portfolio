import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler #library for data preprocessing
from sklearn.ensemble import  RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import time
import numpy as np
import matplotlib.pyplot as plt
import joblib

file_path = '/home/danny/Desktop/processed_datafinal6.csv'
processed_data = pd.read_csv(file_path)
file_shape = processed_data.shape
id_columns  = ['Industry', 'Sector', 'Company','Symbol']
exclude_columns = id_columns + ['Risk Category', 'Risk Score']

features = [columns for columns in processed_data.columns if columns not in exclude_columns ]

df_prepared = processed_data.dropna()

X = processed_data[features]
Y = processed_data['Risk Category']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# the hyperparameters used in tuning for Random Forest with an extended search space
param_grid_rf = {
    'n_estimators': [7, 10, 11,],    #number of trees in the forest
    'max_depth': [2, 4, 6, 8],    #depth of each tree
    'min_samples_split': [2, 5, 7,], #min number of samples for splitting
    'min_samples_leaf': [2, 3, 4]    #min number of samples in the node
}

#initialize RandomForestRegressor model
rf_model = RandomForestClassifier(random_state=42)

#record the start time
start_time = time.time()

#GridSearchCV is used for finding the best hyperparameters
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train_scaled, y_train)

#record the end time
end_time = time.time()

#calculate the training time of the model
training_time = end_time - start_time
print(f'Training Time: {training_time} seconds')

#get the best Random Forest model from the grid search function
best_rf_model = grid_search_rf.best_estimator_

#make predictions based on the test set
y_pred = best_rf_model.predict(X_test_scaled)
y_pred_train = best_rf_model.predict(X_train_scaled)
y_pred_probability = best_rf_model.predict_proba(X_test_scaled)



