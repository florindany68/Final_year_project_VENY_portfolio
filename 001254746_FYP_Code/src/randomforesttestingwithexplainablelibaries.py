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

file_path = '/home/danny/Desktop/001254746_FYP_Code/data/processed_file.csv'
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



cm = confusion_matrix(y_test, y_pred)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(20, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(Y),
            yticklabels=np.unique(Y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Risk Classification Confusion Matrix')
plt.savefig('rf_confuision_matrix1.png')
print("Saved confusion matrix'")
results_for_user = pd.DataFrame(index=X_test.index)



for col in id_columns:
    results_for_user[col] = processed_data.loc[X_test.index, col].values
results_for_user['Actual Risk Score'] = y_test.values
results_for_user['Predicted Score'] = y_pred

categorisation_risk = best_rf_model.classes_
for company, category in enumerate(categorisation_risk):
    results_for_user[f"Probability : {category}"] = y_pred_probability[:, company]

# Save to CSV
results_for_user.to_csv('resultat_clasificare_companii_RandomForest3.csv', index=False)

# Save the model and scaler
joblib.dump(best_rf_model, 'RandomForest_modelMax.pkl')
joblib.dump(scaler, 'feature_scalerRF_Max.pkl')

for category in ['Low','Medium','High']:
    companies = results_for_user[results_for_user['Predicted Score'] == category]
    print(f"\n{category} Risk Companies ({len(companies)} total):")
    if not companies.empty:
        print(companies[['Company', 'Symbol', 'Industry', 'Predicted Score']].head(5))
    else:
        print("No companies in this category")


print(f"\nAccuracy testing: {accuracy_score(y_test, y_pred):.4f}")
print(f"Accuracy score for training: {accuracy_score(y_train, y_pred_train):.4f}")


from lime.lime_tabular import LimeTabularExplainer
explainer = LimeTabularExplainer(
    training_data=X_train.values,
    #Trainiing_data value takes the training data of the model. It also converts the panda DataFrame into numpy array for compatibility
    feature_names=features, #This parameter takes the features from the model
    class_names=['Low','Medium','High'],#This parameter takes the target variables of the model
    mode='classification',
    discretize_continuous=True
)




instance_index = 1


instance = X_test_scaled[instance_index]

explanation = explainer.explain_instance(
    data_row=instance,
    predict_fn = best_rf_model.predict_proba,
    num_features=len(features),
)

print(f"Company: {results_for_user.loc[X_test.index[instance_index], 'Company']}\n")
plt.figure(figsize=(20, 30))
explanation.as_pyplot_figure()
plt.title(f"LIME Explanation for {results_for_user.loc[X_test.index[instance_index]]}")
plt.tight_layout()
plt.savefig(f"lime_explanation_companyRF.png")
plt.show()

feature_contributions = explanation.as_list()
print("\nFeature contributions:")
for feature, contribution in feature_contributions:
    print(f"{feature}: {contribution}")
print("\n------------------------------------------\n")




import shap


explainer = shap.Explainer(best_rf_model, X_test_scaled)
shap_values = explainer(X_test_scaled,check_additivity=False)

plt.figure(figsize = (20,20))
shap.summary_plot(shap_values,
                  feature_names=features,
                  max_display=20,
                  plot_type="bar",
                  class_names=['Low','Medium','High'],
                  show=False)
plt.tight_layout()
plt.title('SHAP Feature Importance Summary')
plt.legend(labels=['Low','Medium','High'])
plt.savefig('shap_summary_bar_plotrandomforest.png')
plt.show()




