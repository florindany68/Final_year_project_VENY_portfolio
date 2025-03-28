import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from lime.lime_tabular import LimeTabularExplainer
import time

processed_data = pd.read_csv('/home/danny/Desktop/processed_data23.csv')
id_columns  = ['Industry', 'Sector', 'Company','Symbol']
exclude_columns = id_columns + ['Risk Category', 'Risk Score']

features = [columns for columns in processed_data.columns if columns not in exclude_columns ]

X = processed_data[features]
Y = processed_data['Risk Category']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid_svm = {
    'C' : [0.1,1,10, 100],
    'gamma' : [ 0.1, 1, 10],
    'kernel' : ['poly'],
}
svm_model = SVC(class_weight='balanced',kernel='poly', C=1.0, gamma='scale', probability=True, random_state=42)
start_time = time.time()
grid_tuning = RandomizedSearchCV(estimator=svm_model, param_distributions=param_grid_svm, refit = True, verbose=3,cv=5,n_jobs=-1,
                                 n_iter = 10)
grid_tuning.fit(X_train_scaled, y_train)
end_time = time.time()
best_svm_model = grid_tuning.best_estimator_
y_pred = best_svm_model.predict(X_test_scaled)
y_pred_train = best_svm_model.predict(X_train_scaled)

training_time = end_time - start_time
print(f"Training time: {training_time}")
# Prediction probability class
y_pred_probability = best_svm_model.predict_proba(X_test_scaled)

results_for_user = pd.DataFrame(index=X_test.index)

for col in id_columns:
    results_for_user[col] = processed_data.loc[X_test.index, col].values
results_for_user['Actual Risk Score'] = y_test.values
results_for_user['Predicted Score'] = y_pred

categorisation_risk = best_svm_model.classes_
for company, category in enumerate(categorisation_risk):
    results_for_user[f"Probability : {category}"] = y_pred_probability[:, company]

# Save to CSV
results_for_user.to_csv('resultat_clasificare_companii3.csv', index=False)


for category in ['Low','Medium','High']:
    companies = results_for_user[results_for_user['Predicted Score'] == category]
    print(f"\n{category} Risk Companies ({len(companies)} total):")
    if not companies.empty:
        print(companies[['Company', 'Symbol', 'Industry', 'Actual Risk Score']].head(5))
    else:
        print("No companies in this category")
"""

The next part is for initiating LimeTabularExplainer library for showcasing the importance of the features. 

"""
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
    predict_fn = best_svm_model.predict_proba,
    num_features=len(features),
)
print(f"Company: {results_for_user.loc[X_test.index[instance_index], 'Company']}\n")
plt.figure(figsize=(10, 6))
explanation.as_pyplot_figure()
plt.title(f"LIME Explanation for {results_for_user.loc[X_test.index[instance_index]]}")
plt.tight_layout()
plt.savefig(f"lime_explanation_company.png")
plt.show()

feature_contributions = explanation.as_list()
print("\nFeature contributions:")
for feature, contribution in feature_contributions:
    print(f"{feature}: {contribution}")
print("\n------------------------------------------\n")



# Evaluate model
print("\nModel Prediction")

print(f"Accuracy testing: {accuracy_score(y_test, y_pred):.4f}")
print(f"Accuracy training: {accuracy_score(y_train, y_pred_train)}")



cm = confusion_matrix(y_test, y_pred)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(Y),
            yticklabels=np.unique(Y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SVM Risk Classification Confusion Matrix')
plt.savefig('svm_confusion_matrix15.png')
print("Saved confusion matrix to 'svm_confusion_matrix15.png'")

# Save the model and scaler
joblib.dump(svm_model, 'simple_svm_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')

print("\nModel and scaler saved")
