import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv(r'Datasources\Generated\processed_f1_data.csv')

# Independent and dependent variables
independent_variables = [
    'year','round','circuitRef',
    'constructorRef','previousConstructorPoints',
    'driverRef','previousDriverPoints',
    'startGridPosition','avgOvertakes'
]
dependent_variable = 'winner'
X = dataset[independent_variables].values
y = dataset[dependent_variable].values

# Identify categorical and numerical columns
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), [2, 3, 5])],
    remainder='passthrough'
)
X = ct.fit_transform(X)

# Feature Scaling
sc = StandardScaler()
X = sc.fit_transform(X)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(
    n_estimators=200,
    criterion='entropy',
    random_state=0,
    class_weight='balanced',    
    max_depth=25,               
    min_samples_split=5,
    min_samples_leaf=2
)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred_proba = classifier.predict_proba(X_test)[:, 1]

# Making the Confusion Matrix
#
# [4158  617]
# [  33  171]
# 
# Accuracy: 86.95%
# 
# True Negatives (Non-winners correctly predicted): 4,747
# False Positives (Non-winners predicted as winners): 617
# False Negatives (Winners predicted as non-winners): 33
# True Positives (Winners correctly predicted): 171
# 
# 4,747 / (4,747 + 617) = 88.5% for non-winner class
# 171 / (171 + 617) = 21.7% for winner class  
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print(f"Accuracy: {accuracy*100:.2f}%")

# Applying k-Fold Cross Validation
# Cross-Validation Accuracy: 87.18% (+/- 0.77%)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(f"Cross-Validation Accuracy: {accuracies.mean()*100:.2f}% (+/- {accuracies.std()*100:.2f}%)")