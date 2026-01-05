import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

# Importing the dataset
dataset = pd.read_csv(r'Datasources\Generated\processed_f1_data.csv')

# Use LabelEncoding instead of OneHot for 800+ drivers to avoid "feature explosion"
le_driver = LabelEncoder()
le_constructor = LabelEncoder()
le_circuit = LabelEncoder()

dataset['driverRef'] = le_driver.fit_transform(dataset['driverRef'])
dataset['constructorRef'] = le_constructor.fit_transform(dataset['constructorRef'])
dataset['circuitRef'] = le_circuit.fit_transform(dataset['circuitRef'])

# Train on 1950-2018, Test on 2019-2024 for a realistic simulation
train_df = dataset[dataset['year'] < 2019]
test_df = dataset[dataset['year'] >= 2019]

features = [
    'year', 'round', 'circuitRef', 'constructorRef', 
    'previousConstructorPoints', 'driverRef', 'previousDriverPoints', 
    'startGridPosition', 'avgOvertakes'
]

# Setting up training and testing data
X_train, y_train = train_df[features], train_df['winner']
X_test, y_test = test_df[features], test_df['winner']

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the K-NN model on the Training set
classifier = KNeighborsClassifier(weights='distance',)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred_proba = classifier.predict_proba(X_test)[:, 1]

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print(f"Accuracy: {accuracy*100:.2f}%")

# Applying k-Fold Cross Validation
# Cross-Validation Accuracy (Default values) : 95.33% (+/- 0.24%)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(f"Cross-Validation Accuracy: {accuracies.mean()*100:.2f}% (+/- {accuracies.std()*100:.2f}%)")

# Applying Grid Search CV to find the best model and the best parameters
#parameters = [
#    {
#        'n_neighbors': [3, 5, 7, 9, 11, 15],
#        'weights': ['uniform', 'distance'],
#        'metric': ['euclidean', 'manhattan', 'minkowski'],
#        'p': [1, 2]
#    }
#]

#
# Best Parameters: {'metric': 'manhattan', 'n_neighbors': 3, 'p': 1, 'weights': 'distance'}
# Best Cross-Validation Score: 16.67%
#
#grid_search = GridSearchCV(
#    estimator=KNeighborsClassifier(),
#    param_grid=parameters,
#    scoring='f1',
#    cv=10,
#    n_jobs=-1,
#    verbose=2
#)

#print("Starting Grid Search CV...")
#grid_search.fit(X_train, y_train)

# Best parameters and score
#print("\n" + "="*60)
#print("GRID EARCH RESULTS")
#print("="*60)
#print(f"Best Parameters: {grid_search.best_params_}")
#print(f"Best Cross-Validation Score: {grid_search.best_score_*100:.2f}%")
#print("="*60)
