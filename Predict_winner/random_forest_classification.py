import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
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

X_train, y_train = train_df[features], train_df['winner']
X_test, y_test = test_df[features], test_df['winner']

# 3. Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 4. Training the Random Forest Classification model on the Training set
classifier = RandomForestClassifier(
    criterion='gini',
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=2,
    min_samples_split=5,
    class_weight='balanced_subsample',
    random_state=42
)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred_proba = classifier.predict_proba(X_test)[:, 1]

# Making the Confusion Matrix
#
# [2359   72]
# [  66   62]
#
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print(f"Accuracy: {accuracy*100:.2f}%")

# Applying k-Fold Cross Validation
# Cross-Validation Accuracy: 95.15% (+/- 0.56%)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(f"Cross-Validation Accuracy: {accuracies.mean()*100:.2f}% (+/- {accuracies.std()*100:.2f}%)")

# Applying Grid Search CV to find the best model and the best parameters
#parameters = [
#    {
#        'n_estimators': [100, 150, 200, 250, 300],
#        'criterion': ['gini', 'entropy'],
#        'max_depth': [20, 25, 30],
#        'min_samples_split': [2, 5],
#        'min_samples_leaf': [1, 2],
#        'class_weight': ['balanced', 'balanced_subsample', None]
#    }
#]

#
# Best Parameters: {'class_weight': 'balanced_subsample', 'criterion': 'gini', 'max_depth': 20, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100}
# Best Cross-Validation Score: 27.41%
#
#grid_search = GridSearchCV(
#    estimator=RandomForestClassifier(random_state=0),
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
