import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC

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

# Training the Support Vector Machine model on the Training set
classifier = SVC(kernel = 'linear')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print(f"Accuracy: {accuracy*100:.2f}%")

# Applying k-Fold Cross Validation
# Cross-Validation Accuracy (Default values) : 95.84% (+/- 0.00%)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(f"Cross-Validation Accuracy: {accuracies.mean()*100:.2f}% (+/- {accuracies.std()*100:.2f}%)")
