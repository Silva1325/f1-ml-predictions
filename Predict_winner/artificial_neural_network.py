import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
from imblearn.over_sampling import SMOTE

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

# Handling class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Artificial Neural Network model on the Training set
ann = tf.keras.models.Sequential([
    # Input layer + First hidden layer
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=(X_train_balanced.shape[1],)),
    tf.keras.layers.BatchNormalization(),  # Normalize activations
    tf.keras.layers.Dropout(0.3),  # Prevent overfitting
    
    # Second hidden layer
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    # Third hidden layer
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    
    # Output layer
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

ann.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

ann.fit(X_train_balanced, y_train_balanced, batch_size = 32, epochs = 100)

# Predict the Test Results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)

# Convert y_test to numpy array before reshaping
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.values.reshape(len(y_test), 1)), 1))

# Making the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

ac = accuracy_score(y_test, y_pred)
print(ac)

# Applying k-Fold Cross Validation
# Cross-Validation Accuracy (Default values) : 95.84% (+/- 0.00%)
accuracies = cross_val_score(estimator = ann, X = X_train_balanced, y = y_train_balanced, cv = 10)
print(f"Cross-Validation Accuracy: {accuracies.mean()*100:.2f}% (+/- {accuracies.std()*100:.2f}%)")