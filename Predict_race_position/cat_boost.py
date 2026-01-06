import pandas as pd
import numpy as np
from catboost import CatBoostRanker, Pool
from sklearn.metrics import accuracy_score, mean_absolute_error

# Importing the dataset
dataset = pd.read_csv(r'Datasources\Generated\processed_f1_data.csv')

# Prepare data for ranking
train_df = dataset[dataset['year'] < 2019].copy()
test_df = dataset[dataset['year'] >= 2019].copy()

# Filter to positions 1-20
train_df = train_df[(train_df['finishingPosition'] >= 1) & (train_df['finishingPosition'] <= 20)]
test_df = test_df[(test_df['finishingPosition'] >= 1) & (test_df['finishingPosition'] <= 20)]

# Create race IDs (group identifier)
train_df['raceId'] = train_df['year'].astype(str) + '_' + train_df['round'].astype(str)
test_df['raceId'] = test_df['year'].astype(str) + '_' + test_df['round'].astype(str)

features = [
    'year', 'round', 'circuitRef', 'constructorRef', 
    'previousConstructorPoints', 'driverRef', 'previousDriverPoints', 
    'startGridPosition', 'avgOvertakes'
]

# Convert finishing position to relevance score (1st=20, 2nd=19, ..., 20th=1)
train_df['relevance'] = 21 - train_df['finishingPosition']
test_df['relevance'] = 21 - test_df['finishingPosition']

# Sort by raceId
train_df = train_df.sort_values('raceId')
test_df = test_df.sort_values('raceId')

X_train, y_train = train_df[features], train_df['relevance']
X_test, y_test = test_df[features], test_df['relevance']

# Create group IDs
train_group_ids = train_df['raceId'].values
test_group_ids = test_df['raceId'].values

# Create Pool objects
train_pool = Pool(
    data=X_train,
    label=y_train,
    group_id=train_group_ids,
    cat_features=['circuitRef', 'constructorRef', 'driverRef']
)

test_pool = Pool(
    data=X_test,
    label=y_test,
    group_id=test_group_ids,
    cat_features=['circuitRef', 'constructorRef', 'driverRef']
)

# Train CatBoostRanker
ranker = CatBoostRanker()
ranker.fit(train_pool)

# Predict
predictions = ranker.predict(test_pool)

# Rank drivers by predicted score
test_df['predicted_score'] = predictions
test_df['predicted_position'] = test_df.groupby('raceId')['predicted_score'].rank(
    ascending=False, method='first'
).astype(int)

# Evaluate
accuracy = (test_df['finishingPosition'] == test_df['predicted_position']).mean()
mae = mean_absolute_error(test_df['finishingPosition'], test_df['predicted_position'])

print(f"\nExact Position Accuracy: {accuracy*100:.2f}%")
print(f"Mean Absolute Error: {mae:.2f} positions")

# 13.01%