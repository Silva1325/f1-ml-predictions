import pandas as pd
from catboost import CatBoostRanker, Pool
from sklearn.metrics import accuracy_score

# Importing the dataset
dataset = pd.read_csv(r'Datasources\Generated\processed_f1_data.csv')

# Prepare data for ranking
train_df = dataset[dataset['year'] < 2019].copy()
test_df = dataset[dataset['year'] >= 2019].copy()

# Create race IDs (group identifier)
train_df['raceId'] = train_df['year'].astype(str) + '_' + train_df['round'].astype(str)
test_df['raceId'] = test_df['year'].astype(str) + '_' + test_df['round'].astype(str)

features = [
    'year', 'round', 'circuitRef', 'constructorRef', 
    'previousConstructorPoints', 'driverRef', 'previousDriverPoints', 
    'startGridPosition', 'avgOvertakes'
]

# For ranking, use finishing position as target (or create relevance scores)
# If winner=1/0, convert to relevance: winner gets score 1, others get 0
train_df['relevance'] = train_df['winner'].astype(int)
test_df['relevance'] = test_df['winner'].astype(int)

# Sort by raceId to ensure groups are contiguous
train_df = train_df.sort_values('raceId')
test_df = test_df.sort_values('raceId')

X_train, y_train = train_df[features], train_df['relevance']
X_test, y_test = test_df[features], test_df['relevance']

# FIXED: Create group IDs as arrays (one per sample, not group sizes)
train_group_ids = train_df['raceId'].values
test_group_ids = test_df['raceId'].values

# Create Pool objects (CatBoost's data structure)
train_pool = Pool(
    data=X_train,
    label=y_train,
    group_id=train_group_ids,  # Array of race IDs, one per driver
    cat_features=['circuitRef', 'constructorRef', 'driverRef']
)

test_pool = Pool(
    data=X_test,
    label=y_test,
    group_id=test_group_ids,  # Array of race IDs, one per driver
    cat_features=['circuitRef', 'constructorRef', 'driverRef']
)

# Train CatBoostRanker
ranker = CatBoostRanker()

ranker.fit(train_pool)

# Predict
predictions = ranker.predict(test_pool)

# For each race, the driver with highest prediction score is predicted winner
test_df['predicted_score'] = predictions
test_df['predicted_winner'] = test_df.groupby('raceId')['predicted_score'].transform(
    lambda x: (x == x.max()).astype(int)
)

# Evaluate
accuracy = (test_df['winner'] == test_df['predicted_winner']).mean()
print(f"\nRace Winner Prediction Accuracy: {accuracy*100:.2f}%")

# Additional metrics
print(f"\nTotal races in test set: {test_df['raceId'].nunique()}")
print(f"Correctly predicted winners: {(test_df['winner'] == test_df['predicted_winner']).sum() // 2}")  # Divide by 2 because both actual and predicted winners are counted