import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Datasources
constructor_standings_ds = pd.read_csv(r'Datasources\constructor_standings.csv')
constructors_ds = pd.read_csv(r'Datasources\constructors.csv')
drivers_ds = pd.read_csv(r'Datasources\drivers.csv')
races_ds = pd.read_csv(r'Datasources\races.csv')
results_ds = pd.read_csv(r'Datasources\results.csv')

# Merging Datasources
merged_df = results_ds.merge(races_ds, on='raceId', how='left')
merged_df = merged_df.merge(drivers_ds, on='driverId', how='left')
merged_df = merged_df.merge(constructors_ds, on='constructorId', how='left')
merged_df = merged_df.merge(constructor_standings_ds, on=['raceId', 'constructorId'], how='left')

# Rename Columns
merged_df.rename(columns={
    'points_x': 'driverPoints',
    'points_y': 'constructorPoints',
    'grid': 'startGridPosition',
    'positionOrder': 'finishingPosition',
    'name_x': 'circuitRef',
}, inplace=True)


# Get relevant columns
relevant_columns = [
    'year', 'round',
    'raceId',
    'circuitId', 'circuitRef',
    'constructorId', 'constructorRef', 'constructorPoints',
    'driverId','driverRef', 'driverPoints', 'startGridPosition', 'finishingPosition',
]

merged_df = merged_df[relevant_columns]

# Drop NaN values
merged_df.dropna(inplace=True)

# Sort by year, round to ensure chronological order
merged_df = merged_df.sort_values(['year', 'round']).reset_index(drop=True)

# Calculate previous driver points
merged_df['previousDriverPoints'] = merged_df.groupby(['driverId', 'year'])['driverPoints'].shift(1)

# Calculate previous constructor points
merged_df['previousConstructorPoints'] = merged_df.groupby(['constructorId', 'year'])['constructorPoints'].shift(1)

# Fill NaN values (first race of season) with 0
merged_df['previousDriverPoints'] = merged_df['previousDriverPoints'].fillna(0)
merged_df['previousConstructorPoints'] = merged_df['previousConstructorPoints'].fillna(0)

# Calculate positions gained for current race (for building historical data)
merged_df['positionsGainedCurrent'] = merged_df['startGridPosition'] - merged_df['finishingPosition']

# Calculate average overtakes from PREVIOUS races only (expanding mean excluding current race)
merged_df['avgOvertakes'] = merged_df.groupby(['driverId', 'circuitId'])['positionsGainedCurrent'].apply(
    lambda x: x.shift(1).expanding().mean()
).reset_index(level=[0, 1], drop=True)

# Fill NaN (first time at circuit) with 0 or overall driver average
merged_df['avgOvertakes'] = merged_df['avgOvertakes'].fillna(0)

# Drop columns
merged_df.drop('positionsGainedCurrent', axis=1, inplace=True)
merged_df.drop('driverPoints', axis=1, inplace=True)
merged_df.drop('constructorPoints', axis=1, inplace=True)

# Add winner column
merged_df['winner'] = (merged_df['finishingPosition'] == 1).astype(int)

# Save the processed data to a new CSV file
merged_df.to_csv(r'Datasources\Generated\processed_f1_data.csv', index=False)

# Uncomment to verify the results
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(merged_df[['year', 'round', 'driverRef', 'circuitRef',
                     'avgOvertakes', 'startGridPosition', 'finishingPosition']].head(50))