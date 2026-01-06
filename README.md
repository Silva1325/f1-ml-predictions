# F1 ML Predictions

Machine learning models to predict Formula 1 race outcomes, including race winners and finishing positions.

## Project Structure

```
f1-ml-predictions/
├── .venv/                      # Virtual environment
├── catboost_info/              # CatBoost training logs
├── Data Processing/            # Data preparation scripts
├── Datasources/               # Raw and processed F1 data
├── Predict_race_position/     # Position prediction models
├── Predict_winner/            # Race winner prediction models
├── .gitignore
├── LICENSE
└── README.md
```

## Features

- **Race Winner Prediction**: Binary classification to predict which driver will win
- **Position Prediction**: Predict finishing positions (1-20) for all drivers
- Multiple ML algorithms: K-NN, Random Forest, Gradient Boosting, ANN, CatBoost Ranker

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/f1-ml-predictions.git
cd f1-ml-predictions

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn tensorflow catboost imbalanced-learn matplotlib seaborn
```

## Data

The project uses processed F1 race data with features:
- `year`, `round`, `circuitRef`
- `driverRef`, `constructorRef`
- `previousDriverPoints`, `previousConstructorPoints`
- `startGridPosition`, `avgOvertakes`
- `finishingPosition`, `winner`

Training data: 1950-2018  
Test data: 2019-2024

## Models

### Winner Prediction
- Binary classification (win/not win)
- Models: Logistic Regression, Random Forest, ANN

### Position Prediction
- Multi-class classification or regression
- Models: K-NN, Random Forest, Gradient Boosting, ANN, CatBoost Ranker
- Evaluation: Exact accuracy, MAE, Within ±N positions

## Usage

```bash
# Train winner prediction model
cd Predict_winner
python artificial_neural_network.py

# Train position prediction model
cd Predict_race_position
python artificial_neural_network.py

# Train CatBoost ranker
python catboost_ranker.py
```

## Results

Position prediction typical performance:
- Exact accuracy: 10-20%
- MAE: 3-5 positions
- Within ±3 positions: 40-60%

## License

See LICENSE file for details.
