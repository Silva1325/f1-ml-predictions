# 🏎️ F1 ML Predictions

Machine learning models to predict Formula 1 race outcomes, including race winners and finishing positions.

## 📁 Project Structure
```
f1-ml-predictions/
├── catboost_info/              # CatBoost training logs
├── Data Processing/            # Data preparation scripts
├── Datasources/               # Raw and processed F1 data
├── Predict_race_position/     # Position prediction models
├── Predict_winner/            # Race winner prediction models
├── .gitignore
├── LICENSE
└── README.md
```

## ✨ Features

- **🏆 Race Winner Prediction**: Binary classification to predict which driver will win
- **📊 Position Prediction**: Predict finishing positions (1-20) for all drivers
- **🤖 Multiple ML algorithms**: K-NN, Random Forest, Gradient Boosting, ANN, CatBoost Ranker

## 🚀 Installation
```bash
# Clone repository
git clone https://github.com/yourusername/f1-ml-predictions.git
cd f1-ml-predictions

# Install dependencies
pip install pandas numpy scikit-learn tensorflow catboost imbalanced-learn matplotlib seaborn
```

## 📦 Data

### Dataset Source
This project uses F1 data from Kaggle:  
[Formula 1 World Championship (1950-2020)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)

**📅 Training Period**: 1950-2018  
**🧪 Test Period**: 2019-2024

### Data Processing Pipeline

The raw F1 data undergoes several processing steps:

#### 1. Merging Datasources
Combines 5 separate datasets into one complete dataset:
- Results
- Races
- Drivers
- Constructors
- Constructor Standings

#### 2. Feature Engineering

After extensive experimentation, including testing weather data and other variables, three key features emerged as the most impactful for prediction accuracy:

**previousDriverPoints**
- Championship points accumulated by the driver before the current race
- Captures current season performance and momentum
- Reset to 0 at the start of each season
- Uses `shift(1)` to prevent data leakage

**previousConstructorPoints**
- Championship points accumulated by the team before the current race
- Reflects car performance and team competitiveness throughout the season
- Accounts for technical development and reliability

**avgOvertakes**
- Average positions gained/lost at specific circuits from previous years
- Calculated as: `startGridPosition - finishingPosition`
- Circuit-specific metric showing driver's historical performance at each track
- Only uses data from previous races at that circuit
- Captures driver skill on different track layouts (street circuits vs. high-speed tracks)

> **🌦️ Note on Weather Data**: Meteorological features were tested but showed minimal impact on prediction accuracy. The three features above proved most relevant for model performance.

#### 3. Data Cleaning
- ✅ Removes incomplete records
- 📅 Sorts chronologically by year and round
- 🏷️ Renames columns for clarity
- 🗑️ Drops intermediate calculation columns

#### 4. Target Variables
- **finishingPosition**: Race finish position (1-20) for position prediction
- **winner**: Binary flag (1/0) for winner prediction

> **🔒 Key Principle**: All features use only information available **before** the race to prevent data leakage.

## 🤖 Models

### Winner Prediction
- **Type**: Binary classification (win/not win)
- **Algorithms**: Logistic Regression, Random Forest, ANN

### Position Prediction
- **Type**: Multi-class classification or regression
- **Algorithms**: K-NN, Random Forest, Gradient Boosting, ANN, CatBoost Ranker
- **📏 Evaluation Metrics**: 
  - Exact accuracy
  - Mean Absolute Error (MAE)
  - Within ±N positions

## 💻 Usage
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

## 📈 Results

### 🏆 Winner Prediction Performance

| Model | Accuracy |
|-------|----------|
| **Random Forest** ✅ | **95.57% (±0.32%)** |
| K-Nearest Neighbors | 95.33% (±0.24%) |
| CatBoost | 95.08% |
| Naive Bayes | 92.21% (±6.77%) |
| Kernel SVM | 81.04% (±5.68%) |
| Logistic Regression | 75.83% (±6.24%) |
| Support Vector Machine | 73.86% (±5.67%) |
| Artificial Neural Network | 0.95% |

### 📊 Position Prediction Performance

| Model | Accuracy |
|-------|----------|
| **Artificial Neural Network** ✅ | **14.97%** |
| CatBoost | 13.01% |
| Kernel SVM | 10.13% (±1.05%) |
| Logistic Regression | 9.18% (±1.24%) |
| Random Forest | 7.36% (±2.61%) |
| K-Nearest Neighbors | 7.32% (±0.83%) |
| Support Vector Machine | 7.21% (±2.64%) |
| Naive Bayes | 6.26% (±1.89%) |

### 💡 Conclusion

After comprehensive model comparison with default hyperparameters:

**🏆 Winner Prediction**: **Random Forest** emerged as the best performer with 95.57% accuracy and low variance (±0.32%), demonstrating excellent consistency across cross-validation folds. K-Nearest Neighbors and CatBoost were close competitors, but Random Forest's stability makes it the optimal choice.

**📊 Position Prediction**: **Artificial Neural Network** achieved the highest accuracy at 14.97%, significantly outperforming traditional machine learning methods. CatBoost came second at 13.01%. The relatively low accuracy rates (10-15%) reflect the inherent difficulty of predicting exact finishing positions in F1, where 20 possible outcomes and high race unpredictability make this a challenging multi-class classification problem.

> **Key Insight**: While predicting exact positions remains challenging, these models still provide valuable insights. For practical applications, metrics like "Within ±3 positions" (40-60% accuracy) offer more realistic performance expectations.
