# Stock Prediction Using Machine Learning

This project demonstrates how machine learning algorithms can be applied to predict the future stock prices of Apple Inc. (AAPL) using historical data. The project utilizes various regression models to predict the stock prices, including Decision Tree Regressor, Linear Regression, Support Vector Machine (SVM), and AdaBoost Regressor.

## Dataset

The dataset used in this project consists of historical stock data for Apple Inc. (AAPL) obtained from Yahoo Finance. The data includes the following features:

- **Date**: The date of the stock data.
- **Close**: The closing price of the stock on that date.

The dataset is retrieved using the `yfinance` Python library and spans from **2020-01-01** to **2023-12-31**.

### Feature Engineering
Several technical indicators and features are created to enhance the model:

- **MA_5**: 5-day moving average of stock prices.
- **MA_10**: 10-day moving average of stock prices.
- **Pct_Change**: Percentage change in the stock price.
- **Lag_1**: Previous day's closing price (lag feature).

## Steps

### 1. Data Collection & Preprocessing:
- **Data Retrieval**: Stock data for AAPL is downloaded using the `yfinance` library.
- **Feature Creation**: Moving averages, percentage change, and lag features are created.
- **Data Cleaning**: The missing values are handled by removing rows with NaN values.
- **Data Scaling**: Standardization of features using `StandardScaler` to bring them to a comparable scale.

### 2. Model Training & Evaluation:
The following machine learning models are trained and evaluated:

- **Decision Tree Regressor (Decision Stump)**: A simple decision tree model with a maximum depth of 1.
- **Linear Regression**: A simple linear regression model for price prediction.
- **Support Vector Machine (SVM)**: A regression model using a radial basis function (RBF) kernel.
- **AdaBoost Regressor**: An ensemble learning method based on boosting.

Each model is evaluated based on:
- **Correlation Coefficient**: Measures the relationship between actual and predicted values.
- **Mean Absolute Error (MAE)**: Measures the average magnitude of errors in prediction.
- **Root Mean Squared Error (RMSE)**: Measures the average magnitude of the error (penalizes large errors).
- **R-Squared (Accuracy)**: Measures the proportion of the variance in the dependent variable that is predictable from the independent variables.

The results are averaged over 10 iterations to ensure robust performance evaluation.

### 3. Results:
The results include the performance of each model with the following metrics:
- **Correlation Coefficient**: How closely the predicted and actual values correlate.
- **MAE**: Mean Absolute Error.
- **RMSE**: Root Mean Squared Error.
- **Accuracy**: R-squared score (measured as a percentage of variance explained).

## Requirements

To run this project, you need to install the following Python libraries:

- `yfinance`: To fetch historical stock data from Yahoo Finance.
- `pandas`: For data manipulation and processing.
- `numpy`: For numerical operations.
- `sklearn`: For machine learning models and metrics.
- `matplotlib`: For data visualization (optional).Decision Stump:
- output example:
Correlation coefficient: 0.9483
MAE: 2.3842
RMSE: 3.7816
Accuracy: ~94.83%

Linear Regression:
Correlation coefficient: 0.9712
MAE: 1.8593
RMSE: 2.9241
Accuracy: ~97.12%

SVM (C-Class):
RMSE: 3.2124 +/- 0.5494
Accuracy: 96.54% +/- 1.12%

Boosting (AdaBoostM1):
RMSE: 2.8457 +/- 0.5682
Accuracy: 96.91% +/- 1.23%


You can install the dependencies using `pip`:

```bash
pip install yfinance pandas numpy scikit-learn matplotlib
