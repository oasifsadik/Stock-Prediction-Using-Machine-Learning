import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

stock_data = yf.download("AAPL", start="2020-01-01", end="2023-12-31")

if stock_data.empty:
    print("No data retrieved from Yahoo Finance. Please check the ticker or dates.")
    exit()

stock_data_close = stock_data[['Close']].copy()

stock_data_close['MA_5'] = stock_data_close['Close'].rolling(window=5).mean()
stock_data_close['MA_10'] = stock_data_close['Close'].rolling(window=10).mean()
stock_data_close['Pct_Change'] = stock_data_close['Close'].pct_change() * 100
stock_data_close['Lag_1'] = stock_data_close['Close'].shift(1)

stock_data_close = stock_data_close.dropna()

X = stock_data_close[['MA_5', 'MA_10', 'Pct_Change', 'Lag_1']]
y = stock_data_close['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def evaluate_model(model, X_train, X_test, y_train, y_test, iterations=10):
    correlations, maes, rmses, accuracies = [], [], [], []
    
    for _ in range(iterations):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        y_test_flat = y_test.flatten()
        y_pred_flat = y_pred.flatten()
        
        correlation = np.corrcoef(y_test_flat, y_pred_flat)[0, 1]
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        accuracy = r2_score(y_test, y_pred)
        
        correlations.append(correlation)
        maes.append(mae)
        rmses.append(rmse)
        accuracies.append(accuracy)
    
    correlation_mean = np.mean(correlations)
    mae_mean = np.mean(maes)
    rmse_mean = np.mean(rmses)
    accuracy_mean = np.mean(accuracies)
    
    rmse_std = np.std(rmses)
    accuracy_std = np.std(accuracies)
    
    return correlation_mean, mae_mean, rmse_mean, accuracy_mean, rmse_std, accuracy_std

decision_stump = DecisionTreeRegressor(max_depth=1)
decision_stump_results = evaluate_model(decision_stump, X_train_scaled, X_test_scaled, y_train, y_test)
print(f"Decision Stump:")
print(f"Correlation coefficient: {decision_stump_results[0]:.4f}")
print(f"MAE: {decision_stump_results[1]:.4f}")
print(f"RMSE: {decision_stump_results[2]:.4f}")
print(f"Accuracy: ~{decision_stump_results[3]*100:.2f}%")

linear_regression = LinearRegression()
linear_regression_results = evaluate_model(linear_regression, X_train_scaled, X_test_scaled, y_train, y_test)
print(f"\nLinear Regression:")
print(f"Correlation coefficient: {linear_regression_results[0]:.4f}")
print(f"MAE: {linear_regression_results[1]:.4f}")
print(f"RMSE: {linear_regression_results[2]:.4f}")
print(f"Accuracy: ~{linear_regression_results[3]*100:.2f}%")

svm = SVR(kernel='rbf')
svm_results = evaluate_model(svm, X_train_scaled, X_test_scaled, y_train, y_test)
print(f"\nSVM (C-Class):")
print(f"RMSE: {svm_results[2]:.4f} +/- {svm_results[5]:.4f}")
print(f"Accuracy: {svm_results[3]*100:.2f}% +/- {svm_results[5]*100:.2f}%")

adaboost = AdaBoostRegressor(n_estimators=50)
adaboost_results = evaluate_model(adaboost, X_train_scaled, X_test_scaled, y_train, y_test)
print(f"\nBoosting (AdaBoostM1):")
print(f"RMSE: {adaboost_results[2]:.4f} +/- {adaboost_results[5]:.4f}")
print(f"Accuracy: {adaboost_results[3]*100:.2f}% +/- {adaboost_results[5]*100:.2f}%")
