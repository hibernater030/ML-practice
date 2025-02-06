#python test2.py
#回測&新增特徵

# Import necessary packages
import yfinance as yf
import pandas as pd
import numpy as np
import talib as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Function to display all basic stock information for the selected date range
def display_stock_info(df):
    basic_info_df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    print("Basic Stock Information (Open, High, Low, Close, Adj Close, Volume) for the selected date range:")
    print(basic_info_df)
    print("\nSummary of Basic Stock Information:")
    print(basic_info_df.describe())

# Step 1: Fetch data and calculate technical indicators
def fetch_data_with_indicators(ticker, start_date="2005-01-01", end_date="2024-10-04"):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Return'] = data['Adj Close'].pct_change()
    data['MA_10'] = ta.SMA(data['Adj Close'], timeperiod=10)
    data['RSI'] = ta.RSI(data['Adj Close'], timeperiod=14)
    data['MACD'], data['MACD_signal'], _ = ta.MACD(data['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    data['BB_Mid'] = ta.SMA(data['Adj Close'], timeperiod=20)
    data['BB_Upper'] = data['BB_Mid'] + 2 * ta.STDDEV(data['Adj Close'], timeperiod=20)
    data['BB_Lower'] = data['BB_Mid'] - 2 * ta.STDDEV(data['Adj Close'], timeperiod=20)
    
    # New features
    data['Price_Momentum'] = data['Adj Close'] - data['Adj Close'].shift(5)  # 5-day price momentum
    data['Volume_Change'] = data['Volume'].pct_change(5)  # 5-day volume change
    
    data['Future Price'] = data['Adj Close'].shift(-5)
    data['Label'] = np.where(data['Future Price'] > data['Adj Close'], 1, 0)
    data.dropna(inplace=True)
    return data

# Fetch data for SPY, DIA, and QQQ
tickers = ['SPY', 'DIA', 'QQQ']
data = {ticker: fetch_data_with_indicators(ticker) for ticker in tickers}

# Display all basic stock information for each ticker
for ticker in tickers:
    print(f"\n=== {ticker} Basic Information ===")
    display_stock_info(data[ticker])

# Step 2: Generate indicator signals
def generate_indicator_signals(df):
    df['RSI_signal'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))
    df['MACD_signal'] = np.where(df['MACD'] > df['MACD_signal'], 1, -1)
    df['MA_signal'] = np.where(df['Adj Close'] > df['MA_10'], 1, -1)
    return df

data = {ticker: generate_indicator_signals(df) for ticker, df in data.items()}

# Store models and results
models = {}
accuracies = {}
auc_scores = {}
final_accuracies = {}
final_auc_scores = {}

# Build and evaluate models for each ETF
for ticker in tickers:
    print(f"\n=== {ticker} ===")
    
    # Define features and labels
    features = data[ticker][['Adj Close', 'Return', 'RSI_signal', 'MACD_signal', 'MA_signal', 'BB_Upper', 'BB_Lower', 
                             'Price_Momentum', 'Volume_Change']]
    labels = data[ticker]['Label']
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=False)
    
    # Random forest model
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate accuracy and AUC
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    print(f"{ticker} Model Accuracy: {accuracy:.4f}")
    print(f"{ticker} AUC Score: {roc_auc:.4f}")

    # Confusion matrix and evaluation metrics
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Buy", "Buy"]))

    # Visualize - Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Buy", "Buy"], yticklabels=["No Buy", "Buy"])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'{ticker} - Confusion Matrix')
    plt.show()

    # Combined signal for backtesting
    X_test['Model_Prediction'] = y_pred
    X_test['Model_Prob'] = y_pred_prob
    X_test['Combined_Signal'] = X_test['RSI_signal'] + X_test['MACD_signal'] + X_test['MA_signal'] + X_test['Model_Prediction']
    X_test['Final_Prediction'] = np.where(X_test['Combined_Signal'] > 2, 1, 0)

    # Calculate accuracy and AUC for combined signal
    final_accuracy = accuracy_score(y_test, X_test['Final_Prediction'])
    final_auc = roc_auc_score(y_test, X_test['Final_Prediction'])
    print(f"{ticker} Combined Signal Accuracy: {final_accuracy:.4f}")
    print(f"{ticker} Combined Signal AUC Score: {final_auc:.4f}")
    
    final_accuracies[ticker] = final_accuracy
    final_auc_scores[ticker] = final_auc

    # Visualize - Accuracy and ROC curve
    plt.figure(figsize=(14, 6))
    
    # Accuracy bar chart
    plt.subplot(1, 2, 1)
    sns.barplot(x=['Model Accuracy', 'Combined Signal Accuracy'], y=[accuracy, final_accuracy], palette="Blues_d")
    plt.title(f"{ticker} - Accuracy Comparison")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    final_fpr, final_tpr, _ = roc_curve(y_test, X_test['Final_Prediction'])

    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label=f'Model AUC = {roc_auc:.4f}', color='blue')
    plt.plot(final_fpr, final_tpr, label=f'Combined Signal AUC = {final_auc:.4f}', color='orange')
    plt.plot([0, 1], [0, 1], 'k--', color='grey')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{ticker} - ROC Curve Comparison")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Backtesting cumulative returns
    X_test['Return'] = data[ticker].loc[X_test.index, 'Return']
    X_test['Strategy_Return'] = X_test['Return'] * X_test['Final_Prediction']
    X_test['Cumulative_Strategy_Return'] = (1 + X_test['Strategy_Return']).cumprod()
    X_test['Cumulative_Market_Return'] = (1 + X_test['Return']).cumprod()

    # Plot backtest results
    plt.figure(figsize=(10, 6))
    plt.plot(X_test.index, X_test['Cumulative_Market_Return'], label='Market Return', color='blue')
    plt.plot(X_test.index, X_test['Cumulative_Strategy_Return'], label='Strategy Return', color='orange')
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title(f"{ticker} - Backtest Cumulative Returns")
    plt.legend()
    plt.show()

# Predict future prices for the next 5 days
def predict_future_prices(tickers, data):
    predictions = {}
    for ticker in tickers:
        last_data = data[ticker].iloc[-1][['Adj Close', 'Return', 'RSI_signal', 'MACD_signal', 'MA_signal', 
                                           'BB_Upper', 'BB_Lower', 'Price_Momentum', 'Volume_Change']].values.reshape(1, -1)
        prediction = model.predict(last_data)
        predictions[ticker] = 'Up' if prediction == 1 else 'Down'
    return predictions

# Display predictions for next 5 days
print("\nPredictions for the next 5 days:")
print(predict_future_prices(tickers, data))
