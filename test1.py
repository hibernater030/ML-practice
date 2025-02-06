#最終版 註解已翻譯

# 匯入所需模組
import yfinance as yf
import pandas as pd
import numpy as np
import talib as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# -------- 展示ETF基本資訊 --------
# 設定圖表樣式
sns.set(style="whitegrid")

# 顯示所選日期範圍內所有股票的基本資訊
def display_stock_info(df):
    # 基本資訊
    basic_info_df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    
    # 顯示所有基本資訊
    print("Basic Stock Information (Open, High, Low, Close, Adj Close, Volume) for the selected date range:")
    print(basic_info_df)
    print("\nSummary of Basic Stock Information:")
    print(basic_info_df.describe())  # 顯示統計數據

# -------- 抓取資料並計算技術指標 --------
def fetch_data_with_indicators(ticker, start_date="2005-01-01", end_date="2024-10-04"):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Return'] = data['Adj Close'].pct_change()
    data['MA_10'] = ta.SMA(data['Adj Close'], timeperiod=10)
    data['RSI'] = ta.RSI(data['Adj Close'], timeperiod=14)
    data['MACD'], data['MACD_signal'], _ = ta.MACD(data['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    data['BB_Mid'] = ta.SMA(data['Adj Close'], timeperiod=20)
    data['BB_Upper'] = data['BB_Mid'] + 2 * ta.STDDEV(data['Adj Close'], timeperiod=20)
    data['BB_Lower'] = data['BB_Mid'] - 2 * ta.STDDEV(data['Adj Close'], timeperiod=20)
    data['Future Price'] = data['Adj Close'].shift(-5)
    data['Label'] = np.where(data['Future Price'] > data['Adj Close'], 1, 0)
    data.dropna(inplace=True)
    return data

# 抓取 SPY、DIA 和 QQQ 的資料
tickers = ['SPY', 'DIA', 'QQQ']
data = {ticker: fetch_data_with_indicators(ticker) for ticker in tickers}

# 顯示所有基本股票資訊
for ticker in tickers:
    print(f"\n=== {ticker} Basic Information ===")
    display_stock_info(data[ticker])

# -------- 生成指標信號 --------
def generate_indicator_signals(df):
    df['RSI_signal'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))
    df['MACD_signal'] = np.where(df['MACD'] > df['MACD_signal'], 1, -1)
    df['MA_signal'] = np.where(df['Adj Close'] > df['MA_10'], 1, -1)
    df['BB_signal'] = np.where(df['Adj Close'] < df['BB_Lower'], 1, 
                                np.where(df['Adj Close'] > df['BB_Upper'], -1, 0))

    return df

data = {ticker: generate_indicator_signals(df) for ticker, df in data.items()}

# 儲存模型和結果
models = {}
accuracies = {}
auc_scores = {}
final_accuracies = {}
final_auc_scores = {}

# -------- 訓練與評估模型 --------
# 幫每個ETF建立、評估模型
for ticker in tickers:
    print(f"\n=== {ticker} ===")
    
    # 定義特徵和label
    features = data[ticker][['Adj Close', 'Return', 'RSI_signal', 'MACD_signal', 'MA_signal', 'BB_Upper', 'BB_Lower']]
    labels = data[ticker]['Label']
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=False)
    
    # Random forest模型
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    # 計算準確率、AUC、precision、recall、F1 score
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"{ticker} Model Accuracy: {accuracy:.4f}")
    print(f"{ticker} AUC Score: {roc_auc:.4f}")
    print(f"{ticker} Precision: {precision:.4f}")
    print(f"{ticker} Recall: {recall:.4f}")
    print(f"{ticker} F1 Score: {f1:.4f}")

    # 特徵重要性分析
    feature_importances = model.feature_importances_
    feature_names = features.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # 視覺化呈現特徵重要性
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title(f'{ticker} Feature Importance')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.show()

    # 加入混淆矩陣和評估指標
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # 顯示分類報告
    report = classification_report(y_test, y_pred, target_names=["No Buy", "Buy"])
    print("Classification Report:")
    print(report)

    # 視覺化呈現混淆矩陣
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Buy", "Buy"], yticklabels=["No Buy", "Buy"])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'{ticker} - Confusion Matrix')
    plt.show()

    # 生成綜合信號
    data_test = X_test.copy()
    data_test['Model_Prediction'] = y_pred
    data_test['Model_Prob'] = y_pred_prob
    data_test['BB_signal'] = data[ticker]['BB_signal'].loc[X_test.index]  # Use loc to get BB_signal by index
    data_test['Combined_Signal'] = (
        data_test['RSI_signal'] + 
        data_test['MACD_signal'] + 
        data_test['MA_signal'] + 
        data_test['BB_signal'] +
        data_test['Model_Prediction']
    )
    data_test['Final_Prediction'] = np.where(data_test['Combined_Signal'] > 2, 1, 0)
    
    # 計算綜合信號的準確率和AUC
    final_accuracy = accuracy_score(y_test, data_test['Final_Prediction'])
    final_auc = roc_auc_score(y_test, data_test['Final_Prediction'])
    print(f"{ticker} Combined Signal Accuracy: {final_accuracy:.4f}")
    print(f"{ticker} Combined Signal AUC Score: {final_auc:.4f}")
    
    final_accuracies[ticker] = final_accuracy
    final_auc_scores[ticker] = final_auc

    # 視覺化呈現準確率、ROC curve
    plt.figure(figsize=(14, 6))
    
    # 生成準確率長條圖
    plt.subplot(1, 2, 1)
    sns.barplot(x=['Model Accuracy', 'Combined Signal Accuracy'], y=[accuracy, final_accuracy], palette="Blues_d")
    plt.title(f"{ticker} - Accuracy Comparison")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")

    # 視覺化呈現ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    final_fpr, final_tpr, _ = roc_curve(y_test, data_test['Final_Prediction'])

    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label=f'Model AUC = {roc_auc:.4f}', color='blue')
    plt.plot(final_fpr, final_tpr, label=f'Combined Signal AUC = {final_auc:.4f}', color='orange')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{ticker} - ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()


# -------- 預測五日後的漲跌 --------
# 根據資料預測未來價格
def predict_future_price(tickers, data):
    predictions = {}
    for ticker in tickers:
        # 初始化資料框來儲存未來預測
        future_prediction = pd.DataFrame(columns=['Date', 'Predicted_Price', 'Predicted_Change'])

        # 獲取最後的data point
        last_data_point = data[ticker].iloc[-1]

        # 根據最後的data point準備預測的輸入特徵
        feature_set = pd.DataFrame({
            'Adj Close': [last_data_point['Adj Close']],
            'Return': [last_data_point['Return']],
            'RSI_signal': [last_data_point['RSI_signal']],
            'MACD_signal': [last_data_point['MACD_signal']],
            'MA_signal': [last_data_point['MA_signal']],
            'BB_Upper': [last_data_point['BB_Upper']],
            'BB_Lower': [last_data_point['BB_Lower']]
        })

        # 預測未來5天的價格
        for i in range(5):
            future_price = model.predict(feature_set)[0]
            # 為預測建立一個新的DataFrame
            new_prediction = pd.DataFrame({
                'Date': [pd.Timestamp.now() + pd.Timedelta(days=i + 1)],
                'Predicted_Price': [future_price],
                'Predicted_Change': [future_price - last_data_point['Adj Close']]
            })
            
            # 將新的預測與future_prediction的DataFrame連接
            future_prediction = pd.concat([future_prediction, new_prediction], ignore_index=True)

            # 更新下一天的feature_set
            feature_set['Adj Close'] = future_price  # 更新下一次預測的「最後預測價格」

        predictions[ticker] = future_prediction
    return predictions

# 呼叫函式來預測未來價格
future_price_predictions = predict_future_price(tickers, data)
for ticker, preds in future_price_predictions.items():
    print(f"\nPredicted Future Prices for {ticker}:")
    print(preds)
