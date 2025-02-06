#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install yfinance')


# In[ ]:


import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import statistics


# ### Data

# In[ ]:


data = yf.download('00646.tw',start='2018-01-01',end='2024-01-01')
data = data.reset_index()


# In[ ]:


data.describe()


# ### Data Preprocessing

# In[ ]:


df = pd.DataFrame()
df['date'] = data['Date'].shift(-5)
df['close'] = data['Close'].shift(-5)
df['open'] = data['Open']
df['low'] = data['Low']
df['high'] = data['High']
df['volume'] = data['Volume']
df = df.dropna()
df.head()


# In[ ]:


train_set, test_set = np.split(df, [int(.8 *len(df))])
train_X , train_y = train_set.drop(['date','close'], axis=1), train_set['close']
test_X , test_y= test_set.drop(['date','close'], axis=1), test_set['close']


# In[ ]:


test_set


# In[ ]:


data[1163:1454]


# ### Model

# linear regression

# In[ ]:


regressor = linear_model.LinearRegression()
regressor.fit(train_X,train_y)
predlm_y = regressor.predict(test_X)


# randomforest

# In[ ]:


param_grid = {"max_depth":[4,5,6],
          "min_samples_leaf":[6,7,8],
          "min_samples_split":[7,8,9] }
forest_reg = RandomForestRegressor(n_estimators = 100)
grid = GridSearchCV(forest_reg, param_grid, cv=5)
grid.fit(train_X,train_y)


# In[ ]:


grid.best_estimator_


# In[ ]:


regressor = RandomForestRegressor(max_depth=5, min_samples_leaf=7, min_samples_split=8, n_estimators=100)
regressor.fit(train_X,train_y)
predrf_y = regressor.predict(test_X)


# support vector regression

# In[ ]:


scaler = StandardScaler().fit(train_X)
train_X_scaled = scaler.transform(train_X)
test_X_scaled = scaler.transform(test_X)


# In[ ]:


svc = SVR()
grid = GridSearchCV(
estimator=SVR(kernel='rbf'),
param_grid={
'C': [1e05,1e06,1e07],
'epsilon': [0.1,1,10],
'gamma': [1e-06,1e-05,1e-04]
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
grid.fit(train_X_scaled,train_y)


# In[ ]:


grid.best_estimator_


# In[ ]:


regressor = SVR(kernel='rbf', C=1000000.0, epsilon=1, gamma=1e-05)
regressor.fit(train_X_scaled,train_y)
predsvr_y = regressor.predict(test_X_scaled)


# performance

# In[ ]:


def performance(Y_test, predict):
  mse = np.mean((Y_test-predict)**2)
  rmse = np.sqrt(np.mean((Y_test-predict)**2))
  mae = np.mean(np.abs(Y_test-predict))
  mape = np.mean((np.abs(Y_test-predict))/Y_test) * 100

  return mse, rmse, mae, mape


# In[ ]:


rf=performance(test_y, predrf_y)
lm=performance(test_y, predlm_y)
svr=performance(test_y, predsvr_y)


# In[ ]:


import prettytable as pt
tb1 = pt.PrettyTable()
tb1.field_names = ['Model', 'MSE', 'RMSE', 'MAE', 'MAPE']
tb1.add_row(['RandomForest',rf[0],rf[1],rf[2],rf[3]])
tb1.add_row(['Linear',lm[0],lm[1],lm[2],lm[3]])
tb1.add_row(['SVR',svr[0],svr[1],svr[2],svr[3]])
print(tb1)


# ### 交易策略

# In[ ]:


def sharpe_ratio(list1, risk_free_rate):
  re = []
  for i in range(1,len(list1)):
    re.append((list1[i]-list1[i-1])/list1[i-1])
  return round((statistics.mean(re)-risk_free_rate)/statistics.pstdev(re)*(252**0.5),3)


# In[ ]:


def LdailyPL(position, close):
  pl = 0
  pl += (close-position["price"])*position["unit"]
  return pl

def SdailyPL(position, close):
  pl = 0
  pl += (position["price"]-close)*position["unit"]
  return pl

def LEndPL(position, end, cost):
  pl = 0
  pl += ((end-position["price"])-position["price"]*cost)*position["unit"]
  return pl

def SEndPL(position, end, cost):
  pl = 0
  pl += ((position["price"]-end)-position["price"]*cost)*position["unit"]
  return pl


# In[ ]:


def Longstrategyd(future, equity1, equity2, cost):
  record = [0]*len(future)
  b_h = [] #Buy and hold 部位
  trade = [0]*len(future) #計算交易部位損益
  daily_pl = [] #總損益
  daily = []  #每日交易紀錄
  signal = 0  #做多:1 無交易:0
  num2 = int(equity2/future['close'][0])
  equity2 -= (future['close'][0]*cost*num2)
  for i in range(0,len(future)):
    if i != 0:
      equity2 += (future['close'][i]-future['close'][i-1])*num2
    b_h.append(equity2)

    if future['week'][i] == 1:
      if (signal == 1) & (future['signal'][i]==0):
        equity1 += LEndPL(position, future['close'][i], cost)
        position = {}
        daily.append({"date":future['date'][i], "trading":"long close", "unit":num1, "price":future['close'][i]})
        record[i] = -1
        signal = 0
        trade[i] = equity1

      if (signal == 0) & (future['pred_nw'][i] > future['close'][i]):
        num1 = int(equity1/future['close'][i])
        position={"unit": num1, "price": future['close'][i]}
        daily.append({"date":future['date'][i], "trading":"long", "unit":num1, "price":future['close'][i]})
        record[i] = 1
        signal = 1
        equity1 -= future['close'][i]*cost*num1
    else:
      if (signal == 1) & (future['signal'][i]==0):
        equity1 += LEndPL(position, future['close'][i], cost)
        position = {}
        daily.append({"date":future['date'][i], "trading":"long close", "unit":num1, "price":future['close'][i]})
        record[i] = -1
        signal = 0
        trade[i] = equity1

    if signal == 1:
      trade[i] = equity1 + LdailyPL(position, future['close'][i])
    else:
      trade[i] = equity1

    daily_pl.append(trade[i]+b_h[i])
  return daily, trade, b_h, daily_pl, record


# In[ ]:


data = yf.download('00646.tw',start='2018-01-01',end='2024-01-01') #日期可更改
data = data.reset_index()


# In[ ]:


short_window = 20
long_window = 60
data['signal'] = 0.0
data['short_mavg'] = data['Close'].rolling(short_window).mean().shift(1)
data['long_mavg'] = data['Close'].rolling(long_window).mean().shift(1)
data['signal'][short_window:] = np.where(data['short_mavg'][short_window:] > data['long_mavg'][short_window:], 1.0, 0.0)
data = data[1163:1454]
data = data.reset_index(drop='True')
data['pred_nw'] = predsvr_y
data


# In[ ]:


data = data.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close"})
data["week"] = [0]*291
for i in range(0,len(data["date"])):
  ind = data.loc[pd.to_datetime(data.date)==pd.Timestamp(data["date"][i])].index
  data['week'][ind] = 1


# In[ ]:


point = 1
cost = 0.001425
equity1 = 5000
equity2 = 5000


# In[ ]:


bk1 = Longstrategyd(data, equity1, equity2, cost)
equity = equity1 +equity2
rr = round(((bk1[3][len(bk1[3])-1]-equity)/equity)*100, 2)
rr1 = round(((bk1[1][len(bk1[1])-1]-equity1)/equity1)*100, 2)
rr2 = round(((bk1[2][len(bk1[2])-1]-equity2)/equity2)*100, 2)


# In[ ]:


data['trade'] = bk1[1]
data['b_h'] = bk1[2]
data['total'] = bk1[3]


# ### 回測結果

# In[ ]:


s1 = sharpe_ratio(data['trade'], 0.0001)
s2 = sharpe_ratio(data['b_h'], 0.0001)
s3 = sharpe_ratio(data['total'], 0.0001)


# In[ ]:


fig = plt.figure( figsize=(15,5) )
ax1 = fig.add_subplot(111,  ylabel='Price')
plt.plot(data['date'],data['close'], color='grey', lw=1., label='Close')
plt.plot(data['date'],data['short_mavg'], color='orange', lw=1., label='short_mavg')
plt.plot(data['date'],data['long_mavg'], color='skyblue', lw=1., label='long_mavg')
plt.legend(loc = 2)
ax2=ax1.twinx()
plt.plot(data['date'],data['trade'], color='black', lw=1.2, label='trade')
ax2.set_ylabel('Profit and Loss', color = "black")
plt.title("00646.tw, Long")
plt.legend(loc = 4)
plt.show()


# In[ ]:


fig = plt.figure( figsize=(15,5) )
ax1 = fig.add_subplot(111,  ylabel='Price')
plt.plot(data['date'],data['trade'], color='blue', lw=1., label='50% trade '+str(rr1)+"% Sharpe ratio:"+str(s1))
plt.plot(data['date'],data['b_h'], color='orange', lw=1., label='50% B&H '+str(rr2)+"% Sharpe ratio:"+str(s2))
plt.legend(loc = 2)

ax2=ax1.twinx()
plt.plot(data['date'],data['total'], color='black', lw=1., label='Portfolio '+str(rr)+"% Sharpe ratio:"+str(s3))
ax2.set_ylabel('Profit and Loss', color = "black")
plt.title("00646.tw, Long")
plt.legend(loc = 4)
plt.show()


# ### Classification

# #### Data

# In[ ]:


data = yf.download('00646.tw',start='2018-01-01',end='2024-01-01')
data = data.reset_index()


# In[ ]:


data['Direction'] = np.where(data['Adj Close'].shift(-5) > data['Adj Close'], 'up', 'down')
data


# #### Data Preprocessing

# In[ ]:


df = data[0:1454]
df


# In[ ]:


train_set, test_set = np.split(df, [int(.8 *len(df))])
X_train , y_train = train_set.drop(['Date','Direction','Adj Close'], axis=1), train_set['Direction']
X_test , y_test= test_set.drop(['Date','Direction','Adj Close'], axis=1), test_set['Direction']


# #### Model

# Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay
from sklearn import metrics


# hyperparameter optimization

# In[ ]:


param_grid = {"max_depth":[1,2,3],
          "min_samples_leaf":[2,3,4,5],
          "min_samples_split":[2,3,4],
          "bootstrap": [True, False]}
rf = RandomForestClassifier(n_estimators = 100)
rf_grid = GridSearchCV(rf, param_grid, cv=5)
rf_grid.fit(X_train, y_train)
rf_grid.best_estimator_


# model and performance

# In[ ]:


rf_model = RandomForestClassifier(n_estimators=100,max_depth=1, min_samples_leaf=2, min_samples_split=4, random_state=1)
rf_model.fit(X_train, y_train)
print('Correct Prediction (%): ', accuracy_score(y_test, rf_model.predict(X_test), normalize=True)*100.0)

report = classification_report(y_test, rf_model.predict(X_test))
print(report)


# rf feature importance

# In[ ]:


feature_imp = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feature_imp


# 根據feature importance重新配適模型

# In[ ]:


train_set, test_set = np.split(df, [int(.8 *len(df))])
X_train , y_train = train_set.drop(['Date','Direction','Adj Close','Low','Volume'], axis=1), train_set['Direction']
X_test , y_test= test_set.drop(['Date','Direction','Adj Close','Low','Volume'], axis=1), test_set['Direction']


# In[ ]:


param_grid = {"max_depth":[1,2,3],
          "min_samples_leaf":[2,3,4,5],
          "min_samples_split":[3,4,5],
          "bootstrap": [True, False]}
rf = RandomForestClassifier(n_estimators = 100)
rf_grid = GridSearchCV(rf, param_grid, cv=5)
rf_grid.fit(X_train, y_train)
rf_grid.best_estimator_


# In[ ]:


rf_model = RandomForestClassifier(n_estimators=100,max_depth=2, min_samples_leaf=4, min_samples_split=4, random_state=1)
rf_model.fit(X_train, y_train)
print('Correct Prediction (%): ', accuracy_score(y_test, rf_model.predict(X_test), normalize=True)*100.0)

report = classification_report(y_test, rf_model.predict(X_test))
print(report)


# In[ ]:


predicted = rf_model.predict(X_test)
confusion_matrix = metrics.confusion_matrix(y_test, predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels=["down","up"])
cm_display.plot()
plt.show()


# ROC and AUC

# In[ ]:


fig, ax = plt.subplots()
rfc_disp = RocCurveDisplay.from_estimator(rf_model, X_test, y_test, alpha = 0.8, name='ROC Curve', lw=1, ax=ax)
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="ROC Curve Random Forest")
ax.legend(loc="lower right")
plt.show()


# #### 加入技術指標

# 

# In[ ]:


get_ipython().system('wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz')
get_ipython().system('tar -xzvf ta-lib-0.4.0-src.tar.gz')
get_ipython().run_line_magic('cd', 'ta-lib')
get_ipython().system('./configure --prefix=/usr')
get_ipython().system('make')
get_ipython().system('make install')
get_ipython().system('pip install Ta-Lib')
import talib as ta


# In[ ]:


data = yf.download('00646.tw',start='2018-01-01',end='2024-01-01')
data = data.reset_index()
data['Direction'] = np.where(data['Adj Close'].shift(-5) > data['Adj Close'], 'up', 'down')
data


# In[ ]:


data['EMA10'] = ta.EMA(data['Close'], timeperiod=10)

macd, signal, hist = ta.MACD(data['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
data['MACD'] = macd
data['MACD_HIST'] = hist

data['RSI30'] = ta.RSI(data['Close'], timeperiod=30)


slowk,slowd = ta.STOCH(data['High'],data['Low'],data['Close'], fastk_period=9,slowk_period=3,slowd_period=3)
data['SlowK'] = slowk
data['SlowD'] = slowd

data['MOM10'] = ta.MOM(data['Close'], timeperiod=10)

data['CCI'] = ta.CCI(data['High'],data['Low'],data['Close'],timeperiod=14)

data['WILLR'] = ta.WILLR(data['High'],data['Low'],data['Close'],timeperiod=14)


data = data.dropna()
data


# In[ ]:


df = data[:len(data)-5]
train_set, test_set = np.split(df, [int(.8 *len(df))])
X_train , y_train = train_set.drop(['Date','Direction','Adj Close'], axis=1), train_set['Direction']
X_test , y_test= test_set.drop(['Date','Direction','Adj Close'], axis=1), test_set['Direction']


# In[ ]:


param_grid = {"max_depth":[1,2,3,4],
          "min_samples_leaf":[2,3,4],
          "min_samples_split":[10,11,12]}
rf = RandomForestClassifier(n_estimators = 250)
rf_grid = GridSearchCV(rf, param_grid, cv=5)
rf_grid.fit(X_train, y_train)
rf_grid.best_estimator_


# In[ ]:


rf_model = RandomForestClassifier(n_estimators=250,max_depth=2, min_samples_leaf=3, min_samples_split=10, random_state=1)
rf_model.fit(X_train, y_train)
print('Correct Prediction (%): ', accuracy_score(y_test, rf_model.predict(X_test), normalize=True)*100.0)

report = classification_report(y_test, rf_model.predict(X_test))
print(report)


# In[ ]:


#檢查相關性
data.drop(['Date','Direction'], axis=1).pct_change().dropna().corr()


# In[ ]:


#特徵重要性
feature_imp = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feature_imp


# In[ ]:


#價格資料彼此之間相關性高，因此移除
#移除 feature importance < 0.2 的 feature
train_set, test_set = np.split(df, [int(.8 *len(df))])
X_train , y_train = train_set.drop(['Date','Direction','Open','Close','High','Low','Adj Close','Volume','CCI','MOM10'], axis=1), train_set['Direction']
X_test , y_test= test_set.drop(['Date','Direction','Open','Close','High','Low','Adj Close','Volume','CCI','MOM10'], axis=1), test_set['Direction']


# In[ ]:


param_grid = {"max_depth":[1,2,3,4],
          "min_samples_leaf":[2,3,4,5],
          "min_samples_split":[6,7,8,9]}
rf = RandomForestClassifier(n_estimators = 250)
rf_grid = GridSearchCV(rf, param_grid, cv=5)
rf_grid.fit(X_train, y_train)
rf_grid.best_estimator_


# In[ ]:


rf_model = RandomForestClassifier(n_estimators=250,max_depth=1, min_samples_leaf=2, min_samples_split=7, random_state=1)
rf_model.fit(X_train, y_train)
print('Correct Prediction (%): ', accuracy_score(y_test, rf_model.predict(X_test), normalize=True)*100.0)

report = classification_report(y_test, rf_model.predict(X_test))
print(report)


# In[ ]:


predicted = rf_model.predict(X_test)
confusion_matrix = metrics.confusion_matrix(y_test, predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels=["down","up"])
cm_display.plot()
plt.show()


# In[ ]:


fig, ax = plt.subplots()
rfc_disp = RocCurveDisplay.from_estimator(rf_model, X_test, y_test, alpha = 0.8, name='ROC Curve', lw=1, ax=ax)
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="ROC Curve Random Forest")
ax.legend(loc="lower right")
plt.show()

