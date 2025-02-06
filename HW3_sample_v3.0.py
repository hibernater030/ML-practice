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


# ### Classification

# #### Data

# In[ ]:


start_time = '2005-01-01'
end_time = '2024-10-05'
split_ratio = .8
future_days = 5


# In[ ]:


data = yf.download('QQQ',start=start_time,end=end_time)
data = data.reset_index()


# In[ ]:


data.describe()


# In[ ]:


data


# In[ ]:


data['Direction'] = np.where(data['Adj Close'].shift(-future_days) > data['Adj Close'], 'up', 'down')
data


# #### Data Preprocessing

# In[ ]:


df = data[0:-future_days]
df


# In[ ]:


train_set, test_set = np.split(df, [int(split_ratio *len(df))])
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


train_set, test_set = np.split(df, [int(split_ratio *len(df))])
X_train , y_train = train_set.drop(['Date','Direction','Adj Close','High','Volume', 'Open'], axis=1), train_set['Direction']
X_test , y_test= test_set.drop(['Date','Direction','Adj Close','High','Volume', 'Open'], axis=1), test_set['Direction']


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


data = yf.download('QQQ',start=start_time,end=end_time)
data = data.reset_index()
data['Direction'] = np.where(data['Adj Close'].shift(-future_days) > data['Adj Close'], 'up', 'down')
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


df = data[:len(data)-future_days]
train_set, test_set = np.split(df, [int(split_ratio *len(df))])
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


predicted = rf_model.predict(X_test)
confusion_matrix = metrics.confusion_matrix(y_test, predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels=["down","up"])
cm_display.plot()
plt.show()


# In[ ]:


#檢查相關性
data.drop(['Date','Direction'], axis=1).pct_change().dropna().corr()


# In[ ]:


#特徵重要性
feature_imp = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feature_imp


# In[ ]:


#價格資料彼此之間相關性高，因此移除
#移除 feature importance < 0.07 的 feature
train_set, test_set = np.split(df, [int(split_ratio *len(df))])
X_train , y_train = train_set.drop(['Date','Direction','Open','Adj Close','High','Low','MACD_HIST','SlowK','CCI','WILLR', 'RSI30', 'MOM10', 'SlowD'], axis=1), train_set['Direction']
X_test , y_test= test_set.drop(['Date','Direction','Open','Adj Close','High','Low','MACD_HIST','SlowK','CCI','WILLR', 'RSI30', 'MOM10', 'SlowD'], axis=1), test_set['Direction']


# In[ ]:


param_grid = {"max_depth":[1,2,3,4],
          "min_samples_leaf":[2,3,4,5],
          "min_samples_split":[6,7,8,9]}
rf = RandomForestClassifier(n_estimators = 250)
rf_grid = GridSearchCV(rf, param_grid, cv=5)
rf_grid.fit(X_train, y_train)
rf_grid.best_estimator_


# In[ ]:


rf_model = RandomForestClassifier(n_estimators=250,max_depth=10, min_samples_leaf=5, min_samples_split=7, random_state=1)
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
# 添加0.5的參照線
ax.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r', label='Reference Line (0.5)', alpha=0.8)

ax.legend(loc="lower right")
plt.show()


# In[ ]:




