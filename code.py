
import pandas as pd
from sklearn import linear_model
import random
import numpy as np
import matplotlib


%matplotlib jupyter

df = pd.read_csv('battledeaths_n300_cor99.csv')

df.head(10)

df.plot(x='friendly_battledeaths', y='enemy_battledeaths', kind='scatter')

X = df['friendly_battledeaths']
y = df['enemy_battledeaths']

X_test = X[0:30].reshape(-1,1)
y_test = y[0:30]

X_train = X[30:].reshape(-1,1)
y_train = y[30:]

ols = linear_model.LinearRegression()

model = ols.fit(X_train, y_train)

model.coef_

model.score(X_test, y_test)

list(model.predict(X_test)[0:5])

list(y_test)[0:5]

((y_test - model.predict(X_test)) **2).sum()

np.mean((model.predict(X_test) - y_test) **2)









