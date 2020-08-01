import pandas as pd 
import math 
import numpy as np             
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt           
from matplotlib import style
from  datetime import datetime
from os.path import split
import time
style.use('ggplot')
# making data 

df = pd.read_csv('https://www.quandl.com/api/v3/datasets/CHRIS/MGEX_MW4.csv?api_key=wazxAo2-synpC95b-X_h')
df = df[['Open','High','Low','Last','Volume','Open Interest']]

forecast_col = 'Open Interest'
df.fillna(-99999, inplace=True)
forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

# make data into array

x = np.array(df.drop(['label'], 1))
x = preprocessing.scale(x)
x_lately = x[-forecast_out:]
x = x[:-forecast_out]
df.dropna(inplace=True)
y = np.array(df['label'])
y = y[0:6171]

# assigining data for training

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# make classification 
clf = LinearRegression(n_jobs = -1)
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
forecast_set = clf.predict(x_lately)






