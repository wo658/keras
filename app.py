import numpy as np
import pandas as pd

data = pd.read_csv('gpascore.csv')

# print(data.isnull().sum())
data = data.dropna()

y_data = data['admit'].values

x_data = []

for i, rows in data.iterrows():
    x_data.append([rows['gre'],rows['gpa'],rows['rank']])

import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64 , activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),   

 ])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'] )

model.fit( np.array(x_data) , np.array(y_data) , epochs =1000 )

#predict

predict = model.predict( np.array([  [750,3.70,3] ,[400,2.2,1] ]))
print(predict)

# data 전처리
# 파라미터 튜닝
# 