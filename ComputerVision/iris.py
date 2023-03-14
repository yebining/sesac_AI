import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import utils

import numpy as np
import pandas as pd


# 데이터 입력
df = pd.read_csv('iris.csv', names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])

dataset = df.values
X = dataset[:,0:4] # 속성
X_np = np.array(X, dtype=np.float)
Y_obj = dataset[:,4] # 정답 클래스

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

Y_encoded = utils.to_categorical(Y)


model = tf.keras.Sequential()	# 모델 선언
model.add(layers.Dense(16, input_dim=4, activation='relu'))	
model.add(layers.Dense(3, activation='softmax'))	

model.compile(loss='mean_squared_error', 
              optimizer='sgd', 
              metrics=['acc'])

model.fit(X_np, Y_encoded, epochs=50, batch_size=1)

print("\n Accuracy: %.4f" % (model.evaluate(X_np, Y_encoded)[1]))


