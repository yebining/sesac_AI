from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 필요한 라이브러리를 불러옵니다.
import numpy as np

# 준비된 수술 환자 데이터를 불러들입니다.
Data_set = np.loadtxt("ThoraricSurgery.csv", delimiter=",")

X = Data_set[:,0:17] # 속성
Y = Data_set[:,17] # 정답 클래스

model = Sequential()	# 모델 선언
model.add(Dense(30, input_dim=17, activation='relu'))	
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

model.fit(X, Y, epochs=30, batch_size=10)
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))


