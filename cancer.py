from keras.models import Sequential, load_model
from keras.layers import Dense

# 필요 라이브러리 호출
import numpy as np
import tensorflow as tf

# 실행 시마다 같은 결과값 도출을 위한 시드 설정
np.random.seed(0)
tf.random.set_seed(0)

# csv 파일을 읽어 ','기준으로 나눠 Dataset에 불러오기
Dataset = np.loadtxt("ThoraricSurgery.csv", delimiter=",")

# 환자 정보는 0-16번(17개)까지이므로 해당 부분까지 X에 담기
X = Dataset[:, 0:17]
# 수술 후 결과는 마지막 17번은 클래스로 Y에 담기
Y = Dataset[:, 17]

# 딥러닝 모델 구조 설정(3개층, 속성이 17개 input 값, relu와 sigmoid 활성화 함수 이용)
model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 딥러닝 실행(오차함수는 평균제곱오차, 최적화함수는 adam 이용)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=30, batch_size=5)

# 예측
test_x = np.array([[75,3,4.6,3.28,1,0,0,0,1,0,11,0,0,0,1,0,55]])
print(model.predict(test_x))
