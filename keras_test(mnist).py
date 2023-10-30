#-----------------------------------------------------------------------------------------------------------------------
from data import load_mnist
from keras.models import Sequential
from keras import utils
from keras.layers import Dense, Activation
import tensorflow as tf
import keras
import numpy as np
#-----------------------------------------------------------------------------------------------------------------------
np.random.seed(0) #일단보류
tf.random.set_seed(0)
#data: (훈련 이미지,훈련 답), (테스트 이미지,테스트 답)
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False) #1차원 배열로, 0~1로 정규화
t_train = utils.to_categorical(t_train, 10)
t_test = utils.to_categorical(t_test, 10)
#-----------------------------------------------------------------------------------------------------------------------
model=Sequential()
model.add(Dense(16,input_shape=(784,),activation='relu',kernel_initializer='he_normal'))
model.add(Dense(16,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,t_train,batch_size=16,epochs=30)

model.evaluate(x_test, t_test)
#-----------------------------------------------------------------------------------------------------------------------
#차이: He초기화,k/dis

