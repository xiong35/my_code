import numpy as np
np.random.seed(7)  
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], -1) / 255. 
X_test = X_test.reshape(X_test.shape[0], -1) / 255.  
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)


model = Sequential() 

model.add(Dense(666,activation='relu',input_shape=(784,)))
model.add(Dense(666,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss=
'categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_batch,batch_size=100,epochs=20)

result = model.evaluate(x_train,y_train,batch_size = 10000)

print('test loss: ', recule[0])
print('test accuracy: ', result[1])
