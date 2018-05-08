from keras.models import Sequential
from keras.layers import *
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import numpy as np
import string

from util import *

row, col = 28, 28
inputShape = (row, col, 1)
batchSize = 100
epochNum = 10

# load dataset
X, Y = get_data(100000)

# split the data into training (70%) and testing (30%)
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, train_size=0.7, random_state=1)

X_train = X_train.reshape(X_train.shape[0], row, col, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], row, col, 1).astype('float32')

# one hot encode outputs
Y_train = np_utils.to_categorical(Y_train)
Y_test_2 = np_utils.to_categorical(Y_test)

# create model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=inputShape, kernel_initializer='glorot_uniform',bias_initializer='zeros',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), kernel_initializer='glorot_uniform',bias_initializer='zeros',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), kernel_initializer='glorot_uniform',bias_initializer='zeros',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(Y_test_2.shape[1], activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test_2), epochs=epochNum, batch_size=batchSize, verbose=2)

# Final evaluation of the model
pred_y = model.predict(X_test, batchSize, verbose=0, steps=None)
pred_y = np.argmax(pred_y, axis=1)

#visualization configuration 
vtest = False
n_show_result = 5
display_error = False
visualize(vtest, display_error, n_show_result, pred_y, X_test, Y_test, None)

scores = model.evaluate(X_test,Y_test_2, verbose=0)
print("Accuracy: %f" % scores[1])
