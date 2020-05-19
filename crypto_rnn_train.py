import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint


pickle_in = open("X_train.pickle","rb")
X_train = pickle.load(pickle_in)

pickle_in = open("y_train.pickle","rb")
y_train = pickle.load(pickle_in)

pickle_in = open("X_test.pickle","rb")
X_validation = pickle.load(pickle_in)

pickle_in = open("y_test.pickle","rb")
y_validation = pickle.load(pickle_in)


train_x = np.asarray(X_train)
train_y = np.asarray(y_train)
validation_x = np.asarray(X_validation)
validation_y = np.asarray(y_validation)


# Architecture of the Model
model = Sequential()
model.add(LSTM(128,input_shape = (X_train.shape[1:]),return_sequences=True))
model.add(Dropout(0.2))
# Normalizes activation outputs
model.add(BatchNormalization())

model.add(LSTM(128))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2,activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

history = model.fit(
    train_x, train_y,
    batch_size=64,
    epochs=10,
    validation_data=(validation_x, validation_y),
)