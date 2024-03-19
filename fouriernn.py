import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop

(mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = mnist.load_data()

train_images = mnist_train_images.reshape(60000,784)
test_images = mnist_test_images.reshape(10000,784)
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255

train_labels = keras.utils.to_categorical(mnist_train_labels,10)
test_labels = keras.utils.to_categorical(mnist_test_labels, 10)


n = len(train_images[0])
lim = 110
train_fourier = np.fft.fft(train_images[0], n)
psd = train_fourier * np.conj(train_fourier) / n
freq = np.arange(lim)

size_lim = 60
fourier_images = np.fft.fft(train_images)
small_f_images = fourier_images[:,:size_lim]
print(small_f_images.shape)

small_test_images = np.fft.fft(test_images)[:,:size_lim]

model = Sequential()
model.add(Dense(126, activation='relu', input_shape=(size_lim,)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
history = model.fit(small_f_images, train_labels, batch_size=100, epochs=13,verbose=2, validation_data=(small_test_images, test_labels))

score = model.evaluate(small_test_images, test_labels, verbose=0)
print('test accuracy: ', score[1])