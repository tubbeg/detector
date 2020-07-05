from keras.datasets import mnist, cifar10 # download mnist data and split into train and test sets
import matplotlib.pyplot as plt # plot the first image in the dataset
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from tensorflow import keras
import cv2
import numpy as np

"""
CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
There are 50000 training images and 10000 test images.
"""

class Detector:
    def __init__(self, path):
        if path is None:
            raise Exception("path None!")
        self.model = None
        self.path = path
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.res = 50
        self.output_number = 2
        self.index_size = 6
        self.dimensions = 3  # set to 1 for grayscale images

    def load_data(self, x_train, y_train, x_test, y_test):
        # X_train, y_train), (X_test, y_test) = mnist.load_data()
        # reshape data to fit model
        X_train = x_train
        X_test = x_test


        X_train = X_train.reshape(self.index_size, self.res, self.res, self.dimensions)
        X_test = X_test.reshape(self.index_size, self.res, self.res, self.dimensions)

        # one-hot encode target column
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        self.x_train = X_train
        self.y_train = y_train
        self.x_test = X_test
        self.y_test = y_test

    def save_model(self, file):
        if file is None:
            self.model.save(self.path + "model")
        else:
            self.model.save(file)

    def build_model(self):
        self.__create_model()
        self.__compile_model()

    def __create_model(self):
        self.model = Sequential()  # add model layers
        self.model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(self.res, self.res, 3)))
        self.model.add(Conv2D(32, kernel_size=3, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(self.output_number, activation='softmax'))

    def __compile_model(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self):
        if self.model is None:
            raise Exception("model None!")
        self.model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=2)

    def load_model(self, model):
        if model is None:
            raise Exception("model None!")
        self.model = keras.models.load_model(model)

    def predict(self, image_array):
        return self.model.predict(image_array)


def test_detector():
    detector = Detector('./')
    # real interesting guide https://github.com/jerett/Keras-CIFAR10/blob/master/softmax.ipynb
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    detector.load_data(X_train, y_train, X_test, y_train)
    detector.build_model()
    detector.train_model()
    detector.save_model("./my_model2")
    #detector.load_model("./my_model")
    print(detector.predict())


def test_detector2():
    detector = Detector('./')
    # real interesting guide https://github.com/jerett/Keras-CIFAR10/blob/master/softmax.ipynb
    #(X_train, y_train), (X_test, y_test) = mnist.load_data()
    y_train = [0, 1, 0, 1, 0, 1]
    y_test = y_train
    im1 = cv2.imread("square.jpg")
    im2 = cv2.imread("circle.jpg")
    X_train = np.array([im1, im2, im1, im2, im1, im2])
    X_test = X_train

    detector.load_data(X_train, y_train, X_test, y_train)
    detector.build_model()
    detector.train_model()
    detector.save_model("./my_model2")
    #detector.load_model("./my_model")
    print(detector.predict(X_train[:6]))


def main():
    test_detector2()


if __name__ == '__main__':
    main()
    #(X_train, y_train), (X_test, y_test) = mnist.load_data()
    #plt.imshow(X_train[1])
    #print(type(X_train))
    #plt.show()

    #plt.imshow()
    #test_detector()