from keras.datasets import mnist # download mnist data and split into train and test sets
import matplotlib.pyplot as plt # plot the first image in the dataset
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from tensorflow import keras


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

    def load_data(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_train.reshape(60000, 28, 28, 1)
        self.x_test.reshape(10000, 28, 28, 1)
        # one-hot encode target column
        self.y_train = to_categorical(y_train)
        self.y_test = to_categorical(y_test)


    def save_model(self):
        self.model.save(self.path)

    def build_model(self):
        self.__create_model()
        self.__compile_model()

    def __create_model(self):
        self.model = Sequential()  # add model layers
        self.model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(28, 28, 1)))
        self.model.add(Conv2D(32, kernel_size=3, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(10, activation='softmax'))

    def __compile_model(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self):
        if self.model is None:
            raise Exception("model None!")
        self.model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=2)

    def load_model(self, model):
        if model is None:
            keras.models.load_model(self.path)

    def predict(self):
        return self.model.predict(self.x_test[:4])

def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    plt.imshow(X_train[0])
    plt.show()
    # reshape data to fit model
    X_train = X_train.reshape(60000, 28, 28, 1)
    X_test = X_test.reshape(10000, 28, 28, 1)

    # one-hot encode target column
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    print(y_train[0])

    # create model
    model = Sequential()  # add model layers
    model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape = (28, 28, 1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    # compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2)

    # predict first 4 images in the test set
    print(model.predict(X_test[:4]))

    # actual results for first 4 images in test set
    print(y_test[:4])


def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    plt.imshow(X_train[0])
    plt.show()
    # reshape data to fit model
    X_train = X_train.reshape(60000, 28, 28, 1)
    X_test = X_test.reshape(10000, 28, 28, 1)

    # one-hot encode target column
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    print(y_train[0])

    # create model
    model = Sequential()  # add model layers
    model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape = (28, 28, 1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    # compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2)

    # predict first 4 images in the test set
    print(model.predict(X_test[:4]))

    # actual results for first 4 images in test set
    print(y_test[:4])

def main2():
    raise Exception()

def test_detector():
    detector = Detector('./')

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    detector.load_data(X_train, y_train, X_test, y_train)
    detector.build_model()
    detector.train_model()
    detector.save_model()
    print(detector.predict())


if __name__ == '__main__':
    # main()
    test_detector()
