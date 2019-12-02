# Manish Balakrishnan
# EE551 Final Project

# MNIST Digit Hand Classifier

# Please refer to github and Readme for background on project and how to run properly.


# ********************** Libraries used ***********************************
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPool2D  # MaxPooling2D
from keras.models import load_model
from random import randint
# ********************** Libraries used ***********************************


def trainModel():
    # Keras provides the mnist dataset already built into the package
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # dividing the dataset into a set used for training and another for testing
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print('Number of images in x_train', x_train.shape[0])
    print('Number of images in x_test', x_test.shape[0])

    neuralNetModel(x_train, y_train, x_test, y_test, input_shape)


# creating a sequential neural network model
def neuralNetModel(x_train, y_train, x_test, y_test, input_shape):
    # Creating a model of type sequential that consists of a convolution layer, Max-pooling layer
    # Flattening layer (to flatten the layers), dropout and dense layer

    print("Creating Neural Network")
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation='relu'))  # tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))  # tf.nn.softmax))

    # Compiles the above model using the adam optimer, a loss function and accuracy metrics
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Epoch defines how many times the classifier is run. A higher epoch will provide better accuracy
    # but will also take longer and more demanding of the computer
    model.fit(x=x_train, y=y_train, epochs=5)

    # prints a summary of the model, detailing the different layers and parameters
    print('\n\n')
    model.summary()

    # evaluates the trained model with the known values to output a % accuracy
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # save the created model as a .h5 file
    model.save("model.h5")
    print("Classifier model saved.")

    random = randint(0, x_test.shape[0])
    plt.imshow(x_test[random].reshape(28, 28), cmap='Greys')
    pred = model.predict(x_test[random].reshape(1, 28, 28, 1))
    print("Predicted value is: ", pred.argmax())
    plt.title('Predicted value= {}'.format(pred.argmax()))
    plt.show()


#
# def test():
#     nmodel = load_model("model.h5")
#     # load_model()
#     nmodel.summary()
#     (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#     my_randoms = random.sample(range(x_test.shape[0]), 9)
#     print(my_randoms)
#
#     fig, axis = plt.subplots(3, 3, figsize=(32, 32))
#     for i, ax in enumerate(axis.flat):
#         ax.imshow(x_test[my_randoms[i]], cmap='Greys')
#         pred = nmodel.predict(x_test[my_randoms[i]].reshape(1, 28, 28, 1))
#         val = pred.argmax()
#         ax.set(title="\n\nPredicted value= {}".format(val))
#     plt.show()

if __name__ == '__main__':

    print("Welcome to classifier.py. Shortly, the MNIST hand drawn digit dataset will be run through a sequential")
    print("classifier, so that any handdrawn digits from the dataset can be analyzed and deciphered")
    input("Press any key to continue.")

    # starts the classifier training.
    trainModel()

    # test()
