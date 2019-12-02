# Manish Balakrishnan
# EE551 Final Project

# MNIST Digit Hand Classifier

# Please refer to github and Readme for background on project and how to run properly.



# ********************** Libraries used ***********************************
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
import random
import os
# ********************** Libraries used ***********************************


def test():
    nmodel = load_model("model.h5")

    # Outputting a summary of the model detailing various layers.

    nmodel.summary()

    # Loading the dataset into various portions and then randomly taking an image from the datset and storing it in list
    # to test

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    my_randoms = random.sample(range(x_test.shape[0]), 9)
    print(my_randoms)

    # Keeping track of which values were classified correctly/incorrectly

    correct = []
    wrong = []

    # Plotting the randomly selected images in a 3x3 matrix and displaying the predicted value
    # If the predicted value matches the actual value it is appended to the appropriate list

    fig, axis = plt.subplots(3, 3, figsize=(32, 32))
    for i, ax in enumerate(axis.flat):
        ax.imshow(x_test[my_randoms[i]], cmap='Greys')
        pred = nmodel.predict(x_test[my_randoms[i]].reshape(1, 28, 28, 1))
        val = pred.argmax()
        ax.set(title="\n\nPredicted value= {}".format(val) )
        actual = y_test[my_randoms[i]]
        # print(actual)

        if (val == actual):
            correct.append(val)
        else:
            wrong.append(val)

    print(len(correct),'/9 images shown have been classified correctly.')

    plt.show()


if __name__ == '__main__':

    print("\nWelcome to testDigits.py. The digit classifier model will be tested against randomly selected")
    print("images and compared with the known values.\n\n")
    input("Press any key to continue.")

    # checks if a model.h5 file is present.

    if os.path.isfile('model.h5'):
        test()

    else:
        print("Trained Model not found!")
        print("Run classifier.py or download model.h5 from repository and place in current directory.")

