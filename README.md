# Exploration Into Machine Learning: Classifying Hand-Drawn Digits
### EE551 Final Project Fall 2019
Manish Balakrishnan <br />

### Introduction
Python offers many opportunites for Machine Learning applications. This project explores Python's Machine Learning packages to correctly identify digits from a dataset of thousands of handwritten images of digits. This project will use dtasets from MNIST("Modified National Institute of Standards and Technology") provided by Kaggle to train and test classification algorithms. A Convolutional Neural Network implemented using Keras. The skills learned from this project will be very educational as they are the basis for many computer vision and machine learning fundamentals.\
As I am very new to Machine Learning, this project will be helpful in understanding the fundamentals of machine learning

The scope and goals of this project are:
* Code written in Python
* Organized Github repository
* Inclusion of test code
* Reproducibility
* Neural Network Fundamentals
* Additional goals if achievable

The project will be completed by the deadline of 12/02/2019 Monday at 5pm ET

### Packages Necessary

* tensorflow
* matplotlib
* keras
* random

The packages should be installable by running the command:
```
pip install *package*
```
### Contents of Repository 

``` classifier.py``` - Contains the main python code used to train the model\
``` testDigits.py ``` - Contains test code to test the model against randomly selected images and compare to its actual values\
``` model.h5``` - Contains the model that was trained by classifier.py (found in /model directory)

If test.py is being run first, make sure to download ```model.h5``` and place in the same directory as ```test.py```. If ```training.py``` is being run first, the program will generate and save the generated model after which test.py can be run.

### Image Classification Theory
Image classification is the concept of taking an image and being able to classify it based on the contents of the image. This concept has now been acheivable through the use of Machine Learning. When an image is inputted to a computer, it is read as a bitmap of pixels. With the advent of machine learning, we are able to pass the image into different layers to pick up attributes/uinque features of the image in order to classify it. To acheive this, numerous layers are used, discussed below.  

#### Convolution Layer
The first layer where the features from the images are extracted. This layer helps to filter the image and reduce the size of the image. It is essentially a mathematical convultion conducted on the image.
  
#### Pooling Layer
Usually inserted after a convolution layer, this layer is used to reduce the spatial size of the output from the previous layer. It will reduce the 'dimensions' of the map without losing the information in them.

A graphical interpretation of the various layers can be seen below:
![](https://miro.medium.com/max/948/1*4GLv7_4BbKXnpc6BRb0Aew.png)

The above mention just some of the layers commonly used in image classification. 

  
### Acknowledgements <br />
The project will use datasets obtained from the Mnist library in the Keras package, an open-source neural network library. Additional information about the package and image classification concepts were learned from the websites below.
* https://www.tensorflow.org/
* https://www.tensorflow.org/guide/keras
* https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148 

