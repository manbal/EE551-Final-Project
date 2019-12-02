### Code

``` training.py``` - Contains the main python code used to train the model\
``` test.py ``` - Contains test code to test the model against randomly selected images and compare to its actual values\
``` model.h5``` - Contains the model that was trained by training.py

If test.py is being run first, make sure to download ```model.h5``` and place in the same directory as ```test.py```. If ```training.py``` is being run first, the program will generate and save the generated model after which test.py can be run.


### Results

The program is able to successfully classify an image and output an integer based on the image. The classification is fairly accurate with an accuracy of 98%. Although this is very accuracte, for application purposes, it could be higher, because it means on average it misclassifies 2 out of 100 images. The image below shows the classifier as it is trained on the mnist dataset, while running ```training.py```. 

![2](https://user-images.githubusercontent.com/7034609/69936063-a560d480-14a4-11ea-8c5e-8306223a2151.PNG)

Once the training is complete, a random image from the dataset is taken and inputted through the model to display the predicted integer, as shown below.

![Capture](https://user-images.githubusercontent.com/7034609/69938194-a6950000-14aa-11ea-82dc-aa0bf91e012a.PNG)


When the ```test.py``` is run, a 3x3 matrix of randomly selected images that are fed through the model are shown along with their predicted values. Such an output would look like the following.

![3](https://user-images.githubusercontent.com/7034609/69938348-08ee0080-14ab-11ea-8632-6bd63ba7b5e8.PNG)
