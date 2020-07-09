# LeNet-MNIST-

The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.
It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting.

ABOUT PROJECT :
In this model I have used Convolutional Neural Network, Tensorflow, Keras and matplotlib. We have to import tensorflow and matplotlib. Now, Keras comes with tensorflow so you only have to install tensorflow in your system.

This project is about Predicting the numbers present in the MNIST dataset as much accurate as possible. In this we are using 80% of the dataset for training and 20% of the dataset for validation and testing. 

This project is a transformation learning kind of project because I followed LeNet- Yann LeCunn(LeNet-5 paper), which is a 7 layer neural network.
In mnist dataset images are of size 28x28 and are of GRAY SCALE and in 2D form but for using those images in my model I have reshaped those images into 32x32 and into 3D images.
But i want the output in 2D so i have to reshape those 3D outputs in 2D and then plot it by the help of matplotlib library.

For Importing 'Tensorflow' and 'matplotlib' in your system use these pip commands :
1) pip install tensorflow     
2) pip install matplotlib

Note: You have to download the version of tensorflow according to the Python version of yours. 
