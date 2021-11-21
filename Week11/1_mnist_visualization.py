from tensorflow.keras.datasets import mnist
from matplotlib import pyplot

"""
The MNIST dataset is an acronym that stands for the Modified National Institute of Standards and Technology dataset.

It is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.

The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values 
from 0 to 9, inclusively.

It is a widely used and deeply understood dataset and, for the most part, is “SOLVED.”
 Top-performing models are deep learning convolutional neural networks that achieve a classification accuracy of above 99%, 
 with an error rate between 0.4 %and 0.2% on the hold out test dataset.

The example below loads the MNIST dataset using the Keras API and creates 
a plot of the first nine images in the training dataset.


http://yann.lecun.com/exdb/mnist/

https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/

"""

# load dataset
(trainX, trainy), (testX, testy) = mnist.load_data()
# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
# plot first few images
for i in range(9):
    # define subplot
    pyplot.subplot(330 + 1 + i)
    # plot raw pixel data
    pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))
# show the figure
pyplot.show()


"""
Although the MNIST dataset is effectively solved, 
it can be a useful starting point for developing and practicing 
a methodology for solving image classification tasks using convolutional neural networks.

Instead of reviewing the literature on well-performing models on the dataset,
 we can develop a new model from scratch.

The dataset already has a well-defined train and test dataset that we can use.
"""
