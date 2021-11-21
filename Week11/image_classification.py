import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

"""
 Data preparation
    1 Load data
    2 Check for null and missing values
    3 Normalization
    4 Reshape
    5 Label encoding
    6 Split training and valdiation set

 CNN
    1 Define the model
    2 Set the optimizer and annealer
    3 Data augmentation

 Evaluate the model
    1 Training and validation curves
    2 Confusion matrix

 Prediction
    1 Predict and Submit results
"""

"""
Download and prepare the CIFAR10 dataset
The CIFAR10 dataset contains 60,000 color images in 10 classes, 
with 6,000 images in each class. The dataset is divided 
into 50,000 training images and 10,000 testing images. 
The classes are mutually exclusive and there is no overlap between them.
"""

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

"""
To verify that the dataset looks correct, 
let's plot the first 25 images from the training set and display the class name below each image:
"""
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

"""
The 6 lines of code below define the convolutional base using a common pattern: a stack of Conv2D and MaxPooling2D layers.

As input, a CNN takes tensors of shape (image_height, image_width, color_channels), 
ignoring the batch size. If you are new to these dimensions, 
color_channels refers to (R,G,B). 
In this example, you will configure your CNN to process inputs of shape (32, 32, 3), 
which is the format of CIFAR images. You can do this by passing the argument input_shape to your first layer.
"""
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

"""
Let's display the architecture of your model so far:
(describe and interpret your model in your report )
"""

model.summary()

"""
Add Dense layers on top
To complete the model, you will feed the last output tensor 
from the convolutional base (of shape (4, 4, 64)) 
into one or more Dense layers to perform classification. 
Dense layers take vectors as input (which are 1D), 
while the current output is a 3D tensor. First, 
you will flatten (or unroll) the 3D output to 1D, 
then add one or more Dense layers on top. 
!!!! CIFAR has 10 output classes !!!! so you use a final Dense layer with 10 outputs.
"""

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

"""
Compile and train the model
"""

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)
