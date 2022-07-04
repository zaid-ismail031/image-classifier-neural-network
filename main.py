import cv2
import numpy
import matplotlib.pyplot as pyplot
from keras import models, layers
from keras.datasets import cifar10

(training_images, training_labels), (testing_images, testing_labels) = cifar10.load_data()
training_images, testing_images = training_images/255, testing_images/255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

for i in range(16):
    pyplot.subplot(4,4,i+1)
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.imshow(training_images[i], cmap=pyplot.cm.binary)
    pyplot.xlabel(class_names[training_labels[i][0]])

pyplot.show()

training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

model = models.load_model('image_classifier.model')



