import numpy as np
import idx2numpy 
import matplotlib.pyplot as plt
import logistic_regression as lr

images = idx2numpy.convert_from_file('data/train-images.idx3-ubyte')
labels = idx2numpy.convert_from_file('data/train-labels.idx1-ubyte')

def filter_data(data, condition):
    images, labels = data

    new_images = images[labels == condition]
    new_labels = labels[labels == condition]
    return new_images, new_labels

train_images_0, train_labels_0 = filter_data((images, labels), condition=0)
train_images_1, train_labels_1 = filter_data((images, labels), condition=1)

train_images  = np.concatenate((train_images_0, train_images_1), axis=0)
train_labels = np.concatenate((train_labels_0, train_labels_1), axis=0)
test_images  = np.concatenate((train_images_0, train_images_1), axis=0)
test_labels = np.concatenate((train_labels_0, train_labels_1), axis=0)
