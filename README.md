# CIFAR 10 Image Classification

In this notebook, I am going to classify images from the CIFAR-10 dataset. The dataset consists of airplanes, dogs, cats, and other objects. You'll preprocess the images, then train a convolutional neural network on all the samples. The dataset can downloaded by clicking [here](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

## The Dataset

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

![](dataset.png)

The classes are completely mutually exclusive. There is no overlap between automobiles and trucks. "Automobile" includes sedans, SUVs, things of that sort. "Truck" includes only big trucks. Neither includes pickup trucks.

## Update - ResNet
Previously the model was limited to a test accuracy of around 80%. Adding more layers and making the model more deeper would not actually improve the performance of the model due to vanishing gradient. The ResNet architecture solves this problem. It consists of skips connections which skips the next layers and transfers the output to a deeper layer.

![](ResNet/model_plot.png)

I trained the model for 130 epochs and the metrics are below.

    Test Loss: 0.3836
    Test Accuracy: 90.35 %

### ResNet Model loss and accuracy

![](ResNet/ResNet_metrics.png)

## Predictions
- ### On Test Data
![](ResNet/Test_pred.png)

- ### On images from internet
![](ResNet/prediction1.png)
![](ResNet/prediction2.png)
![](ResNet/prediction3.png)
![](ResNet/prediction4.png)
![](ResNet/prediction5.png)
![](ResNet/prediction6.png)

## Previous Model Metrics
### Final Testing Results

    Test Loss: 0.5822
    Test Accuracy: 81.330 %

### Model Loss and Accuracy

![](model_metrics.png)

## Dependencies

- Jupyter Notebook v6.4.8
- Python v3.9.7
- Tensorflow v2.9.1
- Keras v2.9.0
- Numpy v1.22.2
- Pandas v1.4.2
- Matplotlib v3.2.2