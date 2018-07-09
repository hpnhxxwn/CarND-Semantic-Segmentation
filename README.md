# Semantic Segmentation
### Introduction
In this project, I implemented a Fully Convolutional Network (FCN) on top of the last layer of vgg16 to label the pixels of a road in images.

### Setup
##### GPU
AWS g3.x4large
##### Frameworks and Packages
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Architecture
I picked a pre-trained VGG-16 network and added a de-convolution layer. The last fully connected layer to is converted to a 1x1 convolution and the depth is set to the number of classes (road and not-road). Performance is improved via two approach:
1. Upsample convolved layer 7 (the last layer).
2. Scale the output of layer 3 and layer 4 before performing 1x1 convolution on them.
3. Skip connections. Scaled and 1x1-convolved layer 4 is added element-wisely to upsampled 1x1-convolved layer 7, and then the sum is further upsampled and added element-wisely to scaled and 1x1-convolved layer 3. Each convolution and transpose convolution layer includes a kernel initializer and L2 regularizer.

### Optimizer
The loss function is cross entropy, and the optimizer is Adam optimizer.

### Training
The scaling factors for the outputs from layer 3 and layer 4 are 0.0001 and 0.01. Without the scaling factors the training will converge much slower and the result is worse.
keep_prob: 0.5
learning_rate: 0.0001
epochs: 50
batch_size: 5

### Results
Loss per batch is below 0.200 after one epoch, and below 0.80 after ten epochs. Average loss per batch at epoch 20 is below 0.055, at epoch 30 is below 0.030, at epoch 40 is below 0.020, and at epoch 50 around 0.016.

### Samples
Below are a few sample images from the output of the fully convolutional network, with the segmentation class overlaid upon the original image in green.
![img1]()
![img2]()
![img3]()
![img4]()
![img5]()
![img6]()
![img7]()
![img8]()
![img9]()
![img10]()
