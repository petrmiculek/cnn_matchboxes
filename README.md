# cnn_matchboxes

Experimenting with Convolutional Neural Networks as a part of my Bachelor Thesis at BUT FIT

Detecting keypoints in images

Own dataset - photos of blocks of matchboxes (not included)

![annotated-photo.png](https://i.imgur.com/ntyj3CR.png)

Photos annotated with >2000 keypoints of the following **8** categories:
* corner-top  (Darker Blue)
* edge-side   (Lighter Blue)
* intersection-side  (Pink)
* corner-bottom  (Red)
* edge-bottom  (Light Green)
* edge-top   (Yellow)
* intersection-top (Blue-Green)

+extra category
* background

(background is randomly sampled from parts of image far enough from the keypoints)

The resulting training set is made up of NxN cutouts with given keypoint in the middle.

An outdated version of the cutouts dataset can be found here: https://nextcloud.fit.vutbr.cz/s/qSHJxRGe9o5kTbK/download


Approach v1:

Classifying 64x64 regions extracted from images

Simple Sequential model
- (Convolution(3x3), ReLU, BatchNorm, MaxPool(2x2)) * N
- (64, 64, 3) -> (1, 1, 8)
- Sliding window -> heatmap of activations
  - slow, not scalable
  - low output resolution (stride 64)

Approach v2:

Fully Convolutional Network

- (32, 32, 3) -> (1, 1, 8)
( still training on classification )

- (Convolution(3x3), ReLU, BatchNorm) * N
  Only resolution change is the convolution crop
  => output dimensions are K pixels smaller than input (K = 31, currently)
  
- Inference outputs high resolution heatmap of class activations.
  ( per pixel classification )
  Inference is run on whole image at once.

![output-heatmap.png](https://lh4.googleusercontent.com/Acxpa6797yTrwP3wLLgbt0L0Z-HzTCxd65AQ83dDe5WchKmzWOcApwktG7xhG3Z5Vy9MTh2MTAVQNkYKf04FheizYsE4FxWsntm4bG9H=s1000)

Further plans:

- Instance "segmentation"

- Counting of boxes

- Multi-view prediction fusion
