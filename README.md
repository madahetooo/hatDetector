# hatDetector
This repository  is to build a model that detects people, heads, and hardhats in images. Each image can contain multiple people, multiple heads and multiple hardhats.  The solution will help build a safety app that will alert the user if there are workers in the field that do not comply with safety rules.
This Notebook is for Training Purpose of Keras RetinaNet.
RetinaNet is very slow as compared to F-RCNN so I've kept epochs and steps per epoch small for fast commiting purpose.

Step-1 : Installing Keras-RetinaNet
Step-2 : Let's look at the data
Step-3 : EDA
Step-4 : Visualizing images

------------------
What can we tell from visualizations:

there are plenty of overlappind bounding boxes
all photos seem to be taken vertically
all plants are can be rotated differently, there is no single orientation. this means that different flip and roration augmentations should probably help
colors of wheet heads are quite different and seem to depend a little bit on the source
wheet heads themselves are seen from very different angles of view relevant to the observer
----------------------
Step-5 : Preprocessing Data for Input to RetinaNet

Step-6 : Preparing Files to be given for training
Annotation file contains all the path of all images and their corresponding bounding boxes
Class file contains the number of classes but in our case it is just 1 (Wheat)
---------------------
Step-7 : Downloading the pretrained model

Model Parameters

Step-8 : Training Model.
Step-9 : Predictions.
--------

