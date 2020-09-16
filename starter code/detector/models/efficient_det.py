import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image, ImageDraw
from ast import literal_eval
import matplotlib.pyplot as plt
import urllib
from tqdm.notebook import tqdm
import tensorflow as tf
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

train_img = "/Users/eslammedhat/WORK/Kortobaa/ML Training/starter code/data/raw/train"
test_img = "/Users/eslammedhat/WORK/Kortobaa/ML Training/starter code/data/raw/test"
train_csv = '/Users/eslammedhat/WORK/Kortobaa/ML Training/starter code/data/raw/train/_annotations.csv'
sample_submission = "/Users/eslammedhat/WORK/Kortobaa/ML Training/starter code/data/raw/sample_submission.csv"

with open("classes.csv", "w") as file:
    file.write("helmet,0")

    print(f"Total Bboxes: {train.shape[0]}")
    train['width'].unique() == train['height'].unique() == [1024]
    train['heightBB'] = train['heightBB'].astype(float)
train['widthBB'] = train['widthBB'].astype(float)
unique_images = train['filename'].unique()
len(unique_images)
num_total = len(os.listdir(train_img))
num_annotated = len(unique_images)

print(f"There are {num_annotated} annotated images and {num_total - num_annotated} images without annotations.")


def get_model(cf):
    # TODO: build your model

    PRETRAINED_MODEL = './snapshots/_pretrained_model.h5'


#### OPTION 1: DOWNLOAD INITIAL PRETRAINED MODEL FROM FIZYR ####
URL_MODEL = 'https://github.com/fizyr/keras-retinanet/releases/download/0.5.0/resnet50_coco_best_v2.1.0.h5'
import requests

requests.get(URL_MODEL, PRETRAINED_MODEL)
print('Downloaded pretrained model to ' + PRETRAINED_MODEL)

model_path = os.path.join('snapshots', sorted(os.listdir('snapshots'), reverse=True)[0])

model = models.load_model(model_path, backbone_name='resnet50')
model = models.convert_model(model)

pass


def show_images(images, num=5):
    images_to_show = np.random.choice(images, num)

    for image_id in images_to_show:

        image_path = os.path.join(train_img, image_id)
        image = Image.open(image_path)

        # get all bboxes for given image in [xmin, ymin, width, height]
        bboxes = [literal_eval(box) for box in train[train['filename'] == image_id]['bbox']]

        # visualize them
        draw = ImageDraw.Draw(image)
        for bbox in bboxes:
            draw.rectangle(
                [train['xmin'], train['ymin'], train['xmin'] + train['widthbb'], train['ymin'] + train['heightbb']],
                width=3)

        # plt.figure(figsize = (15,15))
        # plt.imshow(image)
        # plt.show().