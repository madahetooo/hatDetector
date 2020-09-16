import os

import numpy as np
import tensorflow as tf

from data.data_loader import get_dataset
from models.efficient_det import get_model
from utils.exp_utils import prepare_experiment
from utils.model_utils import create_submission


def train(cf):


# Write your training script here


model_path = os.path.join('snapshots', sorted(
    os.listdir('snapshots'), reverse=True)[0])

model = models.load_model(model_path, backbone_name='resnet50')
model = models.convert_model(model)

# TODO: Create the datasets from data/dataloader.py
train_dataset = get_dataset(cf, mode="train")
val_dataset = get_dataset(cf, mode="validation")

# TODO: Create tensorboard and model_saver callbacks

# TODO: Build your model and load weights if you are resuming training

# TODO: Compile and train your model

pass


def test(cf):


# Write the test script here


def predict(image):
    image = preprocess_image(image.copy())
    # image, scale = resize_image(image)

    boxes, scores, labels = model.predict_on_batch(
        np.expand_dims(image, axis=0)
    )

    # boxes /= scale

    return boxes, scores, labels
    THRES_SCORE = 0.5


def draw_detections(image, boxes, scores, labels):
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < THRES_SCORE:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(image, b, color=color)

        caption = "{:.3f}".format(score)
        draw_caption(image, b, caption)

        def show_detected_objects(image_name):
    img_path = test_img + '/' + image_name

    image = read_image_bgr(img_path)

    boxes, scores, labels = predict(image)
    print(boxes[0, 0].shape)
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    draw_detections(draw, boxes, scores, labels)
    # plt.figure(figsize=(15,10))
    # plt.axis('off')
    # plt.imshow(draw)
    # plt.show()

    for img in li:
        show_detected_objects(img)
    preds = []


imgid = []
for img in tqdm(li, total=len(li)):
    img_path = test_img + '/' + img
    image = read_image_bgr(img_path)
    boxes, scores, labels = predict(image)
    boxes = boxes[0]
    scores = scores[0]
    for idx in range(boxes.shape[0]):
        if scores[idx] > THRES_SCORE:
            box, score = boxes[idx], scores[idx]
            imgid.append(img.split(".")[0])
            preds.append("{} {} {} {} {}".format(score, int(box[0]), int(
                box[1]), int(box[2] - box[0]), int(box[3] - box[1])))
            preds[0]
            sub = {"image_id": imgid, "PredictionString": preds}
sub = pd.DataFrame(sub)
sub.head()

sub_ = sub.groupby(["image_id"])['PredictionString'].apply(lambda x: ' '.join(x)).reset_index()
sub_

for idx, imgid in enumerate(samsub['image_id']):
    samsub.iloc[idx, 1] = sub_[sub_['image_id'] == imgid].values[0, 1]

samsub.head()

# TODO: Create the dataloader from data/dataloader.py
dataset = get_dataset(cf, mode="test")
# TODO: Load your saved model

# TODO: Run prediction

# TODO: Create the submission.csv file (it's better to do it in
# utils/model_utils.py) and call it here.
create_submission(results)
samsub.to_csv('/kaggle/working/submission.csv', index=False)
pass

if _name_ == "_main_":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="one out of: train / test",
    )

    parser.add_argument(
        "--exp_dir",
        type=str,
        default="experiments/develop/default",
        help="path to experiment dir. will be created if non existent.",
    )

    parser.add_argument(
        "--exp_results",
        type=str,
        default="experiments/develop/default_results",
        help="path to experiment source dirrectory.",
    )

    parser.add_argument(
        "--resume_to_checkpoint",
        default=False,
        action="store_true",
        help="used to resume a previous training from --exp_results",
    )

    args = parser.parse_args()

    # Current directory
    SOURCE_DIR = os.path.dirname(os.path.realpath(_file_))

    # TODO:

    if args.mode == "train":
        # TODO: Load your config file from exp_source and copy it in
        # exp_results and prepare the necessary folders for the experiment
        # for e.g.:
        # create the folder for tensorboard logs and a folder for checkpoints
        # it's better to do this in utils/exp_utils.py and call it here
        cf = prepare_experiment(args)
        train(cf)

    elif args.mode == "test":
        # TODO: Load your config file from exp_results and prepare the
        # necessary folders for the experiment for e.g.:
        # create the folder for plotting random images with bounding boxes
        # it's better to do this in utils/exp_utils.py and call it here
        cf = prepare_experiment(args)
        test(cf)
    else:
        raise RuntimeError("mode specified in args is not implemented...")