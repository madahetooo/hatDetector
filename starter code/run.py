import os

import numpy as np
import tensorflow as tf

from data.data_loader import get_dataset
from models.efficient_det import get_model
from utils.exp_utils import prepare_experiment
from utils.model_utils import create_submission

def train(cf):
    # Write your training script here

    # TODO: Create the datasets from data/dataloader.py
    train_dataset = get_dataset(cf, mode="train")
    val_dataset = get_dataset(cf, mode="validation")

    # TODO: Create tensorboard and model_saver callbacks

    # TODO: Build your model and load weights if you are resuming training

    # TODO: Compile and train your model

    pass

def test(cf):
    # Write the test script here

    # TODO: Create the dataloader from data/dataloader.py
    dataset = get_dataset(cf, mode="test")
    # TODO: Load your saved model

    # TODO: Run prediction

    # TODO: Create the submission.csv file (it's better to do it in 
    # utils/model_utils.py) and call it here.
    create_submission(results)
    pass


if __name__ == "__main__":

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
    SOURCE_DIR = os.path.dirname(os.path.realpath(__file__))
    
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
