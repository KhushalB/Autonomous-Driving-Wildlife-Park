#!/usr/bin/env python3
# @author: Khushal Brahmbhatt

import argparse
import configparser
import os
import sys

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd

from augment_data import AugData
from model_def import NvidiaDave2


# Parse input and output files
parser = argparse.ArgumentParser(
    description="Specify dataset file to input.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--data", required=True,
                    help="Input dataset file.")
parser.add_argument("-s", "--save", default=None,
                    help="Specify file to save trained model to.")
args = parser.parse_args()

abs_path = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(abs_path, "../../config/train_config.ini")

if os.path.isfile(args.data):
    dataset_file = args.data
    print("[INFO] Found dataset: {0}".format(dataset_file))
else:
    print("[ERROR] Dataset not found. Wrong filename or filepath.")
    sys.exit(1)

if args.save and os.path.isdir(os.path.dirname(args.save)):
    save_model_file = args.save
else:
    dataset_name = os.path.splitext(os.path.basename(dataset_file))[0]
    save_model_file = os.path.join(abs_path, "models/{0}.h5".format(dataset_name))
    print("[INFO] No output file for saving trained model to given, or wrong filepath. Trained model will be saved to: "
          "{0}".format(save_model_file))

# Load config params
try:
    with open(config_file, 'r') as f:
        config = configparser.ConfigParser()
        config.read_file(f)
        print("[INFO] Loading config file: {0}".format(config_file))
except IOError:
    print("[ERROR] Config file not found.")
    sys.exit(1)

bias_steer_l = config.getfloat("data_aug", "bias_steer_l")
bias_steer_h = config.getfloat("data_aug", "bias_steer_h")
bias_steer_frac = config.getfloat("data_aug", "bias_steer_frac")
scale_steer = config.getfloat("data_aug", "scale_steer")
aug_prob = config.getfloat("data_aug", "aug_prob")
width = config.getint("img_dims", "width")
height = config.getint("img_dims", "height")
val_split = config.getfloat("train_params", "val_split")
learning_rate = config.getfloat("train_params", "learning_rate")
batch_size = config.getint("train_params", "batch_size")
epochs = config.getint("train_params", "epochs")

# Load data
print("[INFO] Generating dataset...")
df_dataset = pd.read_csv(dataset_file)

# Drop x% of samples with very large distribution to avoid bias e.g. straight sections of road
df_bias = df_dataset[df_dataset["steer_angle"].between(bias_steer_l, bias_steer_h)]
df_bias = df_bias.sample(frac=bias_steer_frac, random_state=1)
df_dataset = df_dataset.drop(df_bias.index)

# Scale steering angles to make most of the values between -1 and 1
df_dataset["steer_angle"] = df_dataset["steer_angle"] / scale_steer

# Get training and validation sets
df_val = df_dataset.sample(frac=val_split)
df_train = df_dataset.drop(df_val.index)

X_train = [AugData.load_img(img_path) for img_path in df_train["image"]]
Y_train = df_train["steer_angle"].values.tolist()

X_val = [AugData.load_img(img_path) for img_path in df_val["image"]]
X_val = np.array([AugData.preprocess_img(img, width, height) for img in X_val])
Y_val = np.array(df_val["steer_angle"])

# Augment training data
X_aug, Y_aug = AugData.flip_hor(X_train, Y_train, flip_prob=aug_prob)
X_train = X_train + X_aug
X_train = np.array([AugData.preprocess_img(img, width, height) for img in X_train])
Y_train = np.array(Y_train + Y_aug)

height = X_train.shape[1]
width = X_train.shape[2]
depth = X_train.shape[3]
print("[INFO] Training data shape: {0}".format(X_train.shape))
print("[INFO] Training labels shape: {0}".format(Y_train.shape))
print("[INFO] Validation data shape: {0}".format(X_val.shape))
print("[INFO] Validation labels shape: {0}".format(Y_val.shape))

datagen = ImageDataGenerator(preprocessing_function=AugData.rand_bright)

# Compile and train model
model = NvidiaDave2.build(rows=height, cols=width, channels=depth)
model.compile(optimizer=Adam(lr=learning_rate), loss='mse')
model.summary()

print("[INFO] Training network...")
trained_model = model.fit(datagen.flow(X_train, Y_train, batch_size=batch_size), epochs=epochs,
                          validation_data=(X_val, Y_val), verbose=1)
print("[INFO] Finished training. Serializing network...")
model.save(save_model_file)
