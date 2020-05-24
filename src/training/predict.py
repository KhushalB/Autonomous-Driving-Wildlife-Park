#!/usr/bin/env python3
# @author: Khushal Brahmbhatt

import argparse
import configparser
import os
import sys

import cv2
from keras.models import load_model
import numpy as np
import pandas as pd

from augment_data import AugData

# Parse inputs
parser = argparse.ArgumentParser(
    description="Specify test data input and pre-trained model to predict from.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--data", required=True,
                    help="Input test data file.")
parser.add_argument("-m", "--model", required=True,
                    help="Input pre-trained model file.")
args = parser.parse_args()

if os.path.isfile(args.data):
    dataset_file = args.data
    print("[INFO] Found test data: {0}".format(dataset_file))
else:
    print("[ERROR] Test data not found. Wrong filename or filepath.")
    sys.exit(1)

if os.path.isfile(args.model):
    model = load_model(args.model)
    print("[INFO] Loading pre-trained model: {0}".format(args.model))
else:
    print("[ERROR] Model file not found. Wrong filename or filepath.")
    sys.exit(1)

# Load config params
abs_path = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(abs_path, "../../config/train_config.ini")

try:
    with open(config_file, 'r') as f:
        config = configparser.ConfigParser()
        config.read_file(f)
        print("[INFO] Loading config file: {0}".format(config_file))
except IOError:
    print("[ERROR] Config file not found.")
    sys.exit(1)

scale_steer = config.getfloat("data_aug", "scale_steer")

# Load test data
print("[INFO] Loading test data...")
df_data = pd.read_csv(dataset_file)
test_data = [AugData.load_img(img_path) for img_path in df_data["image"]]
test_data = np.asarray(test_data)
actual = np.array(df_data["steer_angle"])

# Run Prediction
print("[INFO] Predicting steering angle on {0} samples...".format(test_data.shape[0]))
predictions = model.predict(test_data, verbose=1)
predictions = predictions * scale_steer  # to remove scaling previously applied on the steering angles

# Scroll through images with predicted steering angles vs actual steering angles
index = 0
cv2.namedWindow("actual vs predicted")
print("Loading data... Press 'm' for next image, 'n' for previous image, and 'Esc' or 'q' to quit.")
while True:
    img = AugData.load_img(df_data["image"][index])
    img = AugData.preprocess_img(img, width=1200, height=396)
    actual_angle = actual[index]
    predicted_angle = predictions[index]
    cv2.putText(img, "Actual steering angle: {0}".format(actual_angle), (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, "Predicted steering angle: {0}".format(predicted_angle), (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("actual vs predicted", img)
    key = cv2.waitKey(100)

    if key == ord("m"):
        index += 1
    elif key == ord("n"):
        index -= 1
    elif key == 27 or key == ord("q"):  # escape key or q
        break
cv2.destroyAllWindows()
