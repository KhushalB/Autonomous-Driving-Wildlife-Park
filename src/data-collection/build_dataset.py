#!/usr/bin/env python3
# @author: Khushal Brahmbhatt

"""
Script to extract frames from video as image data for training, and build dataset. The workflow in this script involves:
1. Extract frames from video at 10fps, skipping over the specified bad data frames.
2. Image cropping, resizing
3. Match frame timestamp to corresponding closest timestamp of driving params from CAN log.
4. Save image and create dataset file.

Usage:
python3 build_dataset.py -i example-vid.avi
"""

import argparse
import configparser
import cv2
import errno
import numpy as np
import os
import pandas as pd
import sys
import time

# Parse input and output files
parser = argparse.ArgumentParser(
    description="Specify video input, video timestamps file, extracted driving params file, bad data frame file, "
                "directory to write frames to, dataset file, and config file. Video input is required. The rest of the "
                "input and output files can be retrieved automatically if they exist, with the correct naming and "
                "directory conventions used.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input", required=True,
                    help="Input video file to extract images from.")
parser.add_argument("-t", "--tsfile", default=None,
                    help="Input timestamps file corresponding to the video.")
parser.add_argument("-c", "--canlog", default=None,
                    help="Input can log csv file with extracted driving signals.")
parser.add_argument("-b", "--baddata", default=None,
                    help="Input bad data csv file with frames to skip over during frame extraction.")
parser.add_argument("-f", "--framesdir", default=None,
                    help="Specify directory to extract frames to.")
parser.add_argument("-o", "--output", default=None,
                    help="Specify dataset csv file to write image paths and driving signals to.")
args = parser.parse_args()

abs_path = os.path.dirname(os.path.abspath(__file__))

# Get config file
config_file = os.path.join(abs_path, "../../config/dataset_config.ini")

# Get video file
if os.path.isfile(args.input):
    vid_file = args.input
    vidname = os.path.splitext(os.path.basename(vid_file))[0]
    print("Found video: '{0}'".format(os.path.basename(vid_file)))
else:
    print("Video not found. Wrong filename or filepath.")
    sys.exit(1)

# Get video timestamps file
if args.tsfile:
    ts_file = args.tsfile
else:
    ts_file = os.path.join(abs_path, "../../data/video-timestamps/{0}_ts.txt".format(vidname))
if os.path.isfile(ts_file):
    print("Found timestamps file: '{0}'".format(os.path.basename(ts_file)))
else:
    print("Timestamps file not found. Wrong filename or filepath.")
    sys.exit(1)

# Get driving can log file
if args.canlog:
    can_file = args.canlog
else:
    can_file = os.path.join(abs_path, "../../data/can-logs/{0}_canlog.csv".format(vidname))
if os.path.isfile(can_file):
    print("Found driving can log: '{0}'".format(os.path.basename(can_file)))
else:
    print("Driving can log not found. Wrong filename or filepath.")
    sys.exit(1)

# Get bad data file with frames to skip over
if args.baddata:
    bad_data_file = args.baddata
else:
    bad_data_file = os.path.join(abs_path, "../../data/bad_data.xlsx")
if os.path.isfile(bad_data_file):
    print("Found bad data file: '{0}'".format(os.path.basename(bad_data_file)))
else:
    print("Bad data file not found. Wrong filename or filepath.")
    sys.exit(1)

# Get directory to save extracted video frames to
if args.framesdir and os.path.isdir(args.framesdir):
    frames_dir = args.framesdir
    print("Extracted frames will be saved to directory: '{0}'".format(frames_dir))
else:
    frames_dir = os.path.join(abs_path, "../../data/frames/{0}".format(vidname))
    try:
        os.mkdir(frames_dir)
        print("Successfully created directory: '{0}'".format(frames_dir))
    except OSError as e:
        if e.errno == errno.EEXIST:
            print("Directory already exists. Extracted frames will be saved to directory: '{0}'".format(frames_dir))
        else:
            raise

# Get dataset file to write image paths and driving params to
if args.output and os.path.isdir(os.path.dirname(args.output)):
    dataset_file = args.output
    print("Dataset will be saved to the specified output file: {0}".format(os.path.basename(dataset_file)))
else:
    dataset_file = os.path.join(abs_path, "../../data/datasets/{0}.csv".format(vidname))
    print("Output file not given or wrong filepath. Dataset will be saved to: {0}".format(
        os.path.basename(dataset_file)))


# Load config params
try:
    with open(config_file, 'r') as f1:
        config = configparser.ConfigParser()
        config.read_file(f1)
        print("Loading config file: {0}".format(os.path.basename(config_file)))
except IOError:
    print("Config file not found.")
    sys.exit(1)
extract_fps = config.getint("fps", "extract_fps")
row_start = config.getint("crop", "row_start")
row_end = config.getint("crop", "row_end")
col_start = config.getint("crop", "col_start")
col_end = config.getint("crop", "col_end")
width = config.getint("resize", "width")
height = config.getint("resize", "height")
scale_x = config.getfloat("resize", "scale_x")
scale_y = config.getfloat("resize", "scale_y")

# Load video frame timestamps
with open(ts_file, 'r') as f2:
    vid_timestamps = f2.readlines()
vid_timestamps = np.array(vid_timestamps).astype(np.float)

# Load driving can log
can_df = pd.read_csv(can_file)
# print(can_df.info())  # dtype of timestamps is automatically converted from string to np.float64 during read
dataset_dict = {"image": [], "steer-angle": [], "steer-torque1": [], "steer-torque2": [], "steer-torque3": [],
                "speed": [], "acc": [], "brake": [], "ice-rpm": [], "tire1-speed": [], "tire2-speed": [],
                "tire3-speed": [], "tire4-speed": [], "timestamp": []}

# Load bad data frames to skip
bad_data_df = pd.read_excel(bad_data_file, sheet_name="{0}".format(vidname))
bad_frames = []
for row in bad_data_df.itertuples():
    bad_list = list(range(row.start_frame, row.end_frame))
    bad_frames.append(bad_list)
bad_frames = np.array([item for sublist in bad_frames for item in sublist])

# Load video
stream = cv2.VideoCapture(vid_file)
no_of_frames = stream.get(int(7))
vid_fps = stream.get(int(5))
print("Video loaded\nNo. of frames in video: {0}\nVideo fps: {1}".format(no_of_frames, vid_fps))

# Extract every nth frame based on specified fps, and remove bad frames
if extract_fps == 0 or extract_fps > vid_fps:
    extract_fps = vid_fps
extract_frames = np.arange(1, no_of_frames+1, round(vid_fps/extract_fps))
extract_frames = np.array([item for item in extract_frames if item not in bad_frames])

# Start frame extraction, image processing and building the dataset
start = time.time()
while stream.isOpened():
    frame_no = int(stream.get(int(1)))
    grabbed, frame = stream.read()

    if grabbed:
        if frame_no in extract_frames:
            frame = frame[row_start:row_end+1, col_start:col_end+1]
            frame = cv2.resize(frame, (width, height), fx=scale_x, fy=scale_y, interpolation=cv2.INTER_AREA)
            img_path = "{0}/{1}_{2}.jpg".format(frames_dir, vidname, frame_no)
            cv2.imwrite(img_path, frame)
            timestamp = vid_timestamps[frame_no]

            dataset_dict["image"].append(img_path)
            dataset_dict["timestamp"].append(str("{0:.6f}".format(timestamp)))

            idx = can_df["steer_ts"].sub(timestamp).abs().idxmin()
            if not pd.isna(idx):
                dataset_dict["steer-angle"].append(can_df["steer-angle"][idx])
                dataset_dict["steer-torque1"].append(can_df["steer-torque1"][idx])
                dataset_dict["steer-torque2"].append(can_df["steer-torque2"][idx])
                dataset_dict["steer-torque3"].append(can_df["steer-torque3"][idx])

            idx = can_df["speed_ts"].sub(timestamp).abs().idxmin()
            if not pd.isna(idx):
                dataset_dict["speed"].append(can_df["speed"][idx])

            idx = can_df["acc_ts"].sub(timestamp).abs().idxmin()
            if not pd.isna(idx):
                dataset_dict["acc"].append(can_df["acc"][idx])

            idx = can_df["brake_ts"].sub(timestamp).abs().idxmin()
            if not pd.isna(idx):
                dataset_dict["brake"].append(can_df["brake"][idx])

            idx = can_df["ice_ts"].sub(timestamp).abs().idxmin()
            if not pd.isna(idx):
                dataset_dict["ice-rpm"].append(can_df["ice-rpm"][idx])

            idx = can_df["tire_ts"].sub(timestamp).abs().idxmin()
            if not pd.isna(idx):
                dataset_dict["tire1-speed"].append(can_df["tire1-speed"][idx])
                dataset_dict["tire2-speed"].append(can_df["tire2-speed"][idx])
                dataset_dict["tire3-speed"].append(can_df["tire3-speed"][idx])
                dataset_dict["tire4-speed"].append(can_df["tire4-speed"][idx])
        cv2.waitKey(1)
    else:
        break

stream.release()
cv2.destroyAllWindows()

dataset_df = pd.DataFrame.from_dict(dataset_dict, orient='index').T
dataset_df.to_csv(dataset_file, index=False)
print("Successfully written {0} frames to dataset using {1} fps".format(len(dataset_df.index), extract_fps))

stop = time.time()
diff = stop - start
hrs = int(diff/3600)
mins = int((diff % 3600)/60)
secs = int((diff % 3600) % 60)
print("Time elapsed: {0}hrs {1}mins {2}secs".format(hrs, mins, secs))
