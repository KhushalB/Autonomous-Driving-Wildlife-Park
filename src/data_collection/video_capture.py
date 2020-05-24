#!/usr/bin/env python3
# @author: Khushal Brahmbhatt

"""
Script to capture and write video to a file.
Timestamps of each frame are also written to a file for frame extraction later.

Usage:
python3 video_capture.py

To view the videostream without writing it to a file:
python3 video_capture.py --view
"""

import argparse
import configparser
import os
import select
import sys
import time
from datetime import datetime, timezone

import cv2

# Parse videocapture options
parser = argparse.ArgumentParser(
    description="Specify videocapture options.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--view", default=False, action='store_true',
                    help="Watch the video stream without writing it to file. Press q to quit.")
parser.add_argument("-o", "--output", default=None,
                    help="Specify video filename without the file extension.")
parser.add_argument("-t", "--tsfile", default=None,
                    help="Input timestamps file corresponding to the video.")
args = parser.parse_args()

# Parse output files
if args.output and os.path.isdir(os.path.dirname(args.output)):
    vid_file = args.output
    print("[INFO] Video will be saved to the specified output file: {0}".format(vid_file))
else:
    vid_file = None
if args.tsfile and os.path.isdir(os.path.dirname(args.tsfile)):
    ts_file = args.tsfile
    print("[INFO] Frame timestamps will be saved to the specified output file: {0}".format(ts_file))
else:
    ts_file = None

# Load config file
abs_path = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(abs_path, "../../config/vidcap_config.ini")
try:
    with open(config_file) as f1:
        print("[INFO] Loading config file: {0}".format(config_file))
        config = configparser.ConfigParser()
        config.read_file(f1)
except IOError:
    print("[ERROR] Config file not found.")
    sys.exit(1)

# Load and initialize video properties
try:
    source = config.getint("video_properties", "video_source")
except ValueError:
    source = config.get("video_properties", "video_source")
fps = config.getint("video_properties", "fps")
width = config.getint("video_properties", "width")
height = config.getint("video_properties", "height")

stream = cv2.VideoCapture(source)
stream.set(3, width)
stream.set(4, height)
stream.set(5, fps)


def stream_video():
    """
    Play the video stream without writing it to file.
    """
    print("[INFO] Streaming video from /dev/video{0} in {1}x{2} at {3}fps".format(
        source, int(stream.get(3)), int(stream.get(4)), int(stream.get(5))))

    while stream.isOpened():
        grabbed, frame = stream.read()
        if grabbed:
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    stream.release()
    cv2.destroyAllWindows()


def write_video():
    """
    Write the video stream to a file.
    """
    codec = config.get("video_properties", "codec")
    ext = config.get("video_properties", "extension")
    curr_ts = datetime.timestamp(datetime.now(timezone.utc))

    if vid_file:
        vid_out = "{0}{1}".format(vid_file, ext)
    else:
        vid_out = os.path.join(abs_path, "../../data/videos/drive{0}{1}".format(str(int(curr_ts*1000000)), ext))

    if ts_file:
        ts_out = ts_file
    else:
        ts_out = os.path.join(abs_path, "../../data/video-timestamps/drive{0}_ts.txt".format(str(int(curr_ts*1000000))))

    fourcc = cv2.VideoWriter_fourcc(*'{0}'.format(codec))
    writer = cv2.VideoWriter(vid_out, fourcc, fps, (width, height))
    f2 = open(ts_out, 'w+')
    print("[INFO] Capturing video from /dev/video{0} in {1}x{2} at {3}fps".format(
        source, int(stream.get(3)), int(stream.get(4)), int(stream.get(5))))
    print("Press <Enter> to exit.")

    start = time.time()
    while stream.isOpened():
        grabbed, frame = stream.read()
        if grabbed:
            f2.write("{0}\n".format(datetime.timestamp(datetime.now(timezone.utc))))
            writer.write(frame)
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                input()
                break
        else:
            break

    stream.release()
    cv2.destroyAllWindows()
    f2.close()

    stop = time.time()
    diff = stop - start
    hrs = int(diff/3600)
    mins = int((diff % 3600)/60)
    secs = int((diff % 3600) % 60)
    print("[INFO] Captured video for {0}hrs {1}mins {2}secs".format(hrs, mins, secs))


if args.view:
    stream_video()
else:
    write_video()
