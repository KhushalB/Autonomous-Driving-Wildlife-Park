#!/usr/bin/env python3
# @author: Khushal Brahmbhatt

"""
Script to estimate actual frame rate of the captured video and other frame info.

Usage:
python3 frame_extract.py -i timestamps-file.txt
"""

import argparse
import numpy as np
import sys

# Parse input file
parser = argparse.ArgumentParser(
    description="Specify video timestamps file.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input", required=True,
                    help="Input timestamps file corresponding to the video.")
args = parser.parse_args()

ts_file = args.input

try:
    f = open(ts_file, 'r')
except IOError:
    print("Input file not found. Wrong filename or filepath.")
    sys.exit(1)

try:
    lines = f.readlines()
except UnicodeDecodeError:
    print("Invalid filetype. Input file should be a txt file or similar with 'utf-8' encoding.")
    sys.exit(1)
no_of_lines = len(lines)
diff = []

for x in range(no_of_lines - 1):
    curr_frame = float(lines[x])
    next_frame = float(lines[x+1])
    diff.append(next_frame - curr_frame)

ave_delay = np.mean(diff)  # mean time difference between frames
total_diff = float(lines[-1]) - float(lines[0])
hrs = int(total_diff/3600)
mins = int((total_diff % 3600)/60)
secs = int((total_diff % 3600) % 60)

print("Video captured for: {0}hrs {1}mins {2}secs".format(hrs, mins, secs))
print("No. of frames: {0}".format(no_of_lines))
print("Average delay between frames: {0}secs".format(ave_delay))
print("Frame rate (calculated as no. of frames/total capture time): {0}".format(no_of_lines/total_diff))
print("Frame rate (calculated as 1/average delay between frames): {0}".format(1/ave_delay))
