#!/usr/bin/env python3
# @author: Khushal Brahmbhatt

"""
Script to decode and extract driving signals from CAN logs.

Usage:
python3 can_extract.py -i canlogfile.log
"""

import argparse
import os
import sys

import pandas as pd

# Parse input file
parser = argparse.ArgumentParser(
    description="Specify CAN log file.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input", required=True,
                    help="Input CAN log file to extract driving signals from.")
parser.add_argument("-o", "--output", default=None,
                    help="Specify csv file to write extracted driving signals to.")
args = parser.parse_args()

# Load CAN log file
can_file = args.input
try:
    df = pd.read_csv(can_file, delimiter=" |#", engine='python', header=None)
    print("[INFO] Found CAN log file: {0}".format(can_file))
except IOError:
    print("[ERROR] CAN log file not found. Wrong filename or filepath.")
    sys.exit(1)

# Get output file
if args.output and os.path.isdir(os.path.dirname(args.output)):
    out_file = args.output
    print("[INFO] Extracted params will be saved to the specified output file: {0}".format(out_file))
else:
    out_file = os.path.splitext(can_file)[0]
    print("[INFO] Output file not given or wrong filepath. Extracted params will be saved to: {0}_canlog.csv".format(
        out_file))

can_data = {"steer_angle": [], "steer_torque1": [], "steer_torque2": [], "steer_torque3": [], "steer_ts": [],
            "speed": [], "speed_ts": [], "acc": [], "acc_ts": [], "brake": [], "brake_ts": [], "ice_rpm": [],
            "ice_ts": [], "tire1_speed": [], "tire2_speed": [], "tire3_speed": [], "tire4_speed": [], "tire_ts": []}


def extract_data(row):
    timestamp = row[0][1:-1]
    data = row[3]

    if row[2] == "025":
        st_angle = int(data[:4], 16)  # convert hex to int
        # anticlockwise rotations are +ve, start from 0x0001
        # clockwise rotations are -ve, start from 0x0FFF, so subtract 4096 (0x1000) from angles larger than 2048
        if st_angle > 2048:
            st_angle = st_angle - 4096
        st_q1 = int(data[8:10], 16)
        st_q2 = int(data[10:12], 16)
        st_q3 = int(data[12:14], 16)
        can_data["steer_angle"].append(st_angle)
        can_data["steer_torque1"].append(st_q1)
        can_data["steer_torque2"].append(st_q2)
        can_data["steer_torque3"].append(st_q3)
        can_data["steer_ts"].append(timestamp)

    elif row[2] == "0B4":
        speed = int(data[10:14], 16)/100
        can_data["speed"].append(speed)
        can_data["speed_ts"].append(timestamp)

    elif row[2] == "245":
        acc = int(data[4:6], 16)
        can_data["acc"].append(acc)
        can_data["acc_ts"].append(timestamp)

    elif row[2] == "224":
        brake = int(data[8:12], 16)
        can_data["brake"].append(brake)
        can_data["brake_ts"].append(timestamp)

    elif row[2] == "1C4":
        ice = int(data[0:4], 16)
        can_data["ice_rpm"].append(ice)
        can_data["ice_ts"].append(timestamp)

    elif row[2] == "0AA":
        tire1 = int(data[0:4], 16)
        tire2 = int(data[4:8], 16)
        tire3 = int(data[8:12], 16)
        tire4 = int(data[12:-1], 16)
        can_data["tire1_speed"].append(tire1)
        can_data["tire2_speed"].append(tire2)
        can_data["tire3_speed"].append(tire3)
        can_data["tire4_speed"].append(tire4)
        can_data["tire_ts"].append(timestamp)


df.apply(extract_data, axis=1)
df_can = pd.DataFrame.from_dict(can_data, orient='index').T  # orient on index and transpose since columns have different lengths
df_can.to_csv("{0}_canlog.csv".format(out_file), index=False)
print("[INFO] Finished extracting {0} params.".format(len(df.index)))
