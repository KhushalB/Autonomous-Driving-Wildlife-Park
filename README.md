# End-to-End Learning for Autonomous Driving in Wildlife Parks
This project aims to use an end-to-end learning technique using convolutional neural networks for steering prediction on
unpaved driving trails in wildlife parks. It is part of a broader study under the Artificial Intelligence for 
Development (AI4D) initiative in sub-Saharan Africa by the International Development Research Centre (IDRC). The study
aims at investigating the technological feasibility of deploying Unmanned Ground Vehicles for automated wildlife patrol
in national parks in order to meet shortages in ranger workforce and increase patrol efficiency.

## Data
### Data collection
As part of this study, we collected about 20 hours of driving data from both unpaved driving trails in nationals park in
Kenya, and paved roads across western Kenya:
* 8.5 hrs (115km) from Nairobi National Park
* 2.5 hrs (30km) from Ruma National Park
* 9 hrs (425km) on highways in western Kenya

The driving data included video of the driving trails recorded using a camera, as well as driving signals obtained
from the vehicle's ECU:
* 720p video at 30fps, encoded using XVID codec and .avi container
* Steering wheel angle
* Steering wheel torque
* Vehicle speed
* Accelerator pedal position
* Brake pedal position
* Individual tire speeds
* GPS coordinates

All the data collected above is timestamped, including individual video frames. The data was also collected at different
times of day under different lighting conditions. Vehicle speed is given in km/h. However, the units for the rest of the
driving parameters are unknown as they couldn't be decoded, thus they are given as recorded after converting from
hexadecimal to decimal.

The vehicle used for the data collection was a Toyota Prius 2012. The driving signals were logged on a Raspberry Pi
from the vehicle's CAN bus which is usually exposed on the OBD-II port that is present on all vehicles manufactured
after 2008. The driving video was recorded on a laptop using the Apeman A80 action camera. The GPS coordinates
were recorded on a smartphone using the *GPS Logger* app.

![Data collection setup](images/data-collection-setup.png?raw=true)

*Fig: Data collection setup*

More details on the data collection and challenges faced can be found on the [AI4D blog](https://ai4d.ai/autonomous-driving/).

### Dataset preparation
Frames were extracted from the video at 10fps. They were cropped to remove pixels above the horizon and resized to
200x66 for training. Segments of the video containing irrelevant data such as U-turns, intersections, vehicle stopped 
etc. were removed before extracting the frames. Segments containing overtaking and driving around potholes or bad road 
conditions were also removed as they presented navigation challenges beyond the scope of this project. A large portion 
of the data contains driving on straight road sections rather than around turns. To remove this skewness in data, long 
straight road sections were also removed from the video, and data augmentation was used to obtain more images of driving
around turns. The steering angles are also scaled by a fixed value of 65 (which is approximately 90 degrees) to ensure
they mostly lie between -1 and 1.

Data augmentation techniques applied include horizontal flipping and random brightness change.

The driving signals were decoded and extracted from the raw CAN logs, and the timestamp of each extracted video frame 
was matched to the closest timestamp of each driving parameter to create each data sample.

![sample image 1](images/sampleimg1.png?raw=true)

![sample image 2](images/sampleimg2.png?raw=true)

![sample image 3](images/sampleimg3.png?raw=true)

*Fig: Sample images from the dataset*

## Training
The data was trained using the Nvidia Dave2 model shown below:

![nvidia dave2 model](images/dave2.png)

In addition, dropout layers were added after each fully connected layer in the above network. ELU was used as the 
activation function between the layers, and tanh at the output neuron to keep the steering angle predictions between
-1 and 1. Adam optimizer with a learning rate of 1e-4 was used, and an MSE loss function.

## Usage
### Dependecies
* tensorflow-gpu=2.2.0
* keras=2.3.1
* opencv-contrib-python=4.2.0
* numpy=1.18.4
* pandas=1.0.1
* matplotlib=3.1.3
* seaborn=0.10.0

### Project structure
To allow the data collection/preparation scripts to identify and load some data files automatically, some naming 
conventions have been used:
* Raw videos are saved in `data/videos` as `example-drive.avi`
* Timestamps for each frame in the video are saved in `data/video-timestamps` as `example-drive_ts.txt`
* Extracted frames are saved in `data/frames/example-drive` as `example-drive_frameno.jpg`
* The frames that are not to be extracted to the dataset are saved in `data/bad_data.xlsx` and the sheet is named 
`example-drive`
* CAN logs are saved in `data/can-logs` as `example-drive.log`
* Driving parameters extracted from the raw CAN log using `can_extract.py` are saved in `data/can-logs` as 
`example-drive_canlog.csv`
* The dataset is saved in `data/datasets` as `example-drive.csv`

All the scripts used in the data collection and dataset preparation can be found in `src/data_collection`:
* `video_capture.py` was used to record and save the video with OpenCV. The video properties can be set in 
`config/vidcap_config.ini`.
* `frame_rate.py` is used to check properties of the recorded video such as video length, frame rate and number of frames.
* `can_extract.py` is used to extract the driving parameters from the raw CAN frames in the CAN log to a csv file.
* `build_dataset.py` is used to extract and process the required frames from the video and match their timestamps to the
driving parameters to create the dataset. Frame extraction parameters can be set in `config/dataset_config.ini`.

The training code is located in `src/training`:
* `explore_data.ipynb` is a Jupyter Notebook that can be used to explore the data.
* `model_def.py` is used to create the model class to be used for training.
* `augment_data.py` contains utility functions for loading, preprocessing and augmenting the images.
* `train.py` is used to train and save the model in the `src/training/models/` directory. The training parameters can be
set in `config/train_config.ini`.
* `predict.py` is used to make predictions on the test set and visualize them.

## Hardware used
For data collection:
* Raspberry Pi 3B+
* PiCAN2 hat
* Apeman A80 camera
* Laptop

For training:
* Ubuntu 18.04, Intel Core i5-8300H CPU @ 2.3GHz x 8, 8GB System RAM
* Nvidia GTX1050 GPU with 4GB Graphics RAM

## Acknowledgements
This project was made possible through a research grant from the [Knowledge 4 All Foundation](https://www.k4all.org/).
It was conducted as part of the [AI4D initiative](https://ai4d.ai/) by [IDRC, Canada](https://www.idrc.ca/).

## Links
* Dataset: To be published
* AI4D blog: https://ai4d.ai/autonomous-driving/
* AfricaNLP workshop presentation at ICLR 2020: https://africanlp-workshop.github.io/ai4dev.html
