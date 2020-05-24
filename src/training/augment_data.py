# @author: Khushal Brahmbhatt

from random import sample

import cv2
import numpy as np


class AugData:
    @staticmethod
    def load_img(img_path):
        """
        Function to load image from image path.
        :param img_path: Path to image, type:string
        :return: image as numpy array
        """
        img = cv2.imread(img_path)
        return img

    @staticmethod
    def flip_hor(img_list, labels, flip_prob):
        """
        Function to flip horizontally a percentage of images from the given list.
        :param img_list: List of images as numpy arrays, type:list
        :param labels: List of steering angle values, type:list
        :param flip_prob: Percentage of samples to flip, type:float
        :return: Lists containing flipped images as numpy arrays, and adjusted steering angles
        """
        # Perform augmentation on x% of training samples
        rand_ind = sample(range(0, len(img_list)), round(flip_prob * len(img_list)))
        img_list = [img_list[i] for i in rand_ind]
        labels = [labels[i] for i in rand_ind]

        x_aug = []
        y_aug = []
        for i, img in enumerate(img_list):
            img = cv2.flip(img, 1)
            angle = -labels[i]
            x_aug.append(img)
            y_aug.append(angle)

        return x_aug, y_aug

    @staticmethod
    def rand_bright(img, br=0.25):
        """
        Function to randomly change brightness of the image.
        :param img: Image as numpy array, type:numpy.ndarray
        :param br: Brightness shift value, type:float
        :return: Image with adjusted brightness as numpy array
        """
        br_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        br_factor = br + np.random.uniform()
        br_img[:, :, 2] = br_img[:, :, 2] * br_factor
        br_img = cv2.cvtColor(br_img, cv2.COLOR_HSV2RGB)
        return br_img

    @staticmethod
    def preprocess_img(img, width, height):
        """
        Function to convert image colour scale and resize.
        :param img: Image as numpy array, type:numpy.ndarray
        :param width: width to resize to from train_config.ini, type:int
        :param height: height to resize to from train_config.ini, type:int
        :return: Image as numpy array
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if width != 0 or height != 0:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

        return img
