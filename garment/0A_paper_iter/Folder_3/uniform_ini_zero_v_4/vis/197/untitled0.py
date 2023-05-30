#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 17:44:19 2023

@author: luyin
"""

import cv2
import glob
import os

def combine_jpg_to_mp4(image_folder, output_video, fps=30, text=None):
    images = glob.glob(os.path.join(image_folder, '*.png'))
    images.sort()

    # Read the first image to get frame size and color mode
    sample_image = cv2.imread(images[0])
    height, width, layers = sample_image.shape
    size = (width, height)

    # Set up the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, size)

    # Set up text properties if needed
    if text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)
        font_thickness = 2

    # Write each image to the video file
    for image_file in images:
        img = cv2.imread(image_file)
        if text:
            text_position = (10, size[1] - 10)  # Bottom-left corner
            img = cv2.putText(img, text, text_position, font, font_scale, font_color, font_thickness)
        video_writer.write(img)

    video_writer.release()

name = 'video_4_folder_3'
image_folder = f'frames'
output_video = f'{name}.mp4'
fps = 50

combine_jpg_to_mp4(image_folder, output_video, fps)