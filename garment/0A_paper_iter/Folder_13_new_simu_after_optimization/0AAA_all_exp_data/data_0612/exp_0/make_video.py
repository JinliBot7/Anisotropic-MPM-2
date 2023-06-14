#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 14:11:53 2023

@author: luyin
"""

import cv2
import os
import numpy as np

def combine_jpg_to_mp4(image_files, output_video, fps=30):


    # Read the first image to get frame size and color mode
    sample_image = cv2.imread(image_files[0])
    height, width, layers = sample_image.shape
    size = (width, height)

    # Set up the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, size)

    # Set up text properties
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # font_scale = 1
    # font_color = (255, 255, 255)
    # font_thickness = 2

    # Write each image to the video file
    for image_file in image_files:
        img = cv2.imread(image_file)
        # text_position = (10, size[1] - 10)  # Bottom-left corner
        # img = cv2.putText(img, text, text_position, font, font_scale, font_color, font_thickness)
        video_writer.write(img)

    video_writer.release()

time_list = np.load('time.npy')
start_frame = 25
end_frame = 0
for i in range(start_frame,120,1):
    if time_list[i] - time_list[start_frame] >=2:
        end_frame = i
        break

image_files = []
count = 0
for i in range(start_frame, end_frame+1, 1):
    image_files.append(f'./rgb/rgb_img_{i}.jpg')
    count += 1



output_video = 'output_video.mp4'
fps = (count + 1) / 2

combine_jpg_to_mp4(image_files, output_video, fps)
























