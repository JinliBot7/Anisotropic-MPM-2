#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:23:05 2023

@author: luyin
"""
import cv2
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip


# Define the paths to the input videos
jpg_files = []
for i in range(1,500,1):
    print(i)
    jpg_files.append(f'{i}.jpg')
    

texts = []
acceleration = []
# Define the text to be added to each video
for i in range(1,500,1):
    print(i)
    texts.append(f'iteration {i}')
    acceleration.append('x 0.5')




def combine_jpg_to_mp4(image_files, text_contents, output_video, fps=30):
    if len(image_files) != len(text_contents):
        raise ValueError("The length of the image_files and text_contents lists must be the same.")

    # Read the first image to get frame size and color mode
    sample_image = cv2.imread(image_files[0])
    height, width, layers = sample_image.shape
    size = (width, height)

    # Set up the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, size)

    # Set up text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_color = (0, 0, 0)
    font_thickness = 3

    # Write each image to the video file
    for image_file, text in zip(image_files, text_contents):
        img = cv2.imread(image_file)
        text_position = (int(width/2)-110, 130)  # Bottom-left corner
        img = cv2.putText(img, text, text_position, font, font_scale, font_color, font_thickness)
        video_writer.write(img)

    video_writer.release()




output_video = 'output_video.mp4'
fps = 30

combine_jpg_to_mp4(jpg_files, texts, output_video, fps)


