#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 21:14:38 2023

@author: luyin
"""

import cv2

def crop_video(input_file, output_file, x, y, crop_width, crop_height):
    # Read input video
    cap = cv2.VideoCapture(input_file)

    # Check if video is opened successfully
    if not cap.isOpened():
        print(f"Error: Unable to open video file '{input_file}'. Please check the file path and format.")
        return

    # Get original video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width,height)

    # Validate cropping dimensions and coordinates
    if x < 0 or y < 0 or x + crop_width > width or y + crop_height > height:
        print("Error: Invalid cropping dimensions or coordinates.")
        return

    # Create the VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS) * 3)
    out = cv2.VideoWriter(output_file, fourcc, fps, (crop_width, crop_height))

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Crop the frame
        cropped_frame = frame[y:y+crop_height, x:x+crop_width]

        # Write the cropped frame to the output video
        out.write(cropped_frame)

    # Release the VideoCapture and VideoWriter objects and close the windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

input_file = '0A_7_with_hanger_half_control_lower_stop.mp4'
output_file = 'cropped_video.mp4'
x, y = 320, 0  # Starting point (x, y) for the cropped region
crop_width, crop_height = 450, 600  # Desired dimensions for the cropped region

crop_video(input_file, output_file, x, y, crop_width, crop_height)