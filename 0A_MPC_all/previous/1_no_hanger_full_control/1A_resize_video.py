#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 20:57:07 2023

@author: luyin
"""

import cv2

def shrink_video(input_file, output_file, scale_factor):
    # Read input video
    #print('what')
    cap = cv2.VideoCapture(input_file)
    # Get original video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate the desired dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Create the VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_file, fourcc, fps, (new_width, new_height))
    print(cap.isOpened())
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Write the resized frame to the output video
        out.write(resized_frame)

    # Release the VideoCapture and VideoWriter objects and close the windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

input_file = 'cropped_video.mp4'
output_file = 'shrunken_video.mp4'
scale_factor = 0.8

shrink_video(input_file, output_file, scale_factor)
