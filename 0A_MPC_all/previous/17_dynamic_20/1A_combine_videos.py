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
video_files = []
for i in range(0,151,10):
    print(i)
    video_files.append(f'./vis/{i}/video.mp4')
    

texts = []
# Define the text to be added to each video
for i in range(0,151,10):
    print(i)
    texts.append(f'iteration {i}')


# Function to add text to frame
def add_text_to_frame(frame, text, position=(500, 50)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    color = (0, 0, 0)
    thickness = 2
    return cv2.putText(frame, text, position, font, scale, color, thickness, cv2.LINE_AA)

# Combine videos, add text, and pause the last frame for 1 second
output_file = 'combined_video_with_text_and_pause.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = None
for i, video_file in enumerate(video_files):
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if out is None:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for j in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        frame_with_text = add_text_to_frame(frame, texts[i])

        if j == frame_count - 1:
            # Add a 1-second pause using the last frame
            for _ in range(fps * 2):
                out.write(frame_with_text)
        else:
            out.write(frame_with_text)

    cap.release()

if out is not None:
    out.release()


