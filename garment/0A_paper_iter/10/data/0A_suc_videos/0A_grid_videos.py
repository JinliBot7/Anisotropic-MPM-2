#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 17:38:44 2023

@author: luyin
"""

import glob
import re
from moviepy.editor import VideoFileClip, clips_array

def sort_key(filename):
    # Extract the 'a' value from the filename using a regular expression
    match = re.search(r'trial_(\d+)_\d+.mp4', filename)
    if match:
        return int(match.group(1))
    else:
        return 0  # Default sort value for files that don't match the pattern

def make_grid(rows, columns, video_files):
    video_clips = [VideoFileClip(f) for f in video_files]
    grid_clips = []

    # Organize the clips into grid
    for r in range(rows):
        row_clips = []
        for c in range(columns):
            clip_idx = r * columns + c
            if clip_idx < len(video_clips):
                row_clips.append(video_clips[clip_idx])
            else:
                row_clips.append(None)  # If there are not enough clips, add None
        grid_clips.append(row_clips)
    
    # Create video grid
    final_clip = clips_array(grid_clips)

    return final_clip

# Use glob to get a list of video files
video_files = glob.glob('*.mp4')

# Sort video files based on 'a' value
video_files.sort(key=sort_key)

final_clip = make_grid(5, 6, video_files)
final_clip.write_videofile('output.mp4', codec='libx264')
