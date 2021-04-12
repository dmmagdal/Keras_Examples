# split_save_videoframes.py
# Get all images from a video, frame by frame and save them
# Source: https://www.codegrepper.com/code-examples/python/
# python+split+video+into+frames
# Python 3.7
# Windows/MacOS/Linux


import os
import cv2
import numpy as np


def main():
	# Note: Make sure that all the paths and files exist before running
	# the program.
	video_path = "Person.of.Interest.S01E01.MULTI.1080p.x264-SMiTH.MKV"
	save_path = "POI_S1_E1"
	#new_video_path =

	# Test splitting up a video by frame into images.
	video_to_frames(video_path, save_path)

	# Test constructing a video from frame images.
	#frames_to_video(new_video_path, save_path)

	# Exit the program.
	exit(0)


def video_to_frames(video_file_path, images_save_path):
	# Verify the paths of both the target video file and the save image
	# folder.
	if not os.path.exists(video_file_path):
		print("Error: Could not locate target video file " + video_file_path)
		exit(1)
	if not os.path.exists(images_save_path):
		print("Error: Could not locate save folder " + images_save_path)
		exit(1)
	if not images_save_path.endswith("/"):
		images_save_path += "/"

	# Extract the video name from the target file path.
	video_name = video_file_path.split("/")[-1]
	video_name = video_name.split(".")
	video_name = ".".join(video_name[:-1]) if len(video_name) > 1 else video_name[0]
	'''
	video_folder = video_file_path.split("/")
	video_folder = "/".join(video_folder[:-1])
	'''

	# Load in video file.
	vidcap = cv2.VideoCapture(video_file_path)
	success, image = vidcap.read()
	count = 0

	# Iterate through the loop while reading the next frame in the
	# video.
	while success:
		# Save frame as JPEG file.
		cv2.imwrite(images_save_path + video_name + "_frame_{}.jpg".format(count), image)
		success, image = vidcap.read()
		print("Read a new frame: ", success)
		count += 1

	# Return the function.
	return


def frames_to_video(images_save_path, video_file_path):
	# Verify the paths of both the target video file and the save image
	# folder.
	video_folder = video_file_path.split("/")
	video_folder = "/".join(video_folder[:-1])
	if not os.path.exists(images_save_path):
		print("Error: Could not locate image folder " + images_save_path)
		exit(1)
	if not os.path.exists(video_folder):
		print("Error: Could not locate save video path " + video_file_path)
		exit(1)

	# INSERT CODE HERE.

	# Return the function.
	return

if __name__ == '__main__':
	main()