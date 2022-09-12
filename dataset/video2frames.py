# video to frames and resize to 256
# input: video files, *.mp4
# output: frame files, *.jpg

# By Pu Jin 2020.11.30

#! encoding: UTF-8

import os
import cv2


videos_src_path = 'D:\\Files\\MUC\\Anomaly detection\\dataset\\NEW\\test\\09\\'
videos_save_path = 'D:\\Files\\MUC\\Anomaly detection\\dataset\\dataset3\\test\\'

videos = os.listdir(videos_src_path)
videos = filter(lambda x: x.endswith('mp4'), videos)

for each_video in videos:

    # get the name of each video, and make the directory to save frames
    each_video_name, _ = each_video.split('.')
    os.mkdir(videos_save_path + each_video_name)

    each_video_save_full_path = os.path.join(videos_save_path, each_video_name) + '\\'

    # get the full path of each video, which will open the video tp extract frames
    each_video_full_path = os.path.join(videos_src_path, each_video)

    cap = cv2.VideoCapture(each_video_full_path)
    frame_count = 0
    success = True
    while success:
        success, frame = cap.read()
        # params = []
        # params.append(cv.CV_IMWRITE_PXM_BINARY)
        # params.append(1)
        if success:
            frame_resized = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
            cv2.imwrite(each_video_save_full_path + "%d.jpg" % frame_count, frame_resized)
        frame_count = frame_count + 1

    cap.release()
