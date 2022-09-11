import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import pdb
import torch.utils.data as data


rng = np.random.RandomState(2020)


def np_load_frame(filename, resize_height, resize_width):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized


class DataLoader(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width, time_step=4, num_pred=1):
        self.dir = video_folder
        self.transform = transform
        self.video_frames = []
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.index_samples = []
        self.setup()

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        if os.path.isdir(videos[0]):
            all_video_frames = []
            videos.sort(key=lambda x: int(x[-2:]))
            for video in videos:
                vide_frames = glob.glob(os.path.join(video, '*.jpg'))
                vide_frames.sort(key=lambda x: int(x[len(video)+1:-4]))
                if len(all_video_frames) == 0:
                    all_video_frames = vide_frames
                else:
                    all_video_frames += vide_frames
        else:
            all_video_frames = videos
        self.video_frames = all_video_frames
        self.index_samples = list(range(len(all_video_frames)-self._time_step))

    def __getitem__(self, index):
        frame_index = self.index_samples[index]
        batch_frames = np.zeros((self._time_step+self._num_pred, 3, self._resize_width, self._resize_height))
        for i in range(self._time_step + self._num_pred):
            image = np_load_frame(self.video_frames[frame_index + i], self._resize_height,
                                  self._resize_width)
            if self.transform is not None:
                batch_frames[i] = self.transform(image)
        return batch_frames

    def __len__(self):
        return len(self.index_samples)
