import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

class ImagePreprocessor:
    def __init__(self, target_size=(640, 640)):
        self.target_size = target_size

    def resize_image(self, image):
        return cv2.resize(image, self.target_size)

    def augment_image(self, image):
        # Light augmentation (e.g., flip, brightness)
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)  # Horizontal flip
        brightness_change = np.random.uniform(0.8, 1.2)
        image = cv2.convertScaleAbs(image, alpha=brightness_change)
        return image

class DatasetSplitter:
    def __init__(self, data_dir, split_ratio=(0.7, 0.2, 0.1)):
        self.data_dir = data_dir
        self.split_ratio = split_ratio

    def split_data(self):
        all_files = os.listdir(self.data_dir)
        train_files, temp_files = train_test_split(all_files, test_size=self.split_ratio[1]+self.split_ratio[2])
        val_files, test_files = train_test_split(temp_files, test_size=self.split_ratio[2]/(self.split_ratio[1] + self.split_ratio[2]))
        return train_files, val_files, test_files

class DatasetOrganizer:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def organize_data(self, train_files, val_files, test_files, train_dir='train/', val_dir='val/', test_dir='test/'):
        for directory in [train_dir, val_dir, test_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        for file_name in train_files:
            os.rename(os.path.join(self.data_dir, file_name), os.path.join(train_dir, file_name))
        for file_name in val_files:
            os.rename(os.path.join(self.data_dir, file_name), os.path.join(val_dir, file_name))
        for file_name in test_files:
            os.rename(os.path.join(self.data_dir, file_name), os.path.join(test_dir, file_name))
