import torch
from torch.utils.data import random_split
from util.DataPreprocessing import Preprocessor
from util.Dataset import LandmarksDataset
from model.model import XceptionNet
import validate

import sys
import os

#
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

preprocessor = Preprocessor(
    image_dim = 128,
    brightness = 0.24,
    saturation = 0.3,
    contrast = 0.15,
    hue = 0.14,
    angle = 14,
    face_offset = 32,
    crop_offset = 16)

train_images = LandmarksDataset(preprocessor, train = True)
test_images = LandmarksDataset(preprocessor, train = False)

len_val_set = int(0.1 * len(train_images))
len_train_set = len(train_images) - len_val_set
train_images, val_images = random_split(train_images, [len_train_set, len_val_set])

batch_size = 32
train_data = torch.utils.data.DataLoader(train_images, batch_size = batch_size, shuffle = True)
val_data = torch.utils.data.DataLoader(val_images, batch_size = 2 * batch_size, shuffle = False)

model = XceptionNet()

validate.validate(model, val_data)