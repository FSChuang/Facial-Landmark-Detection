import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from util.DataPreprocessing import Preprocessor
from util.Visualize import preprocessor, visualize_image, visualize_batch
from util.Dataset import LandmarksDataset
from model.model import XceptionNet
from validate import *


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
test_data = torch.utils.data.DataLoader(test_images, batch_size = 2 * batch_size, shuffle = False)

#import model
model = XceptionNet()
model = model.cuda()

# initializing the objective loss & optimizer
objective = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0008)

#create directory for storing the progress(gif form)
if os.path.isdir('progress'):
    os.remove('progress')
os.mkdir('progress')

#Start training the model
epochs = 30
batches = len(train_data)
best_loss = np.inf
optimizer.zero_grad()

for epoch in range(epochs):
    cum_loss = 0.0

    model.train()
    for batch_index, (features, labels) in enumerate(tqdm(train_data, desc = f'Epoch({epoch + 1}/{epochs})', ncols = 800, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')):
        features = features.cuda()
        labels = labels.cuda()

        outputs = model(features)

        loss = objective(outputs, labels)

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        cum_loss += loss.item()
    
    val_loss = validate(model, val_data, os.path.join('progress', f'epoch({str(epoch + 1).zfill(len(str(epochs)))}).jpg'))

    if val_loss < best_loss:
        best_loss = val_loss
        print('Saving Model...........')
        torch.save(model.state_dict(), 'model.pt')
    
    print(f'Epoch({epoch + 1}/{epochs}) -> Training Loss: {cum_loss/batches: .8f} | Validation Loss: {val_loss: .8f}')

    
                                                     
