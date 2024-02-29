from Visualize import visualize_image, visualize_batch, preprocessor
from Dataset import LandmarksDataset
from torch.utils.data import random_split
import torch


train_images = LandmarksDataset(preprocessor, train = True)
test_images = LandmarksDataset(preprocessor, train = False)


image1, landmarks1 = train_images[64]
visualize_image(image1, landmarks1)

image2, landmarks2 = train_images[64]
visualize_image(image2, landmarks2)

image3, landmarks3 = train_images[64]
visualize_image(image3, landmarks3)

len_val_set = int(0.1 * len(train_images))
len_train_set = len(train_images) - len_val_set

'''
print(f'{len_train_set} images for training')
print(f'{len_val_set} images for validating')
print(f'{len(test_images)} images for testing')
'''

train_images, val_images = random_split(train_images, [len_train_set, len_val_set])

batch_size = 32
train_data = torch.utils.data.DataLoader(train_images, batch_size = batch_size, shuffle = True)
val_data = torch.utils.data.DataLoader(val_images, batch_size = 2 * batch_size, shuffle = False)
test_data = torch.utils.data.DataLoader(test_images, batch_size = 2 * batch_size, shuffle = False)

for x, y in train_data:
    break

#print(x.shape, y.shape, x.max(), x.min(), y.max(), y.min())

for x, y in val_data:
    break

#print(x.shape, y.shape, x.max(), x.min(), y.max(), y.min())

for x, y in test_data:
    break

#print(x.shape, y.shape, x.max(), x.min(), y.max(), y.min())


visualize_batch(x[:16], y[:16], shape = (4, 4), size = 16, title = 'Training Batch Samples')