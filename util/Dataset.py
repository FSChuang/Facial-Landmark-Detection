from xml.etree import ElementTree
from skimage import io
from torch.utils.data import Dataset
import numpy as np
import os

class LandmarksDataset(Dataset):
    def __init__(self, preprocesor, train):
        self.root_dir = './ibug_300W_large_face_landmark_dataset/ibug_300W_large_face_landmark_dataset'

        self.image_paths = []
        self.landmarks = []
        self.crops = []
        self.preprocessor = preprocesor
        self.train = train

        tree = ElementTree.parse(os.path.join(self.root_dir, f'labels_ibug_300W_{"train" if train else "test"}.xml').replace("\\", "/"))
        root = tree.getroot()

        for filename in root[2]:
            self.image_paths.append(os.path.join(self.root_dir, filename.attrib['file']))

            self.crops.append(filename[0].attrib)

            landmark = []
            for num in range(68):
                x_coordinate = int(filename[0][num].attrib['x'])
                y_coordinate = int(filename[0][num].attrib['y'])
                landmark.append([x_coordinate, y_coordinate])
            self.landmarks.append(landmark)

        self.landmarks = np.array(self.landmarks).astype('float32')

        assert len(self.image_paths) == len(self.landmarks)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = io.imread(self.image_paths[index], as_gray = False)
        landmarks = self.landmarks[index]

        image, landmarks = self.preprocessor(image, landmarks, self.crops[index])

        return image, landmarks