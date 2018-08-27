import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms
from avito_transforms import RandomCrop, RandomRotate, Compose

class PascalDataset(Dataset):
    def __init__(self, dataset_dir, data):
        self.table = data
        self.dataset_dir = dataset_dir
        
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            #transforms.ColorJitter(brightness= 0.5, contrast= 0.5, saturation= 0.5, hue= 0.5),
            transforms.ToTensor()
        ])

        #self.augments = Compose([RandomRotate(), RandomCrop()])
        self.augments = Compose([RandomRotate()])
    
    def __len__(self):
        return self.table.shape[0]

    @classmethod
    def from_file(cls, dataset_dir, filename):
        print("creating dataset: ", os.path.join(dataset_dir, filename))
        table = pd.read_csv(os.path.join(dataset_dir, filename))
        return cls(dataset_dir, table)

    def __getitem__(self, idx):
        path = os.path.join(self.dataset_dir, "images", self.table.iloc[idx].image_name)
        image = cv2.imread(path)
        #image = Image.open(path)

        if image.shape[0] == 0 or image.shape[1] == 0:
            print(image)
        assert image.shape[0] != 0 and image.shape[1] != 0, path
        target = self.table.iloc[idx]
        target = np.array([target.x1, target.y1, target.x2, target.y2])
        target = np.clip(target, 0, 1)

        try:
            image, target = self.augments(image, target)
        except ValueError:
            print("uups: target:", target)

        assert image.shape[0] != 0 and image.shape[1] != 0
        image = Image.fromarray(np.uint8(image))
        image = self.transform(image)

        target = torch.from_numpy(target).float()*1000

        return image, target