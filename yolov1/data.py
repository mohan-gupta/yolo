import os

from PIL import Image
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

import config

class YOLOData:
    def __init__(self, data, img_dir, label_dir,
                 grid_size=7, num_classes=20, transforms=None):
        self.data = data
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.transforms = transforms
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label_path = os.path.join(self.label_dir, self.data.iloc[idx, 1])
        img_path = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        
        boxes = []
        with open(label_path) as f:
            for labels in f.readlines():
                label_lst = labels.replace('\n', "").split()
                for idx in range(len(label_lst)):
                    label_lst[idx] = float(label_lst[idx])
                
                boxes.append(label_lst)
        
        img = Image.open(img_path)
        img = np.array(img)
        boxes = torch.tensor(boxes)
        
        if self.transforms:
            augmented = self.transforms(image = img, boxes = boxes)
            img, boxes = augmented['image'], augmented['boxes']
        
        label_matrix = torch.zeros(self.grid_size, self.grid_size, self.num_classes+5)
        
        for box in boxes:
            class_label, x, y, w, h = box.tolist()
            class_label = int(class_label)
            
            i, j = int(self.grid_size*y), int(self.grid_size*x)
            x_cell, y_cell = self.grid_size*x - j, self.grid_size*y - i
            
            w_cell, h_cell = self.grid_size*w, self.grid_size*h
            
            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                box_coords = torch.tensor([x_cell, y_cell, w_cell, h_cell])
                
                label_matrix[i, j, 21:25] = box_coords
                
                label_matrix[i, j, class_label] = 1
        
        return {
            'image': img.to(dtype=torch.float32),
            'label': label_matrix.to(dtype=torch.float32)
        }

def get_loaders(data_path, img_path, label_path, grid_size=7, num_classes=20, transforms=None):
    df = pd.read_csv(data_path)
    
    train_df, val_df = train_test_split(df, test_size=0.25, random_state=42)
    
    train_data = YOLOData(train_df, img_path, label_path, grid_size, num_classes, transforms)
    val_data = YOLOData(val_df, img_path, label_path, grid_size, num_classes, transforms)
    
    train_loader = DataLoader(train_data, batch_size=config.BS, shuffle=True,
                              pin_memory=True)
    
    val_loader = DataLoader(val_data, batch_size=config.BS, shuffle=False, pin_memory=True)
    
    return train_loader, val_loader