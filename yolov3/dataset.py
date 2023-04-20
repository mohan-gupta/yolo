import os

import numpy as np
import pandas as pd

from PIL import Image

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

from utils import iou_wh_score

import config

class YOLOData:
    def __init__(self, data, img_dir, label_dir, anchors,
                scales=[13, 26, 52], num_classes=80, transforms=None):
        self.data = data
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.anchors = torch.tensor(anchors)
        self.scales = scales
        self.num_classes = num_classes
        self.transforms = transforms
        
        self.num_anchors = self.anchors.shape[0]
        self.num_scales = len(self.scales)
        
        self.ignore_iou_thresh = 0.5
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        #Load the image and bounding boxes
        label_path = os.path.join(self.label_dir, self.data.iloc[idx, 1])
        bboxes = np.loadtxt(label_path, delimiter=' ', ndmin=2)
        bboxes = np.roll(bboxes, shift=-1, axis=1).tolist()
        
        img_path = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = np.array(Image.open(img_path).convert("RGB"))
        
        #Apply Transformation
        if self.transforms:
            augmented = self.transforms(image=image, bboxes=bboxes)
            image = augmented["image"]
            bboxes = augmented['bboxes']
            
        targets = [torch.zeros(self.num_scales, S, S, 6) for S in self.scales]
        
        #for each box
        for box in bboxes:            
            x, y, w, h, c = box
            
            #For each Scale
            for scale_idx, anchor in enumerate(self.anchors):
                scale = self.scales[scale_idx]
                #find out IoU of anchor and box
                iou_anchors = iou_wh_score(anchor, torch.tensor(box))
                
                #get max IoU anchors index
                max_iou_scale_idx = torch.argmax(iou_anchors)
                
                #Assign this anchor box and grid cell, the coordinates w.r.t to grid cell
                i, j = int(scale*y), int(scale*x)
                
                targets[scale_idx][max_iou_scale_idx, i, j, 0] = 1
                
                x_cell, y_cell = scale*x - j, scale*y - i
                
                h_cell, w_cell = h*scale, w*scale
                
                coords = torch.tensor([x_cell, y_cell, w_cell, h_cell])
                
                targets[scale_idx][max_iou_scale_idx, i, j, 1:5] = coords
                targets[scale_idx][max_iou_scale_idx, i, j, 5] = c
            
            return {
                'image': image,
                'targets': tuple(targets)
            }

def get_loaders(data_path, img_dir, label_dir, anchors, transforms=None):
    df = pd.read_csv(data_path)
    
    train_df, val_df = train_test_split(df, test_size=0.25, random_state=42)
    
    train_data = YOLOData(train_df, img_dir, label_dir, transforms=transforms, anchors=anchors)
    val_data = YOLOData(val_df, img_dir, label_dir, transforms=transforms, anchors=anchors)
    
    train_loader = DataLoader(train_data, batch_size=config.BS, shuffle=True, pin_memory=True)
    
    val_loader = DataLoader(val_data, batch_size=config.BS, shuffle=False, pin_memory=True)
    
    return train_loader, val_loader