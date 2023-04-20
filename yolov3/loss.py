import einops

import torch
import torch.nn as nn
from utils import iou_score

class YoloLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.bce = nn.BCEWithLogitsLoss()
        self.cross_entrpy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        
        
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
        self.eps = 1e-9
        
    def forward(self, preds, target, anchors):
        obj = (target[..., 0] == 1)
        no_obj = (target[..., 0] == 0)
        
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        
        #Bounding Box Loss
        box_preds = torch.cat((self.sigmoid(preds[..., 1:3]), anchors*torch.exp(preds[..., 3:5])), dim=-1)        
        coord_loss = self.mse(box_preds, target[..., 1:5])

        #Object Loss        
        ious = iou_score(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.mse(self.sigmoid(preds[..., 0:1][obj]), ious*target[..., 0:1][obj])
        
        #No Object Loss
        no_obj_loss = self.bce(preds[..., 0:1][no_obj], target[..., 0:1][no_obj])
        
        #Class Loss
        class_loss = self.cross_entrpy(preds[..., 5:][obj], target[..., 5][obj].long())
        
        #Total Loss
        total_loss = (self.lambda_coord*coord_loss + 
                      object_loss + 
                      class_loss +
                      self.lambda_noobj*no_obj_loss)
        
        return total_loss
        
        