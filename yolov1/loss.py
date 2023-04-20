import torch
import torch.nn as nn
from utils import iou_score

class YoloLoss(nn.Module):
    def __init__(self, grid_size=7, num_boxes = 2, num_classes=20) -> None:
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
        self.eps = 1e-9
        
    def forward(self, pred_sqz, target):
        pred = pred_sqz.reshape(-1, self.grid_size, self.grid_size, self.num_classes + (self.num_boxes*5))
        
        iou_b1 = iou_score(pred[..., 21:25], target[..., 21:25])
        iou_b2 = iou_score(pred[..., 26:30], target[..., 21:25])
        
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        _, best_box = torch.max(ious, dim=0)
        
        exists_box = target[..., 20].unsqueeze(3)
        
        #Bounding Box Loss
        coord_pred = exists_box*(
            (best_box*pred[..., 26:30] + (1-best_box)*pred[..., 21:25])
            )
        
        coord_target = exists_box*target[..., 21:25]
        
        center_pred = coord_pred[..., 0:2]
        center_target = coord_target[..., 0:2]
        
        center_loss = self.mse(torch.flatten(center_pred),
                              torch.flatten(center_target))
        
        wh_pred = torch.sign(coord_pred[..., 2:4]) * torch.sqrt(torch.abs(coord_pred[..., 2:4]) + self.eps)
        wh_target = torch.sign(coord_target[..., 2:4]) * torch.sqrt(torch.abs(coord_target[..., 2:4]) + self.eps)
        
        wh_loss = self.mse(torch.flatten(wh_pred),
                              torch.flatten(wh_target))
        
        coord_loss = center_loss + wh_loss
        
        #Object Loss
        object_pred = best_box*pred[..., 25:26] + (1-best_box)*pred[..., 20:21]
        
        object_loss = self.mse(torch.flatten(exists_box * object_pred),
                               torch.flatten(exists_box * target[..., 20:21]))
        
        #No Object Loss
        no_obj1_loss = self.mse(torch.flatten((1-exists_box) * pred[..., 20:21]),
                                torch.flatten((1-exists_box) * target[..., 20:21]))
        
        no_obj2_loss = self.mse(torch.flatten((1-exists_box) * pred[..., 25:26]),
                                torch.flatten((1-exists_box) * target[..., 20:21]))
        
        no_obj_loss = no_obj1_loss+no_obj2_loss
        
        #Class Loss
        class_loss = self.mse(torch.flatten(exists_box * pred[..., :20]),
                              torch.flatten(exists_box * target[..., :20]))
        
        total_loss = (self.lambda_coord*coord_loss + 
                      object_loss + 
                      class_loss +
                      self.lambda_noobj*no_obj_loss)
        
        return total_loss
