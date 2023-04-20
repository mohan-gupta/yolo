import torch

def iou_wh_score(anchors: torch.Tensor, target: torch.Tensor):
    """    
    anchors: Tensor = [..., w, h]
    target: Tensor = [..., w, h]
    returns: IOU score between anchors and target which have same centers
    """
    anchors_iou = []
    for anchor in anchors:
        anchor_w = anchor[..., 0]
        anchor_h = anchor[..., 1]
        
        target_w = target[..., 0]
        target_h = target[..., 1]
        
        w = torch.min(anchor_w, target_w)
        h = torch.min(anchor_h, target_h)
        
        intersection = w * h
        
        anchor_area = anchor_h * anchor_w
        target_area = target_h * target_w 
        
        area = (anchor_area + target_area - intersection + 1e-9)
        
        anchors_iou.append(intersection/area)
    
    return torch.tensor(anchors_iou)

def iou_score(pred: torch.Tensor, target: torch.Tensor):
    """    
    returns: IOU score between pred and target
    """
    pred_x1 = pred[..., 0:1] - (pred[..., 2:3] / 2)
    pred_y1 = pred[..., 1:2] - (pred[..., 3:4] / 2)
    pred_x2 = pred[..., 0:1] + (pred[..., 2:3] / 2)
    pred_y2 = pred[..., 1:2] + (pred[..., 3:4] / 2)
    
    target_x1 = target[..., 0:1] - (target[..., 2:3] / 2)
    target_y1 = target[..., 1:2] - (target[..., 3:4] / 2)
    target_x2 = target[..., 0:1] + (target[..., 2:3] / 2)
    target_y2 = target[..., 1:2] + (target[..., 3:4] / 2)
    
    x1 = torch.max(pred_x1, target_x1)
    y1 = torch.max(pred_y1, target_y1)
    
    x2 = torch.min(pred_x2, target_x2)
    y2 = torch.min(pred_y2, target_y2)
    
    intersection = (x2-x1).clamp(0) + (y2-y1).clamp(0)
    
    pred_area = abs((pred_x2 - pred_x1) * (pred_y2 - pred_y1))
    target_area = abs((target_x2 - target_x1) * (target_y2 - target_y1))
    
    area = (pred_area + target_area - intersection + 1e-9)
    
    return (intersection / area)
