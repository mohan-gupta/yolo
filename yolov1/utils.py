import torch

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
    
    return (intersection/area)

def save_model(model, optimizer):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
        }
    
    torch.save(checkpoint, "model.pt")