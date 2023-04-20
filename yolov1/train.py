import pandas as pd

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import YOLOv1
from loss import YoloLoss
from data import get_loaders
from engine import train_loop, validate_loop
import config
from utils import save_model

def main(grid_size=7, num_classes=20, num_boxes=2):
    start = 0
    
    transforms = A.Compose([
        A.Resize(448, 448),
        ToTensorV2()
    ])

    train_loader, val_loader = get_loaders(config.DATA_PATH, config.IMG_PATH,
                                           config.LABEL_PATH, transforms = transforms)

    net = YOLOv1(grid_size, num_classes, num_boxes).to(config.DEVICE)

    loss_fn = YoloLoss()

    optimizer = torch.optim.AdamW(net.parameters(), lr=config.LR, weight_decay=config.DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 105], gamma=0.1)

    for epoch in range(start, start+config.EPOCHS):
        print(f"Epoch:{epoch+1}")
        train_loss = train_loop(net, train_loader, loss_fn, optimizer)
        val_loss = validate_loop(net, val_loader, loss_fn)
        scheduler.step()

        print(f"Training Loss={train_loss} Validation Loss = {val_loss}")
    
    save_model(model=net, optimizer=optimizer)
    
if __name__ == "__main__":
    main()