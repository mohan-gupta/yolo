import cv2

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import YOLOv3
from loss import YoloLoss
from dataset import get_loaders
from engine import train_loop, validate_loop
import config

def main(num_anchors=3, num_classes=80):
    start = 0
    scale=1.1
    transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=int(config.IMG_SIZE * scale)),
            A.PadIfNeeded(
                min_height=int(config.IMG_SIZE * scale),
                min_width=int(config.IMG_SIZE * scale),
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.RandomCrop(width=config.IMG_SIZE, height=config.IMG_SIZE),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
            ToTensorV2()
            ],
        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],)
        )

    train_loader, val_loader = get_loaders(config.FILE_PATH, config.IMG_DIR,
                                           config.LABEL_DIR, anchors=config.ANCHORS,
                                           transforms = transforms)

    net = YOLOv3(num_anchors, num_classes).to(config.DEVICE)

    loss_fn = YoloLoss()

    optimizer = torch.optim.AdamW(net.parameters(), lr=config.LR, weight_decay=config.DECAY)

    for epoch in range(start, start+config.EPOCHS):
        print(f"Epoch:{epoch+1}")
        train_loss = train_loop(net, train_loader, loss_fn, optimizer)
        val_loss = validate_loop(net, val_loader, loss_fn)

        print(f"Training Loss={train_loss} Validation Loss = {val_loss}")
    
if __name__ == "__main__":
    main()