import torch

DEVICE = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = "../dataset/data.csv"
IMG_PATH = "../dataset/images"
LABEL_PATH = "../dataset/labels"

BS = 64
LR = 1e-2
EPOCHS = 135
DECAY = 0.0001

# (num_kernels, kernel size, stride, padding)
darknet_config = [
    (64, 7, 2, 3),
    (2, 2),
    (192, 3, 1, 1),
    (2, 2),
    (128, 1, 1, 0),
    (256, 3, 1, 1),
    (256, 1, 1, 0),
    (512, 3, 1, 1),
    (2, 2),
    (
        (256, 1, 1, 0),
        (512, 3, 1, 1),
        4,
    ),
    (1024, 3, 1, 1),
    (2, 2),
    (
        (512, 1, 1, 0),
        (1024, 3, 1, 1),
        2,
    ),
    (1024, 3, 1, 1),
    (1024, 3, 2, 1),
    (1024, 3, 1, 1),
    (1024, 3, 1, 1)
]