import os

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = "../dataset"
FILE_PATH = os.path.join(DATA_PATH, "data.csv")
IMG_DIR = os.path.join(DATA_PATH, "images")
LABEL_DIR = os.path.join(DATA_PATH, "labels")

IMG_SIZE = 416

S = [IMG_SIZE // 32, IMG_SIZE // 16, IMG_SIZE // 8]

BS = 1

LR = 3e-4
DECAY = 0.01

EPOCHS = 20

MODEL_CONFIG = [
    [
        #backbone- darknet53
        (32, 3, 1),     #b0
        (64, 3, 2),     #b1
        ("res", 1),     #b2
        (128, 3, 2),    #b3
        ("res", 2),     #b4
        (256, 3, 2),    #b5
        ("res", 8),     #b6: res 0 for 2nd concatenation
        (512, 3, 2),    #b7
        ("res", 8),     #b8: res 1 for 1st concatenation
        (1024, 3, 2),   #b9
        ("res", 4),     #b10
    ],
    
    [
        #head
        ("convblock", 3),
        
        ("detection",),
        
        ("conv", 256, 1, 1),
        
        ("upsample", 2, 512),
        #After this upsample we will concat b8 output
        
        ("convblock", 3),
        
        ("detection",),
        
        ("conv", 128, 1, 1),
        
        ("upsample", 2, 256),
        #After this upsample we will concat b6 output
        
        ("convblock", 3),
        
        ("detection",)
    ]
]

# ANCHORS = [
#     [(116,90), (156,198), (373,326)],   #anchors for 13x13
#     [(30,61), (62,45), (59,119)],       #anchors for 26x26
#     [(10,13), (16,30), (33,23)],        #anchors for 52x52
#     ]

# def scale_anchors(x):
#     w = round(x[0]/IMG_SIZE, 2)
#     h = round(x[1]/IMG_SIZE, 2)
#     return (w, h)

# for idx, anchor in enumerate(ANCHORS):
#     ANCHORS[idx] = list(map(scale_anchors, anchor))

#Anchors after scaling
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]