import torch
import torch.nn as nn

import config
import sys

def get_conv(inp_channels, size, num_filters, padding, stride):
    conv  = nn.Conv2d(inp_channels,
                     num_filters,
                     kernel_size = size, 
                     stride=stride,
                     padding=padding,
                     bias=False)
    batch_norm = nn.BatchNorm2d(num_filters)
    leaky = nn.LeakyReLU(inplace=False)
    return [
        conv,
        batch_norm,
        leaky
    ]

class YOLOv1(nn.Module):
    def __init__(self, grid_size, num_classes, num_boxes):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        
        self.darknet = self.__create_darknet()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*self.grid_size*self.grid_size, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(4096, self.grid_size*self.grid_size*(self.num_classes + self.num_boxes*5))
        )
    
    def __create_darknet(self):
        darknet = []
        in_channels = 3
        for layer_conf in config.darknet_config:
            if len(layer_conf) == 4:
                conv_block = get_conv(in_channels, num_filters = layer_conf[0], size=layer_conf[1],
                                      stride=layer_conf[2], padding=layer_conf[3])
                darknet.extend(conv_block)
                in_channels = layer_conf[0]
            elif len(layer_conf) == 2:
                max_pool = nn.MaxPool2d(kernel_size=layer_conf[0], stride=layer_conf[1])
                darknet.append(max_pool)
            else:
                for _ in range(layer_conf[-1]):
                    conv_block1 = get_conv(in_channels, num_filters = layer_conf[0][0], size=layer_conf[0][1],
                                      stride=layer_conf[0][2], padding=layer_conf[0][3])
                    in_channels = layer_conf[0][0]
                    
                    conv_block2 = get_conv(in_channels, num_filters = layer_conf[1][0], size=layer_conf[1][1],
                                      stride=layer_conf[1][2], padding=layer_conf[1][3])
                    in_channels = layer_conf[1][0]
                    
                    darknet.extend(conv_block1)
                    darknet.extend(conv_block2)
        
        return nn.Sequential(*darknet)
    
    def forward(self, x):
        x = self.darknet(x)
        return self.fc(x)

if __name__ == "__main__":
    inp = torch.randn((1, 3, 448, 448), device="cuda")
    model = YOLOv1(7, 20, 2)
    model.to(device="cuda")
    out = model(inp)
    print(out.shape) #-> [1, 1470]
        