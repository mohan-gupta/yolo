import einops
import torch
import torch.nn as nn

import config

class ConvLayer(nn.Module):
    """Conv2d -> BatchNorm -> LeakyRelu"""
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        ) 
        
    def forward(self, x):
        return self.layer(x)
    
class ConvBlock(nn.Module):
    """1x1 ConvLayer followed by 3x3 ConvLayer"""
    def __init__(self, in_channels):
        super().__init__()
        
        self.block = nn.Sequential(
            ConvLayer(in_channels, in_channels//2, kernel_size=1),
            ConvLayer(in_channels//2, in_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        return self.block(x)
    

class ResidualBlock(nn.Module):
    """Residual connection after each ConvBlock"""
    def __init__(self, num_block, in_channels):
        super().__init__()
        self.num_block = num_block
        self.blocks = nn.ModuleList()
        
        for _ in range(num_block):
            self.blocks.append(ConvBlock(in_channels))
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x) + x
        
        return x
    
class ScalePred(nn.Module):
    """Detection Head"""
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.pred = nn.Conv2d(in_channels, num_anchors*(num_classes + 5), kernel_size=1)

        self.num_anchors = num_anchors
        self.pred_vec = num_classes+5
    
    def forward(self, x):
        out = self.pred(x)
        
        out = einops.rearrange(
            out,
            "b (num_anchors pred_vec) h w -> b num_anchors h w pred_vec",
            num_anchors = self.num_anchors, pred_vec = self.pred_vec
            )
        
        return out
    
class YOLOv3(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super().__init__()
        self.model_cfg = config.MODEL_CONFIG
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.backbone = self._create_backbone()
        self.head = self._create_head()
        
    def _create_backbone(self):
        backbone = nn.ModuleList()
        in_channels = 3
        for layer in self.model_cfg[0]:
            if len(layer) == 3:
                backbone.append(
                    ConvLayer(in_channels,
                              out_channels=layer[0],
                              kernel_size=layer[1],
                              stride=layer[2],
                              padding=1
                              )
                    )
                in_channels = layer[0]
                
            elif layer[0]=='res':
                backbone.append(
                    ResidualBlock(num_block=layer[1], in_channels=in_channels)
                )
                
        return backbone
    
    def _create_head(self):
        head = nn.ModuleList()
        in_channels = 1024
        
        for layer in self.model_cfg[1]:
            if layer[0] == "conv":
                head.append(
                    ConvLayer(in_channels,
                              out_channels=layer[1],
                              kernel_size=layer[2],
                              stride=layer[3],
                              )
                    )
                in_channels = layer[1]
            
            elif layer[0]=='convblock':
                head.extend(
                    [ConvBlock(in_channels=in_channels) for _ in range(layer[1])]
                )
                
            elif layer[0]=='upsample':
                head.append(nn.Upsample(scale_factor=layer[1]))
                in_channels += layer[2]
            
            elif layer[0]=='detection':
                head.append(
                    ScalePred(in_channels=in_channels,
                              num_anchors=self.num_anchors,
                              num_classes=self.num_classes)
                )
        
        return head

    def forward(self, x):
        detections = []
        res_conn = []

        for layer in self.backbone:
            x = layer(x)
            
            if isinstance(layer, ResidualBlock) and layer.num_block == 8:
                res_conn.append(x)

        res_idx = 1
        
        for layer in self.head:
            if isinstance(layer, ScalePred):
                res = layer(x)
                detections.append(res)
            else:
                x = layer(x)

                if isinstance(layer, nn.Upsample):
                    x = torch.cat((x, res_conn[res_idx]), dim=1)
                    res_idx -= 1

        return detections
            
if __name__ == "__main__":
    inp = torch.randn(1, 3, 416, 416)
    model = YOLOv3(num_anchors=3, num_classes=80)
    
    res = model(inp)
    print(res[0].shape)
    print(res[1].shape)
    print(res[2].shape)