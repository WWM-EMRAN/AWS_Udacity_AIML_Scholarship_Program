import torch
import torch.nn as nn 



    
### Define resnet block with avariable sub-blocks 
class My_ResNet_Block(nn.Module): 
    def __init__(self, in_channels, hidden_channels, expansion_factor, stride): 
        super().__init__()

        self.expansion_factor = expansion_factor
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels 
        
        ### Check if feature map has same size to be directly (identity) added to the next level opr go through transformation (downsample) 
        self.identity_block = None  
        if not self.in_channels == self.hidden_channels*self.expansion_factor: 
            feature_maping_layer = [
                nn.Conv2d(in_channels=self.in_channels, out_channels=self.hidden_channels*self.expansion_factor, kernel_size=1, stride=stride, padding=0, bias=False ), 
                nn.BatchNorm2d(self.hidden_channels*self.expansion_factor)
            ]
            self.identity_block = nn.Sequential(*feature_maping_layer) 
            
            
        # Basicblock 
        self.relu = nn.ReLU(inplace=True) 
        basic_blocks = [ 
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.hidden_channels, kernel_size=3, stride=stride, padding=1, bias=False ),
            nn.BatchNorm2d(self.hidden_channels),

            self.relu,

            nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=3, stride=1, padding=1, bias=False ),
            nn.BatchNorm2d(self.hidden_channels),
        ]
        self.basic_blocks = nn.Sequential(*basic_blocks)
            
            
    def forward(self,x):
        # Basicblock 
        F = self.basic_blocks(x) 
                        
        ### Check if feature map has same size (identity) or need to remap/change 
        residue = self.identity_block(x) if self.identity_block is not None else x 
        
        # Combine feature maps 
        H = F + residue 
        
        ### Final relu 
        H = self.relu(H)
        
        return H
    
    
    
    
    
    
    