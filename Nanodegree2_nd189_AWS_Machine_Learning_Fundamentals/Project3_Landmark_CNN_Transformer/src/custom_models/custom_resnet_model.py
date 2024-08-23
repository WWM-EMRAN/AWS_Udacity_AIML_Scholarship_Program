import torch
import torch.nn as nn 

from src.custom_models.custom_resnet_block import My_ResNet_Block 



# define the Residual CNN architecture 
class My_ResNet_Model(nn.Module):
    def __init__(self, resnet_structure_info, in_channels, num_classes, dropout=0.0): 
        super().__init__()
        
        ### Access network criteria to build the residual network 
        self.input_channels = resnet_structure_info[0]
        self.block_repeatition = resnet_structure_info[1]
        self.strides = resnet_structure_info[2]
        self.expansion_factor = resnet_structure_info[3] 
        
        init_nodes = 64 
        all_layers_with_different_blocks = [ 
            
            ### Basic block with convolution layer-batch normalisation-relu-max pooling 
            nn.Conv2d(in_channels=in_channels, out_channels=init_nodes, kernel_size=7, stride=2, padding=3, bias=False ),
            nn.BatchNorm2d(init_nodes),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),

            ### Four (4) resnet blocks with skip connections and repeatation of sub-blocks 
            self._create_resnet_block(init_nodes , self.input_channels[0], self.block_repeatition[0], self.expansion_factor, stride=self.strides[0] ),
            self._create_resnet_block(self.input_channels[0]*self.expansion_factor , self.input_channels[1], self.block_repeatition[1], self.expansion_factor,  stride=self.strides[1] ),
            self._create_resnet_block(self.input_channels[1]*self.expansion_factor , self.input_channels[2], self.block_repeatition[2], self.expansion_factor,  stride=self.strides[2] ),
            self._create_resnet_block(self.input_channels[2]*self.expansion_factor , self.input_channels[3], self.block_repeatition[3], self.expansion_factor,  stride=self.strides[3] ),

            ### Output layer with average pooling-flatten-dropout-fully connected layer  
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(), 
            nn.Dropout(p=dropout), 
            nn.Linear( self.input_channels[3]*self.expansion_factor , num_classes),
        ]
        
        self.resnet_model = nn.Sequential(*all_layers_with_different_blocks)
        
        
        
    def forward(self,x):
        x= self.resnet_model(x)
        
        return x
    
    
    
    def _create_resnet_block(self,in_channels, hidden_channels, num_repeat, expansion_factor, stride): 
        layers = [] 
        in_ch = in_channels 
        ### Generate repeatational sub-blocks within residual block 
        for i in range(num_repeat):
            st = stride if i==0 else 1  
            layers.append(My_ResNet_Block(in_ch, hidden_channels, expansion_factor, stride=st)) 
            in_ch = hidden_channels*expansion_factor  

        return nn.Sequential(*layers)

     
    
    
    
    