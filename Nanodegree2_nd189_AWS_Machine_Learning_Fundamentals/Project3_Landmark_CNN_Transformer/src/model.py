import torch
import torch.nn as nn

# # Custom model import  
from src.custom_models.custom_resnet_model import My_ResNet_Model 


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))        
    
        ################## Custom Residual Network Model ##################  
        # ### [input_channels, block_repeatition, strides, expansion_factor]
        model_structure = ([64,128,256,512], [3,4,6,3], [1,2,2,2], 1) 
        # model_structure = ([64,128,256,512], [2,2,2,2], [1,2,2,2], 1) 
        self.model = My_ResNet_Model(model_structure, in_channels=3, num_classes=num_classes, dropout=dropout) 
        
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        
        x = self.model(x) 
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"

    
    