import torch
import torchvision
import torchvision.models as models
import torch.nn as nn


def get_model_transfer_learning(model_name="resnet18", n_classes=50):

    # Get the requested architecture
    if hasattr(models, model_name):

        model_transfer = getattr(models, model_name)(pretrained=True)

    else:

        torchvision_major_minor = ".".join(torchvision.__version__.split(".")[:2])

        raise ValueError(f"Model {model_name} is not known. List of available models: "
                         f"https://pytorch.org/vision/{torchvision_major_minor}/models.html")

    # Freeze all parameters in the model
    # HINT: loop over all parameters. If "param" is one parameter,
    # "param.requires_grad = False" freezes it
    # YOUR CODE HERE 
    for param in model_transfer.parameters():
        param.requires_grad = False

    # Add the linear layer at the end with the appropriate number of classes
    # 1. get numbers of features extracted by the backbone
    num_ftrs  = model_transfer.fc.in_features # YOUR CODE HERE 
    
    # 2. Create a new linear layer with the appropriate number of inputs and
    #    outputs 
    # transfer_head = nn.Sequential(
    #     nn.AdaptiveAvgPool2d(output_size=1), 
    #     nn.Flatten(start_dim=1, end_dim=-1), 
    #     # nn.Dropout(p=0.2, inplace=False), 
    #     nn.Linear(in_features=num_ftrs, out_features=n_classes, bias=True), 
    # )
    # #### 
    # transfer_head = nn.Sequential(
    #     nn.AdaptiveAvgPool2d(1), 
    #     nn.Flatten(), 
    #     nn.Dropout(p=0.20), 
    #     nn.Linear( num_ftrs, n_classes ),
    # )
    #### 
    transfer_head = nn.Sequential(
        nn.BatchNorm1d(num_ftrs),
        nn.Linear(num_ftrs, num_ftrs * 2),
        nn.ReLU(),
        nn.BatchNorm1d(num_ftrs * 2),
        nn.Dropout(0.5),
        nn.Linear(num_ftrs * 2, n_classes)
    )
    model_transfer.fc = transfer_head  # YOUR CODE HERE  

    return model_transfer


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_get_model_transfer_learning(data_loaders):

    model = get_model_transfer_learning(n_classes=23)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)
    
    print(model, images.size()) 

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
