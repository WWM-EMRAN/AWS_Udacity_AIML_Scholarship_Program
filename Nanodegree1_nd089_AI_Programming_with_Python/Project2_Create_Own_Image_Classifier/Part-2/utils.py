# Imports python modules and packages 
import logging
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.transforms as T
from PIL import Image

# # Imports functions created for this program
# from utils import *
# from classifier import * 


# Read command line argument for train.py 
def get_training_input_args():
    logger.info(f"### get_training_input_args START...")
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window. 
    Arguments that can be provided: 
    Basic usage: python train.py data_dir
    * Set directory to save checkpoints: python train.py data_dir --save_dir save_directory 
    * Choose architecture: python train.py data_dir --arch "vgg19" 
    * Set hyperparameters: python train.py data_dir --learning_rate 0.001 --hidden_units 4096 --epochs 3
    * Use GPU for training: python train.py data_dir --gpu
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser() 
    
    # Create command line arguments as mentioned above using add_argument() from ArguementParser method 
    parser.add_argument('data_dir', type=str, default='./flowers/', help='path of the folder that contains the images of the flowers') 
    parser.add_argument('--save_dir', type=str, default='./', help='path to save the model checkpoint') 
    parser.add_argument('--arch', type=str, default='vgg19', help='CNN model to be used for image classification, chose between vgg19/alexnet') 
    parser.add_argument('--learning_rate', type=float, default=0.001, help='model learning rate for convergence') 
    parser.add_argument('--hidden_units', type=int, default=4096, help='number of hidden layers for the classifier model') 
    parser.add_argument('--epochs', type=int, default=3, help='number of training epochs to train the model') 
    parser.add_argument('--gpu', help='if GPU available and should be trained on GPU', action='store_true') 
    
    logger.info(f"*** get_training_input_args END...")
    return parser.parse_args() 



# Read command line argument for predict.py 
def get_predict_input_args():
    logger.info(f"### get_predict_input_args START...")
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window. 
    Arguments that can be provided: 
    Basic usage: python predict.py /path/to/image checkpoint
    * Return top K most likely classes: python predict.py input checkpoint --top_k 3 
    * Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json 
    * Use GPU for inference: python predict.py input checkpoint --gpu
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser() 
    
    # Create command line arguments as mentioned above using add_argument() from ArguementParser method 
    parser.add_argument('input', type=str, default='./flowers/test/1/image_06743.jpg', help='image path to predict') 
    parser.add_argument('checkpoint', type=str, default='./', help='model checkpoint path to load the model') 
    parser.add_argument('--top_k', type=int, default=5, help='Probabilities and names of top K values') 
    parser.add_argument('--category_names', type=str, default='./cat_to_name.json', help='path to the file where flower categories and names are stored') 
    parser.add_argument('--gpu', help='if GPU available and should be trained on GPU', action='store_true') 
    
    logger.info(f"*** get_predict_input_args END...")
    return parser.parse_args() 



# Construct data loaders from file path for training, validation and test data 
def get_data_loaders(data_dir):
    logger.info(f"### get_data_loaders START...")
    # Define file paths
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # General settings
    norm_mean = [0.485, 0.456, 0.406] 
    norm_std = [0.229, 0.224, 0.225]
    img_scale = 255 
    img_crop = 244 
    batch_size = 32
    
    image_transforms = transforms.Compose([transforms.Resize(img_scale),
                                           transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(img_crop),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(norm_mean, norm_std)])
    train_transforms = transforms.Compose([transforms.Resize(img_scale),
                                           transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(img_crop),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(norm_mean, norm_std)])
    validation_transforms = transforms.Compose([transforms.Resize(img_scale),
                                          transforms.CenterCrop(img_crop),
                                          transforms.ToTensor(),
                                          transforms.Normalize(norm_mean, norm_std)])
    test_transforms = transforms.Compose([transforms.Resize(img_scale),
                                          transforms.CenterCrop(img_crop),
                                          transforms.ToTensor(),
                                          transforms.Normalize(norm_mean, norm_std)])

    # Load the datasets with ImageFolder
    image_data = datasets.ImageFolder(data_dir, transform=image_transforms)
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    # Using the image datasets and the trainforms, define the dataloaders
    dataloader = torch.utils.data.DataLoader(image_data, batch_size=batch_size, shuffle=True)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    logger.info(f"*** get_data_loaders END...")
    return dataloader, trainloader, validationloader, testloader, train_data 



# Read class labels and associated class names from json file
def get_label_and_class_name(cat_to_name_file):
    logger.info(f"### get_label_and_class_name START...")
    with open(cat_to_name_file, 'r') as f:
        cat_to_name = json.load(f, strict=False)
    logger.info(f"*** get_label_and_class_name END...")
    return cat_to_name



# Process individual image to make it ready for prediction  
def process_image(image):
    logger.info(f"### process_image START...")
    # Process a PIL image for use in a PyTorch model
    norm_max = [255, 255, 255] 
    norm_mean = [0.485, 0.456, 0.406] 
    norm_std = [0.229, 0.224, 0.225]
    img_scale = 256 
    img_crop = 244 
    final_image = None
    
    with Image.open(image) as im:
        # Resize image
#         print(im.size)
        new_width, new_height = width, height = im.size 
        aspect_ratio = float(width)/height            
        width, height = (img_scale, int((img_scale/aspect_ratio))) if width<height else (int((img_scale*aspect_ratio)), img_scale) 
        im_preprocessed = im.resize((width, height))
        
        left = (width - img_crop)/2
        top = (height - img_crop)/2
        right = (width + img_crop)/2
        bottom = (height + img_crop)/2

        # Crop the center of the image
        im_preprocessed = im_preprocessed.crop((left, top, right, bottom))
#         print(im_preprocessed)
        
        # Normalised the image
        np_image = np.array(im_preprocessed)
        np_image = np_image.astype('float64')
        np_image = np_image/norm_max
        np_image = (np_image-norm_mean)/norm_std

        np_image = np_image.transpose((2, 0, 1))
        final_image = torch.from_numpy(np_image) #np_image #torch.from_numpy(np_image)
        
    logger.info(f"*** process_image END...")
    return final_image 
    
    
# Create dictionary of variables with keys as the name and values as the values in the dictionary
def create_dict_for_variables(*args):
    logger.info(f"### create_dict_for_variables START...")
    g = {k: v for k, v in globals().items() if not k.startswith('__')}  
    result = {}
    for arg in args:
        for k, v in g.items():
            try:
                if v == arg:
                    result[k] = v
                else:
                    result[k] = ""
            except ValueError:
                continue  # objects that don't allow comparison

    logger.info(f"*** create_dict_for_variables END...")
    return result



# Logger for file and stdout
# def get_logger(source_name):
#     import logging
#     logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
#     rootLogger = logging.getLogger()

#     fileHandler = logging.FileHandler(f"./logger_{source_name}.txt".format(logPath, fileName))
#     fileHandler.setFormatter(logFormatter)
#     rootLogger.addHandler(fileHandler)

#     consoleHandler = logging.StreamHandler()
#     consoleHandler.setFormatter(logFormatter)
#     rootLogger.addHandler(consoleHandler)
#     return rootLogger


def get_logger(logger_name='root', log_file_name='./log.txt', log_type=logging.INFO):
    logind = log_file_name.rfind('_')
    log_file_name_detail = f'{log_file_name[:logind]}_detail{log_file_name[logind:]}'
    log_file_name_detail_dict = f'{log_file_name[:logind]}_detail_dict{log_file_name[logind:]}'

    log_format = '%(message)s'
    log_format_detail = "%(asctime)s - [%(name)s]: [%(levelname)s] - %(module)s - %(funcName)s() - %(lineno)d \n%(message)s"
    log_format_detail_dict = "{'time':'%(asctime)s', 'loggername':'%(name)s', 'logtype':'%(levelname)s', 'module':'%(module)s',  " \
    "'function':'%(funcName)s()', 'linenumber':%(lineno)d: 'message':'%(message)s'}"

    global logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG) #DEBUG, INFO, WORNING, ERROR, CRITICAL, (logger.exception)

    # ### Log to message only log
    formatter = logging.Formatter(log_format)

    file_handler = logging.FileHandler(log_file_name, mode='a')
    # file_handler.setLevel(logging.ERROR)
    file_handler.setLevel(log_type)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # ### Log to standard output
    # # Log as error
    # stream_handler = logging.StreamHandler()
    # # Log as output
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

#     return logger







