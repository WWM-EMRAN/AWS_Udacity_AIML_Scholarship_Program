#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# *ImageClassifier/train.py
#                                                                         
# PROGRAMMER: Emran Ali 
# DATE CREATED: 24-12-2023
# REVISED DATE: 24-12-2023 
# PURPOSE: Train a Deep Learning (DL) model for image classification using some existing DL classifiers.
#     Tasks are:
#         -Take the arguments to create the DL model and the image data directory
#         -Create a model based on the arguments/with default set up
#         -Load data from the data directory
#         -Train, validate and test the model with appropriate performance metrics and error displayed
#         -Save the trained model to a checkpoint for later usage
#
# Use argparse Expected Call with <> indicating expected user input:
#      python train.py data_dir --save_dir checkpoint_save_directory --arch "vgg19" 
#             --learning_rate 0.001 --hidden_units 512 --epochs 3
#   Example call:
#     Basic usage: python train.py data_dir
#     * Set directory to save checkpoints: python train.py data_dir --save_dir save_directory 
#     * Choose architecture: python train.py data_dir --arch "vgg19" 
#     * Set hyperparameters: python train.py data_dir --learning_rate 0.001 --hidden_units 512 --epochs 3
#     * Use GPU for training: python train.py data_dir --gpu

# Imports python modules
from time import time  
import os
import glob

# Imports functions created for this program
import utils
from utils import * 
from classifier import * 


# Main program function defined below
def main():
    # Get logget
    # logger = get_logger(logger_name='root', log_file_name='./log_training_vgg19.txt') 
    # './log_training_vgg19.txt' './log_training_alexnet.txt' './log_training_densenet121.txt'  
    get_logger(logger_name=__name__, log_file_name='./log_training_vgg19.txt') 
    
    utils.logger.info(f"Start executing program...")
    start_time = time() 
    
    # Get command line arguments
    in_arg = get_training_input_args()
    utils.logger.info(f"Imput arguments for train.py: {in_arg}")
    # data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu 
    
    # Load data/image from the directory 
    dataloader, trainloader, validationloader, testloader, train_data = get_data_loaders(in_arg.data_dir) 
    
    # Get number of classes in the training set
    all_classes = [] 
    in_dir = f"{in_arg.data_dir}/train/"
    num_classes = len(next(os.walk(in_dir))[1]) 
    
    # Set up variables for training and testing criteria 
    epochs = in_arg.epochs
    print_every = 20 
    model_name = in_arg.arch #vgg19/alexnet/densenet121 
    learning_rate = in_arg.learning_rate
    hidden_units = in_arg.hidden_units

    # Use GPU if it's available
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if (in_arg.gpu and torch.cuda.is_available()) else "cpu") 
    utils.logger.info(f"Processing will be done using: {model_name} on {device}")

    # Create model object
    model, criterion, optimizer = create_model(model_name, num_classes, hidden_units, learning_rate, pretrained=True, device=device)  
    print(model)
    
    # Train model
    extra_params = [model_name, num_classes, train_data, hidden_units, learning_rate, True]
    model, ep_training_loss, ep_validation_loss, ep_validation_acc = train_model(model, trainloader, validationloader, criterion, optimizer, epochs=epochs, print_every=print_every, device=device, extra_params=extra_params)
    
    # Test model with test data 
    utils.logger.info('='*150)
    accuracy = test_model(model, testloader, device=device)
    # print(f"Test accuracy: {accuracy:.2f}")
    utils.logger.info(f"Test accuracy: {accuracy:.2f}%")
        
    # Save the checkpoint/trained model 
    save_model(model, model_name, num_classes, criterion, optimizer, train_data, hidden_units, learning_rate, pretrained=True, device=device)

    
    # Measure total program runtime by collecting end time
    end_time = time() 
    
    # Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time-start_time #calculate difference between end time and start time
    utils.logger.info(f"\n** Total Elapsed Runtime: {str(int((tot_time/3600)))}:{str(int((tot_time%3600)/60))}:{str(int((tot_time%3600)%60))}" )
    

# Call to main function to run the program
if __name__ == "__main__":
    main()

    