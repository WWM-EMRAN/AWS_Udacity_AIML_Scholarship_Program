#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# *ImageClassifier/train.py
#                                                                         
# PROGRAMMER: Emran Ali 
# DATE CREATED: 24-12-2023
# REVISED DATE: 24-12-2023 
# PURPOSE: Predict an image using already saved Deep Learning (DL) model.
#     Tasks are:
#         -Take the arguments to load the DL model, image path and other display parameters
#         -Load the model and prepare for final operation
#         -Process the image for prediction
#         -Predict the data and calculate the probability values
#         -Return the flower names and probability value
# Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.
#
# Use argparse Expected Call with <> indicating expected user input:
#      python predict.py input checkpoint --top_k 3 
#             --category_names cat_to_name.json
#   Example call:
#     Basic usage: python predict.py /path/to/image checkpoint
#     * Return top K most likely classes: python predict.py input checkpoint --top_k 3 
#     * Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json 
#     * Use GPU for inference: python predict.py input checkpoint --gpu

# Imports python modules
from time import time 

# Imports functions created for this program
import utils
from utils import * 
from classifier import * 


# Main program function defined below
def main():
    # Get logget
    # logger = get_logger(logger_name='root', log_file_name='./log_training_vgg19.txt') 
    # './log_prediction_vgg19.txt' './log_prediction_alexnet.txt' './log_prediction_densenet121.txt'  
    get_logger(logger_name=__name__, log_file_name='./log_prediction_alexnet.txt') 
    
    utils.logger.info(f"Start executing program...")
    start_time = time() 
    
    
    # Get command line arguments
    in_arg = get_predict_input_args()
    utils.logger.info(f"Imput arguments for predict.py: {in_arg}")
    
    # input, checkpoint, top_k, category_names, gpu
        
    # Use GPU if it's available
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if (in_arg.gpu and torch.cuda.is_available()) else "cpu")
    utils.logger.info(f"Processing will be done using saved model file: {in_arg.checkpoint} on {device}")
    
    # Loads a checkpoint and rebuilds the model
    file_path = in_arg.checkpoint #f'checkpoint_{saved_model_name}.pth' #model_name=#vgg19/alexnet/densenet121
    saved_model, model_info = load_saved_model(file_path, device=device)
    utils.logger.info(saved_model)
    
    # Read class labels and class names 
    class_labels_and_names = get_label_and_class_name(in_arg.category_names) 
    utils.logger.info(class_labels_and_names)
    
#     # Predict image 
#     image_path = in_arg.input
#     topk = in_arg.top_k
#     probs, class_names = predict(saved_model, image_path, class_labels_and_names, topk)
    
    
#     # Prediction result 
#     utils.logger.info('='*150)
#     utils.logger.info(f"The image is classified as '{class_names[0]}' with probability score={probs[0]:.4f} \n") 
#     utils.logger.info(f"The top {topk} classes and their probabilities are: \n") 
#     utils.logger.info('_'*150)
#     for i, (pr, cl) in enumerate(zip(probs, class_names)):
#         utils.logger.info(f"{i+1:>3} --- {cl:>35} --- {pr:.4f}") 
    
    
    # Measure total program runtime by collecting end time
    end_time = time() 
    
    # Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time-start_time #calculate difference between end time and start time
    utils.logger.info(f"\n** Total Elapsed Runtime: {str(int((tot_time/3600)))}:{str(int((tot_time%3600)/60))}:{str(int((tot_time%3600)%60))}" )
    

# Call to main function to run the program
if __name__ == "__main__":
    main()

    