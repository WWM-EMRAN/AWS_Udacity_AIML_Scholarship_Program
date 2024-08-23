# Imports python modules and packages 
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

# Imports functions created for this program
import utils
from utils import * 
# from classifier import * 




# Create new model with model structure criteria 
def create_model(model_name, num_classes, hidden_units, learning_rate, pretrained=True, device='cuda'):     
    utils.logger.info(f"### create_model START...")
    # Create pre-trained model: vgg19/alexnet/densenet121
#     print('----->', model_name, num_classes, hidden_units, learning_rate, pretrained, device)
    model = None 
    if model_name=='vgg19':
#          print('----->')
         model = models.vgg19(pretrained=pretrained)
    elif model_name=='alexnet':
        model = models.alexnet(pretrained=pretrained)
    elif model_name=='densenet121':
        model = models.densenet121(pretrained=pretrained)
    else:
        model = models.vgg19(pretrained=pretrained)
        
#     print('----->', model)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Create classifier layers
    classifier = None 
    if model_name=='vgg19':
        classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(hidden_units, num_classes),
                                 nn.LogSoftmax(dim=1))
    elif model_name=='alexnet':
        classifier = nn.Sequential(nn.Linear(9216, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(hidden_units, num_classes),
                                 nn.LogSoftmax(dim=1))
    elif model_name=='densenet121':
        classifier = nn.Sequential(nn.Linear(1024, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(hidden_units, num_classes),
                                 nn.LogSoftmax(dim=1))
    else:
        classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(hidden_units, num_classes),
                                 nn.LogSoftmax(dim=1))
        
    # Set model classifier 
    model.classifier = classifier 
    
    # Loss criteria
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
#     # Move the model to appropriate dice
#     model = model.to(device)    
    utils.logger.info(f"*** create_model END...")
    return model, criterion, optimizer



# Do training and validation on the training and validation sets 
def train_model(model, trainloader, validationloader, criterion, optimizer, epochs=5, print_every=20, device='cuda', extra_params=[]):   
    utils.logger.info(f"### train_model START...")
    
    # Move the model to appropriate dice
    model = model.to(device) 
    
    # Initialise the common variables used in the training process
    steps = 0
    running_loss = 0    
    ep_training_loss = [] 
    ep_validation_loss = [] 
    ep_validation_acc = [] 
#     print('111111')
    
    # start = time.time()
    for epoch in range(epochs):
#         print('222222')
    #     ep_start = time.time()
        all_training_loss = [] 
        all_validation_loss = [] 
        all_validation_acc = [] 
        for batch, (inputs, labels) in enumerate(trainloader):
            utils.logger.info(f"Training for- Epoch {epoch+1}, Batch {batch+1}... ")
#             print('333333')
    #         bt_start = time.time()
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

#             print(f'444444----111, {device}')
            logps = model(inputs)
#             print('444444----222')
            loss = criterion(logps, labels)

#             print('444444----333')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
#             print('444444')
            
            # Model validation after n batches
            if steps % print_every == 0:
                utils.logger.info(f'Validation for- Epoch {epoch+1}.. Batch {batch+1}...')
                validation_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validationloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model(inputs)
                        batch_loss = criterion(logps, labels)

                        validation_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
#                 tr_loss_ep_display = running_loss/print_every 
#                 val_loss_ep_display = validation_loss/len(validationloader) 
#                 val_acc_ep_display = (accuracy/len(validationloader))*100 
#                 print(f"Epoch {epoch+1}/{epochs}.. Batch {batch+1}.. "
#                       f"Train loss: {tr_loss_ep_display:.4f}.. "
#                       f"Validation loss: {val_loss_ep_display:.4f}.. "
#                       f"Validation accuracy: {val_acc_ep_display:.2f}%")
                utils.logger.info(f"Epoch {epoch+1}/{epochs}.. Batch {batch+1}.. Training loss: {running_loss/print_every:.4f}.. Validation loss: {validation_loss/len(validationloader):.4f}.. Validation accuracy: {(accuracy/len(validationloader))*100:.2f}%")
                all_training_loss.append(running_loss)  
                all_validation_loss.append(validation_loss) 
                all_validation_acc.append(accuracy) 
                running_loss = 0
                model.train()
                if len(extra_params)>0:
                    # save_model(model, model_name, num_classes, criterion, optimizer, train_data, hidden_units, learning_rate, pretrained=True, device=device)
                    # extra_params = [model_name, num_classes, train_data, hidden_units, learning_rate, True, device]
                    save_model(model, extra_params[0], extra_params[1], criterion, optimizer, extra_params[2], extra_params[3], extra_params[4], pretrained=extra_params[5], device=device)
                
        # Calculate losses and accuracy
        ep_training_loss.append(sum(all_training_loss)/len(all_training_loss))  
        ep_validation_loss.append(sum(all_validation_loss)/len(all_validation_loss)) 
        ep_validation_acc.append(sum(all_validation_acc)/len(all_validation_acc)) 
    utils.logger.info(f"*** train_model END...")
    return model, ep_training_loss, ep_validation_loss, ep_validation_acc 



# Do validation/test on the test set 
def test_model(model_, testloader, device='cuda'): 
    utils.logger.info(f"### test_model START...")
    model_ = model_.to(device) 
    model_.eval()
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model_(inputs)

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    model_.train()
    accuracy = (accuracy/len(testloader))*100 
    utils.logger.info(f"*** test_model END...")
    return accuracy



# Save the checkpoint/trained model 
def save_model(model_, model_name, num_classes, criterion, optimizer, train_data, hidden_units, learning_rate, pretrained=True, device='cuda'):
    utils.logger.info(f"### save_model START...")
    file_path = f'checkpoint_{model_name}.pth'
    model_.class_to_idx = train_data.class_to_idx
    model_info = {'model_name': model_name,
                  'pretrained': pretrained, 
                  'hidden_units': hidden_units, 
                  'learning_rate': learning_rate, 
                  'num_classes': num_classes, 
                  'features': model_.features, 
                  'classifier': model_.classifier, 
                  'criterion': criterion, 
                  'optimiszer': optimizer, 
                  'state_dict': model_.state_dict(),
                  'class_to_idx': {v:k for k,v in model_.class_to_idx.items()}}
    torch.save(model_info, file_path)
    utils.logger.info(f"{model_name} model is saved successfully in file: {file_path}")
    utils.logger.info(f"*** save_model END...")

    
    
# Loads a checkpoint and rebuilds the model
def load_saved_model(file_path, device='cuda'):
    utils.logger.info(f"### load_saved_model START...")
    model_info = torch.load(file_path)
    # saved_model = create_model(model_type=model_info['model_type'], num_output=model_info['num_classes']) 
    # create_model(model_name, num_classes, hidden_units, learning_rate, pretrained=True, device='cuda')
#     print('----->', model_info.keys())
    saved_model, criterion_, optimizer_ = create_model(model_info['model_name'], model_info['num_classes'], model_info['hidden_units'], model_info['learning_rate'], pretrained=model_info['pretrained'], device=device)
    saved_model.load_state_dict(model_info['state_dict'])
    saved_model.class_to_idx = model_info['class_to_idx']
    utils.logger.info(f"{model_info['model_name']} model is retrieved successfully from file: {file_path}")  
    utils.logger.info(f"*** load_saved_model END...")  
    return saved_model, model_info  
    
    
    
# Predict image data using the model 
def predict(model_, image_path, class_labels_and_names, topk=5):
    utils.logger.info(f"### predict START...")
    # Implement the code to predict the class from an image file
    probs, classes, class_names = None, None, None 
    model_.eval()
    
    with torch.no_grad(): 
        image = process_image(image_path)
#         print(image)
        image.unsqueeze_(0)
        image = image.float() 

        # Forward pass the data thru the model
        logps = model_(image)

        # Calculate the class probability
        ps = torch.exp(logps)
        probs, classes = ps.topk(topk, dim=1)

        probs = probs.data.numpy().squeeze().tolist() 
        idxs = classes.data.numpy().squeeze().tolist() 
        classes = [model_.class_to_idx[i] for i in idxs]
        
        class_names = [class_labels_and_names[i] for i in classes]

    utils.logger.info(f"*** predict END...")
    return probs, class_names

                
                
                


