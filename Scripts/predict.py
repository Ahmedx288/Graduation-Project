#---------------- Prepare the command line -----------------------------
import argparse
from os.path import isdir

def get_input_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', type = str, default = 'saved_models/checkpoint.pth', 
                    help = 'Specifiy a file that conatins the model state. (Default = checkpoint.pth)')
    
    parser.add_argument('--image_path', type = str, default = 'Face Mask Dataset/Test/WithMask/1175.png', 
                    help = 'Specify an Image file to be used in the prediction process. (Default = Face Mask Dataset/Test/WithMask/1175.png)')
    
    parser.add_argument('--top_k', type = int, default = 2, 
                    help = 'Specify the number of top K likely classes to be displayed. (Default = 2)')
    
    parser.add_argument('--gpu', action='store_true', 
                    help = 'Inference using a gpu.')
    
    return parser.parse_args()
#-----------------------------------------------------------------------------------------------------

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import json

#---------------- Load Function ------------------------------------------
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    # Download pretrained model
    if checkpoint['model_arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif checkpoint['model_arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        
    # Freeze
    for param in model.parameters():
        param.requires_grad = False
    
    #Configure The model
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state'])
    
    #Optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
    optimizer.state = checkpoint['optimizer_state']
    
    return model, optimizer

#---------------- Image Preprocessing Function ------------------------------------------
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    
    #Store the image to be processed
    image = Image.open(image)
    
    # Convert image values to be between 0 and 1    
    image = np.array(image) / 255
    
    # Image normalization of colors
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = (image - mean)/std
    
    # Image dimensions reordering
    image = image.transpose((2, 0, 1))
    
    return image

#---------------- Predict Function ------------------------------------------
def predict(image_path, model, topk=2):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Proccess the input
    image = process_image(image_path)                                # Processed Image (Numpy)
    image_tensor = torch.from_numpy(image)                           # Tensor Image    (Tensor)
    image_feed = image_tensor.unsqueeze(0)                           # Add a dimension [1,3,224,224]
    
    # Go in evaluation mode
    model.eval()
    
    # Move the model to either CPU or GPU
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_device = "cuda" if args.gpu else "cpu"
    print(f"\nThe choosen method for predicting is {selected_device}.")

    if selected_device == 'cpu':
        available_device = 'cpu'
        print("\nStart predicting on the CPU.")
    else:
        available_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if available_device == 'cpu':
            print("\nSorry, there no gpu founded. Start predicting on the CPU.")
        else:
            print("\nStart predicting on the GPU.")
    
    with torch.no_grad():  
        model.to(device)
        image_feed.to(device)
        
        if device == 'cpu':
            logits = model.forward(image_feed.type(torch.FloatTensor))
        else:
            logits = model.forward(image_feed.type(torch.cuda.FloatTensor))
        
        ps = torch.exp(logits)
        top_p, top_class = ps.topk(topk, dim=1)
        top_p, top_class = top_p.cpu(), top_class.cpu()
        print("top class", top_class)
        
        #Invert the dictionary so you get a mapping from index to class
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        print(idx_to_class)
        
        top_classes = [idx_to_class[each] for each in top_class.numpy()[0]]
        print(top_classes)

    return top_p.numpy()[0].tolist(), top_classes
#----------------------------------------------------------------------------------------------------

args = get_input_args()
model = load_checkpoint(args.model_path)
image_path =  args.image_path
img = process_image(image_path)
probs, classes = predict(image_path, model, args.top_k)
print(probs)
print(classes)
         
