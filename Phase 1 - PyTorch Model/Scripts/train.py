#---------------- Prepare the command line -----------------------------
import argparse
import os
from os.path import isdir

def get_input_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument('--data_dir', type = str, default = '../Face Mask Dataset', 
                    help = 'Define a directory which contains the data.')
    
    parser.add_argument('--save_dir', type = str, default = 'saved_models/checkpoint.pth', 
                    help = 'Define save directory for checkpoints as str. (Default = \'saved_models/checkpoint.pth\')')
    
    parser.add_argument('--arch', type = str, default = 'densenet121', 
                    help = 'Pick a Pre-trained Model Architecture. (Default = densenet121)')
    
    parser.add_argument('--learning_rate', type = float, default = 0.003, 
                    help = 'Choose the learning rate of your optimizer (Default = 0.003)')
    parser.add_argument('--hidden_units', type = int, default = 512, 
                    help = 'Choose the hidden units of your first hidden layer. (Default = 512)')
    parser.add_argument('--epochs', type = int, default = 15, 
                    help = 'Choose the number of epochs (Default = 15).')
    
    parser.add_argument('--gpu', action='store_true', 
                    help = 'Train your network using a gpu.')
    
    return parser.parse_args()
#-----------------------------------------------------------------------------------------------------

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json

args = get_input_args()

#---------------- Prepare data from a data directory that is specified -------------------
print(f"\nPrepare data from the folder '{args.data_dir}'")

data_dir = args.data_dir
train_dir = data_dir + '/Train'
valid_dir = data_dir + '/Validation'
test_dir = data_dir + '/Test'

# Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.RandomRotation(30),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])
                                      ])
print("Train transforms pipeline is created.")

valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])
                                      ])
print("Vaild transforms pipeline is created.")

test_transforms  = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])
                                      ])

print("Test transform pipeline is created.")

# Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_dataset  = datasets.ImageFolder(test_dir , transform=test_transforms)


# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
print("\nTrain data loaded successfully.")

validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
print("Validation data loaded successfully.")

testloader  = torch.utils.data.DataLoader(test_dataset , batch_size=64)
print("Test data loaded successfully.")

print(f"\nCount of classes to be classified: {len(train_dataset.class_to_idx)}")
print('num_of_train_withoutmask {}/'.format(len(os.listdir(train_dir+'/WithoutMask'))),'num_of_train_withmask {}'.format(len(os.listdir(train_dir+'/WithMask'))))
print('num_of_test_withoutmask {}/'.format(len(os.listdir(test_dir+'/WithoutMask'))),'num_of_test_withmask {}'.format(len(os.listdir(test_dir+'/WithMask'))))
print('num_of_val_withoutmask {}/'.format(len(os.listdir(valid_dir+'/WithoutMask'))),'num_of_val_withmask {}'.format(len(os.listdir(valid_dir+'/WithMask'))))

#---------------- Build the model to be trained with the help of a pretrained architecture -------------------------------------------
#if args.arch == 'vgg16':
#    input_units = 25088
#elif args.arch == 'densenet121':
#    input_units = 1024
#else:
#    print("Only supported models for now are vgg16 and densenet121.")
#    exit()

#model = getattr(models, args.arch)(pretrained=True)
input_units = 1024
model = models.densenet121(pretrained=True)
print(f"\nThe model densenet121 is being created.....")

# Freeze
for param in model.parameters():
    param.requires_grad = False

print("\nThe Detector parameters are being frozen.")

print(f"\nThe hidden units count is {args.hidden_units}")
# Change classifier
classifier = nn.Sequential(
                            OrderedDict([ ('L1', nn.Linear(input_units, args.hidden_units)),
                                          ('relu1', nn.ReLU()),
                                          ('Dropout1', nn.Dropout(p=0.3)),
                                          ('L2', nn.Linear(args.hidden_units, 256)),
                                          ('relu2', nn.ReLU()),
                                          ('Dropout2', nn.Dropout(p=0.3)),
                                          ('L3', nn.Linear(256, len(train_dataset.class_to_idx))),
                                          ('output', nn.LogSoftmax(dim=1))
                                        ])
                          )

model.classifier = classifier

print("\nThe model classifier is changed: ")
print(model.classifier)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
print(f"\nAn Adam optimizer was created with a learning rate of {args.learning_rate}.")

#---------------- Start Training -------------------------------------------------
print(f"\nThe epochs specified {args.epochs}")
epochs = args.epochs


selected_device = "cuda" if args.gpu else "cpu"
print(f"\nThe choosen method for training is {selected_device}.")

if selected_device == 'cpu':
    available_device = 'cpu'
    print("\nStart training on the CPU.")
else:
    available_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if available_device == 'cpu':
        print("\nSorry, there no gpu founded. Start training on the CPU.")
    else:
        print("\nStart Training on the GPU.")

model.to(available_device)
for epoch in range(epochs):
    running_loss = 0
    valid_loss = 0
    accuracy = 0
    
    for inputs, labels in trainloader:
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(available_device), labels.to(available_device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
    else:
        model.eval()
        
        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(available_device), labels.to(available_device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)

                valid_loss += batch_loss.item()

                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {running_loss/len(trainloader):.3f}.. "
              f"Valid loss: {valid_loss/len(validloader):.3f}.. "
              f"Valid accuracy: {accuracy/len(validloader):.3f}")
        
        model.train()
        
#---------------- Testing the model ------------------------------------------
print("\nStart evaluating the model")

model.eval()

test_losses, accuracies = [], []

with torch.no_grad():
    for inputs, labels in testloader:
        test_loss = 0
        test_accuracy = 0
        
        inputs, labels = inputs.to(available_device), labels.to(available_device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)

        test_loss += batch_loss.item()

        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        test_losses.append(test_loss/len(testloader))
        accuracies.append(test_accuracy*100)
        
        print(f"Test loss: {test_loss/len(testloader):.3f}.. "
              f"Test accuracy: {test_accuracy*100:.3f}%")

print(f"\nAverage Loss: {sum(test_losses)/len(test_losses):.3f}")
print(f"Average Accuracy: {sum(accuracies)/len(accuracies):.3f}%")

#---------------- Saving the model -----------------------------------
model.class_to_idx = train_dataset.class_to_idx
model.class_to_idx

checkpoint = {
    #"model_arch": args.arch,
    "classifier": model.classifier,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    'class_to_idx': model.class_to_idx,
}

if isdir(args.save_dir):
    print(f"\nThe model is being save in the specified directory {args.save_dir}.")
    torch.save(checkpoint, args.save_dir)
else:
    print(f"\nThe specified directory {args.save_dir} is not found. The model will be saved in the root directory with the name 'checkpoint.pth'")
    torch.save(checkpoint, "checkpoint.pth")
