import os
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.models as models
import torch
from torch import nn, optim
from PIL import Image
from collections import OrderedDict

import argparse

def verify_args():
    print("validating parameters")
    print(torch.__version__)
    print(torch.cuda.is_available()) 
    if (args.gpu and not torch.cuda.is_available()):
        raise Exception("Error: No GPU detected")
    
    if(not os.path.isdir(args.data_dir)):
        raise Exception('Error: The input directory does not exist!')
    
    data_dir = os.listdir(args.data_dir)
    if (not set(data_dir).issubset({'test','train','valid'})):
        raise Exception('Error: test, train or valid sub-directories are missing')      
    
    if args.arch not in ('vgg','resnet'):
        raise Exception('Error: Please choose one of: vgg or resnet')

def process_data(data_dir):
    print("processing data from input directory")
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = transforms.Compose([
                              transforms.Resize(255),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                             ])

    # Load the datasets with ImageFolder
    train_datasets = dsets.ImageFolder(train_dir, transform=data_transforms)
    valid_datasets = dsets.ImageFolder(valid_dir, transform=data_transforms)
    test_datasets  = dsets.ImageFolder(test_dir, transform=data_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=32, shuffle=True)
    validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=32, shuffle=True)
    testloaders  = torch.utils.data.DataLoader(test_datasets, batch_size=32, shuffle=True)

    return trainloaders, validloaders, testloaders

def init_model():
    print("Creating model object")
    
    device = 'cpu'
    if (args.gpu):
        device = 'cuda'

    arch_type = args.arch
    learning_rate = float(args.learning_rate)
    epochs = int(args.epochs)
    hidden_units = int(args.hidden_units)

    resnet18 = models.resnet18(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)
    modelarrs = {'resnet': resnet18, 'vgg': vgg16}

    model = modelarrs[arch_type]
    print(model)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    if torch.cuda.is_available() and device == 'cuda':
        model.cuda()
    
    return model, criterion, optimizer

def check_validation_set(valid_loader, device, model):    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total 

def train_model(model, trainloader, validloader, epochs, criterion, optimizer, device):
    epochs = epochs
    print_every = 30
    steps = 0

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_accuracy = check_validation_set(validloader, device, model)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Accuracy: {}".format(round(valid_accuracy,4)))

                running_loss = 0
    print("TRAINING model is completed")

def check_accuracy_on_test(testloader, device, model):    
    correct = 0
    total = 0
    print(f'The device is {device}.\n')
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    return correct / total 

def save_checkpoint(save_dir, model, optimizer, train_datasets):
    print("saving model")
    save_dir=save_dir
    model_arch = args.arch

    checkpoint = {'model': model_arch,
              'input_size': 25088,
              'output_size': 102,
              'features': model.features,
              'classifier': model.classifier,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              #'idx_to_class': {v: k for k, v in train_datasets.class_to_idx.items()}
             }

    torch.save(checkpoint, save_dir)

def create_model():
    verify_args()
    print('Creating model')
    data_directory = args.data_dir
    epochs = int(args.epochs)
    device = 'cpu'
    if (args.gpu):
        device = 'cuda'

    trainloaders, validloaders, testloaders = process_data(data_directory)

    model, criterion, optimizer = init_model()

    print("training the model")
    model = train_model(model, trainloaders, validloaders, epochs, criterion, optimizer, device)
    
    if (args.save_dir is not None):
        save_dir = args.save_dir
        save_checkpoint(save_dir, model, optimizer, trainloaders.dataset)
    return None

def main():
    print("creating the model")
    
    parser = argparse.ArgumentParser(description='Learning Model options.')
   
    # Command Line arguments
    parser.add_argument('data_dir', help='data directory (required)')
    parser.add_argument('--save_dir', help='directory for saving model.')
    parser.add_argument('--arch', help='models arch: vgg or densenet', default='vgg')
    parser.add_argument('--learning_rate', help='learning rate', default=0.001)
    parser.add_argument('--hidden_units', help='number of hidden units', default=120)
    parser.add_argument('--epochs', help='epochs', default=1)
    parser.add_argument('--gpu', action='store_true', help='gpu')
    global args 
    args = parser.parse_args()

    create_model()
    print("create and training model completed.")
    return None

main()