from torchvision import transforms
from PIL import Image
import torch
from torch import tensor
from torchvision import transforms
import argparse
import numpy as np
import json
import torchvision.models as models

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img_pil = Image.open(image)
   
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = adjustments(img_pil)
    
    return img_tensor

def predict(image_path, model, topk=5, device='cuda'):
    '''
    Arguments: The path to the image, the model, the number of prefictions and whether cuda will be used or not
    Returns: The "topk" most probable choices that the network predicts
    '''
    with torch.no_grad():
        img_torch = process_image(image_path)
        img_torch = img_torch.unsqueeze_(0)
        img_torch = img_torch.float()
        if torch.cuda.is_available() and device == 'cuda':
            model.to('cuda:0')
            outputs = model.forward(img_torch.cuda())
        else:
            outputs = model.forward(img_torch)

        probs, classes = torch.exp(outputs).topk(topk)
        return image_path, probs, classes
    
def display_prediction(image_path, probs, classes, cat_filename='cat_to_name.json'):
    file = cat_filename
    with open(file, 'r') as f:
        class_mapping =  json.load(f)
        
    a = np.array(probs[0].tolist())
    labels = [class_mapping[str(index + 1)] for index in np.array(classes[0].tolist())]
    print(f"Predict: {image_path}")
    print(labels)
    print(a)
    
    top_k = args.top_k
    print(f'Top {top_k} Predictions')
    i=0 
    while i < len(labels):
        print("{}, a probability of {}".format(labels[i], a[i]))
        i += 1

def parse_args():
    parser = argparse.ArgumentParser(description='predict an image!')
    parser.add_argument('image_input', help='image file to classifiy')
    parser.add_argument('checkpoint', help='model used for classification')
    parser.add_argument('--top_k', help='prediction categories', default=3, type=int)
    parser.add_argument('--category_names', help='file for category names')
    parser.add_argument('--gpu', action='store_true', help='gpu option')
    args = parser.parse_args()
    return args

def load_checkpoint(filepath):
    model_info = torch.load(filepath)

    arch_type = model_info['model']
    print('Loading checkpoint..')
    print('Model: {0}', arch_type)

    resnet18 = models.resnet18(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)
    modelarr = {'resnet': resnet18, 'vgg': vgg16, 'vgg16': vgg16}

    model = modelarr[arch_type]
    for param in model.parameters():
            param.requires_grad = False

    device = 'cpu'
    if (args.gpu and torch.cuda.is_available()):
        device = 'cuda'
    model.to(device)
    model.classifier = model_info['classifier']
    model.optimizer = model_info['optimizer']

    model.load_state_dict(model_info['state_dict'])
    return model, model_info

def main():
    global args
    args = parse_args() 
    if (args.gpu and not torch.cuda.is_available()):
        raise Exception("No GPU detected")
    
    device = 'cpu'
    if (args.gpu):
        device = 'cuda'

    top_k = args.top_k
    image_path = args.image_input

    path = args.checkpoint
    model, model_info= load_checkpoint(path)

    image_path, probs, classes = predict(image_path, model, top_k, device)
    display_prediction(image_path, probs, classes)

main()