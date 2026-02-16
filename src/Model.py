import torch
from torchvision import models


#Loads pretrained VGG19 model and freezes its parameters
def load_vgg(device="cpu"):  
  
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

    for param in vgg.parameters():
        param.requires_grad_(False)

    vgg.to(device)
    return vgg


#Extracts intermediate feature maps from selected VGG layers
def get_features(image, model):  
   

    layers = {
        '0': 'conv1_1',
        '5': 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',
        '21': 'conv4_2',  # Content layer
        '28': 'conv5_1'
    }

    features = {}

    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features
