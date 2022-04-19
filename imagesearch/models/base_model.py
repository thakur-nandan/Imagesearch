'''
A base network class that implements whatever will be shared across all of
our experiment models.

'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(torch.nn.Module):

    def __init__(self, output_vector_size=10):
        super(ImageEncoder, self).__init__()
        self.output_vector_size = output_vector_size
        self.conv1 = nn.Conv2d(3, 8, 3, padding='same')
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding='same')
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, padding='same')
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, output_vector_size)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x
    
    def get_output_vector_size(self):
        return self.output_vector_size

class ImageDecoder(torch.nn.Module):

    def __init__(self, input_vector_size=10):
        super(ImageDecoder, self).__init__()
        self.input_vector_size = input_vector_size
        self.fc1 = nn.Linear(input_vector_size, 128)
        self.fc2 = nn.Linear(128, 32 * 4 * 4)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 4, 4))
        self.deconv1 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.deconv2 = nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.deconv3 = nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.unflatten(x)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = self.bn3(self.deconv3(x))

        return x

def load_model(model_path, device=None):
    checkpoint = torch.load(model_path)
    try:
        output_vector_size=int(checkpoint['output_vector_size'])
    except:
        output_vector_size=10
    net = ImageEncoder(output_vector_size=output_vector_size)
    if device:
        net = net.to(device)
    net.load_state_dict(checkpoint['model'])
    return net