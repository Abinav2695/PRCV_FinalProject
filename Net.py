#! /usr/bin/env python3

import torch
import torchvision
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
import os

random_seed = 42
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class Net(nn.Module):
    def __init__(self, input_channels=3, output_classes = 29):
        super(Net, self).__init__()

        # Define the layers of the network
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  ##output size 32 * width/4 * height/4

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) ##output size 64 * width/4 * height/4

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(in_features=128 * 4 * 4, out_features=1024)
        self.relu4 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(in_features=1024, out_features=output_classes)  #29 alphabets


    def forward(self, x):
        # Define the forward pass of the network
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.dropout1(x)
        
        x = x.view(-1, 128 * 4 * 4)

        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout2(x)

        x = self.fc2(x)

        return F.log_softmax(x)


# Define the ResNet9 model
class ResNet9(nn.Module):
    def __init__(self, num_classes=29):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x



def main():

    network = Net()
    print(network)

if __name__ == "__main__":

    try:
        main()
    except:
        print("[ERROR] Exception in main")