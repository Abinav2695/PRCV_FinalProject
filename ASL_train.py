# Owner: Abinav Anantharaman and Satwik Bhandiwad
# Organization: Northeastern University.

#! /usr/bin/env python3

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import DataLoader

import os
from Net import Net, Net2, ResNet18, ModifiedNet2

random_seed = 42
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

class ASL():

    def __init__(self):

        self.n_epochs = 10
        self.batch_size_train = 64
        self.batch_size_test = 64
        self.learning_rate = 0.001
        self.momentum = 0.5
        self.log_interval = 10
        self.weight_decay = 1e-5

    #load the training and validation dataset
    def load_data(self, size):
        train_loader = DataLoader(
            torchvision.datasets.ImageFolder('asl_alphabet_train/asl_alphabet_train',
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.Resize((size,size)),
                                        torchvision.transforms.ToTensor(),
                                        # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                    ])),

            batch_size=self.batch_size_train,
            shuffle= True)

        test_loader = DataLoader(
            torchvision.datasets.ImageFolder('asl_alphabet_test/asl_alphabet_test',
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.Resize((size,size)),
                                        torchvision.transforms.ToTensor(),
                                        # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                    ])),
            batch_size=self.batch_size_test,
            shuffle=False)

        return train_loader, test_loader
    

    def run(self, model_save_file_path,
                  optimizer_save_file_path,
                  epochs = 10, use_pretained = False, 
                  model_type = 1, ## 0 -  Net1, 1 - Net2, 3 - Resnet18
                  model_path = 'models/model_net2_final.pth', 
                  optimizer_path = 'models/optimizer_net2_final.pth'):
        size = None
        if (model_type == 0):
            size = 32
        else:
            size = 64
        print('[INFO] Loading Data')
        train_loader, test_loader = self.load_data(size)
        print('[INFO] Test classes: {}'.format(test_loader.dataset.classes))
        print('[INFO] Train classes: {}'.format(train_loader.dataset.classes))
        print('[INFO] Data loaded')
        # Load the previously trained model
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA device")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        
        # self.plot_train_images(train_loader)
        # return
        
        model = None
        optimizer = None
        
        if (model_type == 0):
            model = Net().to(device)
            optimizer = optim.SGD(model.parameters(), lr=self.learning_rate,
                            momentum=self.momentum)
            criterion = F.nll_loss
            
        elif (model_type == 1):
            # model = Net2().to(device)
            print(device)
            model = ModifiedNet2().to(device)
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            criterion = nn.CrossEntropyLoss()
            
        elif (model_type == 2):
            model = ResNet18().to(device)
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay= self.weight_decay)
            criterion = nn.CrossEntropyLoss()
        
        
        if(use_pretained):
            model_state = torch.load(model_path)
            optimizer_state = torch.load(optimizer_path)
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

        
        train_losses = []
        train_counter = []
        test_losses = []
        test_counter = [i * len(train_loader.dataset) for i in range(epochs + 1)]

        self.test(model, test_loader, test_losses, device, criterion)
        for epoch in range(1, epochs + 1):
            self.train(model, epoch, optimizer, train_loader, train_losses, train_counter, device, criterion, model_save_file_path, optimizer_save_file_path)
            self.test(model, test_loader, test_losses, device, criterion)
        self.plot_curve(train_counter, train_losses, test_counter, test_losses)

    #train function
    def train(self, model, epoch, optimizer, train_loader, train_losses, train_counter, device, criterion, model_save_file_path, optimizer_save_file_path):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            # output = self.network(data)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
                # torch.save(self.network.state_dict(), 'results/model.pth')
                torch.save(model.state_dict(), model_save_file_path)
                torch.save(optimizer.state_dict(), optimizer_save_file_path)

    #test function
    def test(self, model, test_loader, test_losses, device,criterion):
        # self.network.eval()
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                print(target)
                # output = self.network(data)
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    
    #plotting function
    def plot_curve(self, train_counter, train_losses, test_counter, test_losses):
        print(len(train_counter), len(train_counter), len(test_losses), len(test_counter))
        plt.plot(train_counter, train_losses, color='blue')
        plt.scatter(test_counter, test_losses, color='red')
        plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
        plt.xlabel('number of training examples seen')
        plt.ylabel('negative likelyhood loss')
        plt.savefig('training_plot_net1_modified.png')
        
    def plot_train_images(self, train_loader):
        
        # Plot the images
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))
        axes = axes.ravel()

        counter = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            
            axes[counter].imshow(np.transpose(images[counter], (1, 2, 0)))
            axes[counter].set_title(f"Label: {labels[counter].item()}")
            counter +=1
            if(counter == 10):
                break
        plt.tight_layout()
        plt.show()
    

def main():
    
    print('[INFO] Started Training')
    asl_train = ASL()
    asl_train.run('models/model_net2_modified.pth', 'models/optimizer_net2_modified.pth',epochs=10, use_pretained= False)

if __name__ == "__main__":

    # try:
    main()
    # except:
    #     print("[ERROR] Exception in main")