#! /usr/bin/env python3

import torch
import torchvision
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import DataLoader

import os
from Net import Net

random_seed = 42
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

class ASL():

    def __init__(self):

        self.network = Net()
        self.n_epochs = 10
        self.batch_size_train = 64
        self.batch_size_test = 64
        self.learning_rate = 0.01
        self.momentum = 0.5
        self.log_interval = 10

    #load the training and validation dataset
    def load_data(self):
        train_loader = DataLoader(
            torchvision.datasets.ImageFolder('asl_alphabet_train/asl_alphabet_train',
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.Resize((32,32)),
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                    ])),

            batch_size=self.batch_size_train,
            shuffle= True)

        test_loader = DataLoader(
            torchvision.datasets.ImageFolder('asl_alphabet_test/asl_alphabet_test',
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.Resize((32,32)),
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                    ])),
            batch_size=self.batch_size_test,
            shuffle=False)

        return train_loader, test_loader
    

    def run(self, epochs=1):
        print('[INFO] Loading Data')
        train_loader, test_loader = self.load_data()
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
        
        return
        model = Net().to(device)
        # model.load_state_dict(torch.load('results/model.pth'))
        # Load the previously used optimizer state
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate,
                            momentum=self.momentum)
        # optimizer.load_state_dict(torch.load('results/optimizer.pth'))

        # optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate,
        #                     momentum=self.momentum)
        train_losses = []
        train_counter = []
        test_losses = []
        test_counter = [i * len(train_loader.dataset) for i in range(epochs + 1)]

        self.test(model, test_loader, test_losses, device)
        for epoch in range(1, epochs + 1):
            self.train(model, epoch, optimizer, train_loader, train_losses, train_counter, device)
            self.test(model, test_loader, test_losses, device)
        self.plot_curve(train_counter, train_losses, test_counter, test_losses)

    #train function
    def train(self, model, epoch, optimizer, train_loader, train_losses, train_counter, device):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            # output = self.network(data)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
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
                torch.save(model.state_dict(), 'results/model.pth')
                torch.save(optimizer.state_dict(), 'results/optimizer.pth')

    #test function
    def test(self, model, test_loader, test_losses, device):
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
                test_loss += F.nll_loss(output, target, reduction='sum').item()
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
        plt.ylabel('negative log likelihood loss')
        plt.savefig('training_plot_dummy.png')
    

def main():
    
    print('[INFO] Started Training')
    asl_train = ASL()
    asl_train.run(epochs=10)

if __name__ == "__main__":

    # try:
    main()
    # except:
    #     print("[ERROR] Exception in main")