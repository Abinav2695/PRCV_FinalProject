# Owner: Abinav Anantharaman and Satwik Bhandiwad
# Organization: Northeastern University.

#! /usr/bin/env python3

import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics import confusion_matrix
import numpy as np

from Net import Net, ResNet18, Net2, ModifiedNet2



labels = ['A','B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P','Q','R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing','space']
def predict_images(model, model_path, folder_path):
    # Load the model
    # model = Net()
    # model = Net2()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # Define the transforms
    transform = torchvision.transforms.Compose([
                                        torchvision.transforms.Resize((64,64)),
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])])

    # Load the images from a folder
  
    fig = plt.figure()
    for filename in os.listdir(folder_path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            # Load the image
            
            img_path = os.path.join(folder_path, filename)
            
            img = Image.open(img_path)
            # Apply the transforms to the image
            img = transform(img)

            # Get the model's prediction
            with torch.no_grad():
                output = model(img.unsqueeze(0))
                _, predicted = torch.max(output.data, 1)
                print('[INFO] Image {},, Prediction {}'.format(img_path,predicted.item()))
            
                # Plot the image and prediction
                img = img.permute(1, 2, 0).cpu().numpy()
                plt.imshow(img)
                plt.title(f'Prediction: {labels[predicted.item()]}')
                plt.show()


class Analysis():
    
    def __init__(self, folder_path = 'test_dataset_custom'):
        
        self.model_net1 = Net()
        self.model_net2 = ModifiedNet2()
        self.model_resnet18 = ResNet18()
        
        self.model_net1_path = 'models/model_net1_final.pth'
        self.model_net2_path = 'models/model_net2_modified.pth'
        self.model_resnet18_path = 'models/model_resnet18.pth'
        
        self.model_net1.load_state_dict(torch.load(self.model_net1_path))
        self.model_net1.eval()
        
        self.model_net2.load_state_dict(torch.load(self.model_net2_path))
        self.model_net2.eval()
        
        self.model_resnet18.load_state_dict(torch.load(self.model_resnet18_path))
        self.model_resnet18.eval()

        self.image_path_list = []
        self.true_labels = []
        self.pred_labels = []
        
        
        self.confusion_matrix_net1 = []
        self.confusion_matrix_net2 = []
        self.confusion_matrix_resnet18 = []
        
        self.folder_path = folder_path
        # Define the labels for the classes
        self.classes = ['A','B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P','Q','R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing','space']


        
        
        
        
    def load_img_data(self):
        
        # loop through the files in the folder in alphabetical order
        for filename in sorted(os.listdir(self.folder_path)):
            # check if the file is an image
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # add the path to the image_paths list
                self.image_path_list.append(os.path.join(self.folder_path, filename))
                image_label = filename.split('.')[0]
                self.true_labels.append(int(image_label))
        print(self.true_labels)
        
        
    def run(self):
        
        # Define the transforms
        for i in range(0,3):
            transform = None
            
            if(i==0):
                transform = torchvision.transforms.Compose([
                                        torchvision.transforms.Resize((32,32)),
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])])
            else:
                transform = torchvision.transforms.Compose([
                                        torchvision.transforms.Resize((64,64)),
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])])
            self.pred_labels = []
            
            for image_path in self.image_path_list:
                
                img = Image.open(image_path)
                # Apply the transforms to the image
                img = transform(img)
                
                # Get the model's prediction
                with torch.no_grad():
                    output = NotImplemented
                    if(i==0):
                        output = self.model_net1(img.unsqueeze(0))
                    elif(i==1):
                        output = self.model_net2(img.unsqueeze(0))
                    elif(i==2):
                        output = self.model_resnet18(img.unsqueeze(0))
                        
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    _, predicted_label = torch.max(output.data, 1)
                    # print('[INFO] Prediction {}'.format(self.labels[predicted.item()]))
                    confidence = probabilities[0][predicted_label.item()].item() * 100
                    prediction = predicted_label.item()
                    # print('[INFO] Prediction: {}, Confidence: {:.2f}%'.format(prediction, confidence))
                    
                    self.pred_labels.append(prediction)
            
            
            # print(self.true_labels)
            # print(self.net2_pred_labels)
                
            # get the confusion matrix
            cm = confusion_matrix(self.true_labels, self.pred_labels)

            # # Normalize the confusion matrix
            # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # calculate precision, recall, and accuracy
            precision = np.diag(cm) / np.sum(cm, axis=0)
            recall = np.diag(cm) / np.sum(cm, axis=1)
            accuracy = np.sum(np.diag(cm)) / np.sum(cm)

            print("Confusion matrix:\n", cm)
            print("Precision:", precision)
            print("Recall:", recall)
            print("Accuracy:", accuracy)
            
            self.plotting_fun(cm, i)
        
    def plotting_fun(self, confusion_matrix, fig_num):
        
        # Define the figure size and add a subplot for the confusion matrix
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 1, 1)

        # Create a heatmap for the confusion matrix
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()

        # Add the labels for the classes to the x and y axes
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)

        # Add the actual numbers for the confusion matrix as text annotations
        thresh = confusion_matrix.max() / 2.
        for i, j in np.ndindex(confusion_matrix.shape):
            plt.text(j, i, format(confusion_matrix[i, j], '.0f'),
                    horizontalalignment="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black")

        # Add axis labels and title
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig( 'confusion_matrix_' + str(fig_num) + '.png')
        # plt.show()
            
            
                
        
        


def main():

    my_analysis = Analysis()
    my_analysis.load_img_data()
    my_analysis.run()


if __name__ == '__main__':
    main()