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
  
    # fig = plt.figure()
    # for filename in os.listdir(folder_path):
    #     if filename.endswith('.png') or filename.endswith('.jpg'):
    #         # Load the image
            
    #         img_path = os.path.join(folder_path, filename)
            
    #         img = Image.open(img_path)
    #         # Apply the transforms to the image
    #         img = transform(img)

    #         # Get the model's prediction
    #         with torch.no_grad():
    #             output = model(img.unsqueeze(0))
    #             _, predicted = torch.max(output.data, 1)
    #             print('[INFO] Image {},, Prediction {}'.format(img_path,predicted.item()))
            
    #             # Plot the image and prediction
    #             img = img.permute(1, 2, 0).cpu().numpy()
    #             plt.imshow(img)
    #             plt.title(f'Prediction: {labels[predicted.item()]}')
    #             plt.show()

    
    fig = plt.figure(figsize=(15, 15))
    i =0 
    for filename in sorted(os.listdir(folder_path)):
        # check if the file is an image
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # filename = f"{i+1}.jpg"
            img_path = os.path.join(folder_path, filename)
            
            # Load the image
            img = Image.open(img_path)
            
            # Apply the transforms to the image
            img = transform(img)

            # Get the model's prediction
            with torch.no_grad():
                output = model(img.unsqueeze(0))
                _, predicted = torch.max(output.data, 1)
                true_label = labels[int(filename.split('.')[0])]
                
                # Plot the image and prediction
                img = img.permute(1, 2, 0).cpu().numpy()
                ax = fig.add_subplot(5, 6, i+1)
                ax.imshow(img)
                # print(predicted.item()[0])
                ax.set_title(f'True: {true_label}, Predicted: {labels[predicted.item()]}')
                i+=1
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.show()


def main():
    # Set the paths to the model and image folder
    # model_path = 'results/model_adam_opti.pth'
    # optimizer_path = 'results/optimizer_adam_opti.pth'
    
    if len(sys.argv) < 2:
        print("Usage: python myscript.py [mdoel num]")
        sys.exit(1)

    # Get the second argument, if provided
    arg2 = sys.argv[2] if len(sys.argv) > 2 else None
    
    model_path = None
    model = NotImplemented
    model_num  = int(sys.argv[1])
    if (model_num == 0):
        model_path = 'models/model_net1_final.pth'
        model = Net()
        
    elif (model_num == 1):
        model_path = 'models/model_net2_modified.pth'
        model = ModifiedNet2()
        
    elif (model_num == 2):
        model_path = 'models/model_resnet18.pth'
        model = ResNet18()
        
    folder_path = 'test_dataset_custom'
    
    # Call the predict_images function
    predict_images(model , model_path, folder_path)


if __name__ == '__main__':
    main()