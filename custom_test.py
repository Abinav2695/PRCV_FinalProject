import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import os
from Net import Net, ResNet9



labels = ['A','B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P','Q','R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing','space']
def predict_images(model_path, optimizer_path, folder_path):
    # Load the model
    # model = Net()
    # model = Net()
    model = ResNet9()
    model.load_state_dict(torch.load(model_path))
    #model = torch.load(model_path)
    model.eval()
    # Define the transforms
    transform = torchvision.transforms.Compose([
                                        torchvision.transforms.Resize((32,32)),
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    # Load the images from a folder
    counter = 0
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



def main():
    # Set the paths to the model and image folder
    model_path = 'results/model_resnet9.pth'
    optimizer_path = 'results/optimizer.pth'
    # model_path = 'results/model.pth'
    
    folder_path = 'custom_test_dataset'

    # Call the predict_images function
    predict_images(model_path, optimizer_path, folder_path)


if __name__ == '__main__':
    main()