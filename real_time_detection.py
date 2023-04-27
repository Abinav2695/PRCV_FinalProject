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
import cv2




class RTD():
    
    def __init__(self, cam_source = 0, model_path = 'models/model_resnet18.pth'):
        
        self.labels = ['A','B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P','Q','R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing','space']
        self.play_feed = True
        # Load the model
        self.model  = ResNet18()
        # self.model  = Net2()
        
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Define the transforms
        self.transform = torchvision.transforms.Compose([
                                            torchvision.transforms.Resize((64,64)),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])])
        # open camera feed
        self.cap = None
        try:
            self.cap = cv2.VideoCapture(cam_source)
        except Exception as e:
            print("[ERROR] :", e)
            sys.exit(1) # Exit the program with a non-zero status code
            
        # define bounding box position and size
        self.bbox_size = 200
        self.bbox_pos = (50, 50)
        self.bbox_top_left = self.bbox_pos
        self.bbox_btm_right = (self.bbox_pos[0] + self.bbox_size, self.bbox_pos[1] + self.bbox_size)
    
    def run(self):
        
        while(self.play_feed):
            # read current frame
            ret, frame = self.cap.read()
            if not ret:
                print('[INFO] No feed')
                continue
            roi = frame[self.bbox_top_left[1]:self.bbox_btm_right[1], self.bbox_top_left[0]:self.bbox_btm_right[0]]
            
            # Convert the color space from BGR to RGB
            cv2_img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            
            # Convert the cv2 image to a PIL image
            pil_img = Image.fromarray(cv2_img)
            
            # Apply the transforms to the image
            img = self.transform(pil_img)

            # Get the model's prediction
            with torch.no_grad():
                output = self.model(img.unsqueeze(0))
                probabilities = torch.nn.functional.softmax(output, dim=1)
                _, predicted_label = torch.max(output.data, 1)
                # print('[INFO] Prediction {}'.format(self.labels[predicted.item()]))
                confidence = probabilities[0][predicted_label.item()].item() * 100
                prediction = self.labels[predicted_label.item()]
                print('[INFO] Prediction: {}, Confidence: {:.2f}%'.format(prediction, confidence))
            
            
            # draw bounding box on frame
            cv2.rectangle(frame, self.bbox_top_left, self.bbox_btm_right, (0, 255, 0), thickness=2)
            
            # set rectangle color and dimensions
            rect_color = (255, 255, 255)
            rect_top_left = (self.bbox_top_left[0], self.bbox_btm_right[1] + 10)
            rect_btm_right = (self.bbox_top_left[0] + 500, self.bbox_btm_right[1] + 40)

            # draw rectangle on frame
            cv2.rectangle(frame, rect_top_left, rect_btm_right, rect_color, cv2.FILLED)

            # put prediction and confidence text on frame
            cv2.putText(frame, "Prediction: {}, Confidence: {:.2f}%".format(prediction, confidence), 
                                (self.bbox_top_left[0], self.bbox_btm_right[1] + 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0, 0), 2)

            # display frame
            cv2.imshow('frame', frame)

            # save frame on key press
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('q'):
                self.play_feed = False
                
        # release camera and close window
        self.cap.release()
        cv2.destroyAllWindows()



def main():
    # Set the paths to the model and image folder
    # Check if the script was run with at least one argument
    if len(sys.argv) < 2:
        print("Usage: python myscript.py [cam source]")
        sys.exit(1)

    # Get the second argument, if provided
    arg2 = sys.argv[2] if len(sys.argv) > 2 else None

    # Get the first argument
    cam_source = int(sys.argv[1])

    #model_path
    model_path = None
    if arg2 is not None:
        model_path = arg2
    else:
        model_path = 'models/model_resnet18.pth'
        
    real_time_detection = RTD(cam_source, model_path)
    real_time_detection.run()
    
    
    
    
    
if __name__=='__main__':
    
    main()



