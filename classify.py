#HW1 file classify.py
import pickle
import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data import DataLoader, TensorDataset
import sys
import argparse
import torchvision.transforms as transforms
from PIL import Image


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class myneural(nn.Module):
    def __init__(self, input_size=, hidden_size= , num_classes=10):
        super(myneural, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=.001)
        
        def forward(self,x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
        def train_model(self, train_loader,num_epochs=1):
            self.train()
            for epoch in range(num_epochs):
                for batch_i, (data, target) in enumerate (train_loader):
                    output = self.forward(data)
                    loss = self.criterion(output,target)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['train', 'test'])
    parser.add_argument('image_path', nargs='?', help='Path to test image')
    return parser.parse_args()
def preprocess(image_path):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                     'dog', 'frog', 'horse', 'ship', 'truck']
    transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(),
                                    transforms.Normalize((.5,.5,.5), (.5,.5,.5))])
    image = Image.open(image_path).convert('RGB') #make sure format consistent
    image_tensor = transform(image).unsqueeze(0)
    flattened = image_tensor.view(1,-1)
    return flattened


if __name__ == "__main__":
    args = parse_args()
    if args.command== 'train':
        #training script
        pass
    elif args.command == "test":
        if not args.image_path:
            print("Missing image Path")
            sys.exit(1)
        #testing script