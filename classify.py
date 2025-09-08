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

Model_Directory = 'model'
Model_path = os.path.join(Model_Directory, 'new_model.pth')
Cifar_Directory = './cifar'
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                     'dog', 'frog', 'horse', 'ship', 'truck']

# this helps load the cifar data
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def cifar_loader(Cifar_Directory):
    print("loading from cifar")
    train_data = []
    train_labels = []
    for i in range(1,6):
        batch = os.path.join(Cifar_Directory,f'data_batch_{i}') # had to look this up
        batch_dictionary = unpickle(batch)
        train_data.append(batch_dictionary[b'data'])
        train_labels.extend(batch_dictionary[b'labels'])
    X_train = np.concatenate(train_data)
    X_train = X_train.reshape((5000,3,32,32)).astype("float32") /255.0
    y_train = np.array(train_labels)
    test_path = os.path.join(Cifar_Directory, 'test_batch')
    test_dictionary = unpickle(test_path)
    X_test = test_dictionary[b'data'].reshape((10000,3,32,32)).astype("float32") / 255.0
    y_test = np.array(test_dictionary[b'labels'])
    
    train_image_tensor = torch.tensor(X_train).view(-1,3*32*32)
    train_label_tensor = torch.tensor(y_train, dtype=torch.long)
    test_image_tensor = torch.tensor(X_test).view(-1,3*32*32)
    test_label_tensor = torch.tensor(y_test, dtype=torch.long)
    
    train_dataset = TensorDataset(train_image_tensor, train_label_tensor)
    test_dataset = TensorDataset(test_image_tensor, test_label_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print("loading successful")
    return train_loader, test_loader


class myneural(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, num_classes=10):
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
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['train', 'test'])
    parser.add_argument('image_path', nargs='?', help='Path to test image')
    return parser.parse_args()
def preprocess(image_path):
    transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(),
                                    transforms.Normalize((.5,.5,.5), (.5,.5,.5))])
    image = Image.open(image_path).convert('RGB') #make sure format consistent
    image_tensor = transform(image).unsqueeze(0)
    flattened = image_tensor.view(1,-1)
    return flattened

def train():
    train_loader, test_loader = cifar_loader(Cifar_Directory)
    model = myneural()
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_i, (data, target) in enumerate(train_loader):
            output = model(data)
            loss = model.criterion(output,target)
            
            model.optimizer.zero_grad()
            model.loss.backward()
            model.optimizer.step()
            running_loss += model.loss.item()
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad(): # I believe that we don't need to compute gradients in val
            for data, target in test_loader:
                ouput = model(data)
                y, predicted = torch.max(ouput.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        average_loss = running_loss/len(train_loader)
        accuracy = (correct/total) *100
        print(f"Epoch: {epoch}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}")
    print("training complete")
    if not os.path.exists(Model_Directory):
        os.makedirs(Model_Directory)
    torch.save(model.state_dict(), Model_path)
    print(f"Model saved to {Model_path}")
    
def test(image_path):
    if not os.path.exists(Model_path):
        print("no path found for model")
        return
    model = myneural()
    model.load_state_dict(torch.load(Model_path))
    model.eval()
    
    image_tensor = preprocess(image_path)
    with torch.no_grad():
        output = model(image_tensor)
        _,predicted = torch.max(output,1)
        prediction = class_names[predicted.item()]
        
    print(f"prediction for {image_path} is {prediction}")

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