import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torchvision import datasets, transforms
import os
from PIL import Image, ImageOps
import sys

class dataset_test(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.transform = transforms.Compose([
                            transforms.Grayscale(num_output_channels=1),
                            transforms.ToTensor(),
                        ])
    def __getitem__(self, index):
        pic_file = str(index).zfill(4)+'.jpg'
        path = os.path.join(self.data_dir, pic_file)
        image = Image.open(path)
        image = ImageOps.equalize(image)
        self.image = image
        img = self.transform(image)
        return torch.FloatTensor(img), index
    def __len__(self):
        return(len([name for name in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, name))]))

def predict(model, test_loader):
    prediction = []
    with torch.no_grad():
        for batch_idx, (img, index) in enumerate(test_loader):
            img = img.to(device)
            out = model(img)
            _,pred_label = torch.max(out,1)
            z = list(zip(index, pred_label))
            for i in range(len(z)):
                prediction.append((z[i][0].item(),z[i][1].item()))
    return prediction

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 48, 48)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=64,             # n_filters
                kernel_size=(5,5),          # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (64, 48, 48)
            nn.LeakyReLU(),                 # activation
            nn.BatchNorm2d(num_features=64),    # batch_size x num_features x height x width
            nn.MaxPool2d(kernel_size=(2,2)),    # choose max value in 2x2 area, output shape (16, 22, 22)
            nn.Dropout(0.25),
        )
        self.conv2 = nn.Sequential(         # input shape (64, 24, 24)
            nn.Conv2d(
                in_channels=64,            
                out_channels=128,           
                kernel_size=(3,3),              # filter size
                stride=1,
                padding=1,
            ),                              # output shape (128, 24, 24)
            nn.LeakyReLU(),                     # activation
            nn.BatchNorm2d(num_features=128),
            nn.MaxPool2d((2,2)),                # output shape (128, 12, 12)
            nn.Dropout(0.3),
        )
        self.conv3 = nn.Sequential(         # input shape (128, 12, 12)
            nn.Conv2d(
                in_channels=128,            
                out_channels=512,           # (512, 12, 12)
                kernel_size=(3,3),              # filter size
                stride=1,
                padding=1,
            ),                              
            nn.LeakyReLU(),                 
            nn.BatchNorm2d(num_features=512),
            nn.MaxPool2d((2,2)),                # output shape (512, 6, 6)
            nn.Dropout(0.35),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,           # (512, 6, 6)
                kernel_size=(3,3),          # filter size
                stride=1,
                padding=1,
            ),                              # output shape (512, 6, 6)
            nn.LeakyReLU(),                     # activation
            nn.BatchNorm2d(num_features=512),
            nn.MaxPool2d((2,2)),                # output shape (512, 3, 3)
            nn.Dropout(0.4),
        )
        self.dense1 = nn.Sequential(
            nn.Linear(512*3*3, 512),              # output shape (512)
            nn.ReLU(), 
            nn.BatchNorm1d(num_features=512),
            nn.Dropout(0.5),
        )
        self.dense2 = nn.Sequential(
            nn.Linear(512, 512),              # output shape (512)
            nn.ReLU(), 
            nn.BatchNorm1d(num_features=512),
            nn.Dropout(0.5),
        )
        self.out = nn.Sequential(
            nn.Linear(512, 7),              # output shape (512)
        )
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1) # flatten the output of conv2 to (batch_size, 32 * 9 * 9)
        x = self.dense1(x)
        x = self.dense2(x)
        output = self.out(x)
        return output

def out(res, name):
    f = open(name, 'w')
    print("id,label", file = f)
    for i, y in res:
        print(i, y, sep = ',', file = f)

if __name__ == '__main__':

    test_transform = transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    ])
    os.system('wget https://www.dropbox.com/s/a5lf0hlbrgqc7zu/cnn2_1500.pickle')
    model = CNN()
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    model.load_state_dict(torch.load('./cnn2_1500.pickle'))
    model.eval()
    test_dataset = dataset_test(sys.argv[1])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 256)
    y_pred = predict(model, test_loader)
    out(y_pred, sys.argv[2])

