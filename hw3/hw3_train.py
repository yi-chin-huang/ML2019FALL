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

class dataset(Dataset):
    def __init__(self, data_dir, label):
        self.data_dir = data_dir
        self.label = label
        self.transform = transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.RandomRotation(30),
                    transforms.ToTensor(),
                ])
    def __getitem__(self, index):
        pic_file = str(index).zfill(5)+'.jpg'
        path = os.path.join(self.data_dir, pic_file)
        image = Image.open(path)
        image = ImageOps.equalize(image)
        img = self.transform(image)
        return torch.FloatTensor(img), self.label['label'][index]
    def __len__(self):
        return self.label.shape[0]

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


def train(model, epoch, train_loader, data_count):
    for e in range(0, epoch):
        model.train()
        train_loss = 0
        correct = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, label)
            loss_val = loss.item()
            train_loss += F.cross_entropy(output, label).item()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == label).sum().item()
        accu = correct/data_count
#         print('epoch', e)
#         print('accu', accu, 'loss', loss_val)

    torch.save(model.state_dict(), './model/cnn.pickle')


if __name__ == '__main__':

	y_train_pd = pd.read_csv(sys.argv[2])
	train_dataset = dataset(sys.argv[1], y_train_pd)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 256)
	
	data_count = len(train_dataset)
	cnn = CNN()
	use_cuda = torch.cuda.is_available()
	device = torch.device('cuda' if use_cuda else 'cpu')
	cnn = cnn.to(device)
	train(cnn, 1501, train_loader, data_count)
