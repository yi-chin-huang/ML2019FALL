import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from PIL import Image
import sys

class AutoEncoder(nn.Module):
    def __init__(self, dim):
        super(AutoEncoder, self).__init__()
        self.dim = dim
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(8),
            
            nn.Conv2d(8, 16, 3, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32))
        
        self.encoder2 = nn.Sequential(
            nn.Linear(4*4*32, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, self.dim))
        
        self.decoder1 = nn.Sequential(
            nn.Linear(self.dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 4*4*32))
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, 2),
            nn.ConvTranspose2d(16, 8, 2, 2),
            nn.ConvTranspose2d(8, 3, 2, 2),
            nn.Tanh())
        
    def forward(self, x):
        encoded1 = self.encoder1(x)
        encoded1 = encoded1.view(-1, 4*4*32)
        encoded = self.encoder2(encoded1)
        
        decoded1 = self.decoder1(encoded)
        decoded1 = decoded1.view(decoded1.size(0), 32, 4, 4)
        decoded = self.decoder2(decoded1)
        
        return encoded, decoded

    
def load_data():
    # detect is gpu available.
    use_gpu = torch.cuda.is_available()
    
    # load data and normalize to [-1, 1]
    trainX = np.load(sys.argv[1])
    trainX = np.transpose(trainX, (0, 3, 1, 2)) / 255. * 2 - 1
    trainX = torch.Tensor(trainX)

    # if use_gpu, send model / data to GPU.
    if use_gpu:
        trainX = trainX.cuda()

        train_dataloader = DataLoader(trainX, batch_size=256, shuffle=True)
        test_dataloader = DataLoader(trainX, batch_size=256, shuffle=False)
    
    return(train_dataloader, test_dataloader)

def train(train_dataloader, model, dim, epoch):
    
    use_gpu = torch.cuda.is_available()
    autoencoder = model(dim)
    
    # load data and normalize to [-1, 1]
    trainX = np.load(argv[1])

    if use_gpu:
        autoencoder.cuda()

    # We set criterion : L1 loss (or Mean Absolute Error, MAE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.005)

    cu_loss = []
    loss_x = []
    for e in range(1, epoch+1):
        cumulate_loss = 0
        for x in (train_dataloader):

            latent, reconstruct = autoencoder(x)
            loss = criterion(reconstruct, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cumulate_loss += loss.item() * x.shape[0]

        if e % 20 == 0:
            print(f'Epoch { "%03d" % e }: Loss : { "%.5f" % (cumulate_loss / trainX.shape[0])}')
            cu_loss.append(cumulate_loss / trainX.shape[0])
            loss_x.append(e)

        if e % 20 == 0:
            torch.save(autoencoder.state_dict(), './model/ae_{}_{}.pickle'.format(dim, e))   
    plt.plot(loss_x, cu_loss)
    return autoencoder

                          
def to_latents(model_class, test_dataloader):
    model = model_class(128)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    model.load_state_dict(torch.load('./model/autoencoder.pickle'))
    
    latents = []
    decoded = []
    for x in test_dataloader:
        latent, reconstruct = model(x)
        latents.append(latent.cpu().detach().numpy())
                  
    # Normalize
    latents = np.concatenate(latents, axis=0)
    latents = latents.reshape([len(test_dataloader.dataset), -1])

    latents_mean = np.mean(latents, axis=0)
    latents_std = np.std(latents, axis=0)
    latents = (latents - latents_mean) / latents_std

    return latents

def out(result, file):
    if np.sum(result[:5]) >= 3:
        result = 1 - result
    # Generate your submission
    df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
    df.to_csv(file,index=False)

if __name__ == '__main__':
    train_dataloader, test_dataloader = load_data()
#     autoencoder = train(train_dataloader, AutoEncoder, 128, 200)               
    latents = to_latents(AutoEncoder, test_dataloader)
    latents3 = TSNE(n_components=2).fit_transform(latents)
    result_t = KMeans(n_clusters = 2).fit(latents3).labels_
    out(result_t, sys.argv[2])
