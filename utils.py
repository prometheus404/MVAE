import torch
from torch import nn
import numpy as np
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
import os
# MNIST-SVHN multi-modal model specification

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchnet.dataset import TensorDataset, ResampleDataset

def imshow(img):
    img = img / 2 + 0.5     # unnormalize to pot
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class encoder(nn.Module):
    def __init__(self,dimz,channels=3,var_x=0.1):
        super().__init__()

        self.dimz = dimz    #dimz is k, the dimension of the latent space

        # self.conv1 is a convolutional layer, with 32 output channels, kernel size 4, stride 2,
        # and padding 1

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32,
                               kernel_size=4, stride=2, padding=1)

        self.relu = nn.ReLU()

        # self.conv2 is a convolutional layer, with 32 output channels, kernel size 4, stride 2,
        # and padding 1

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32,
                               kernel_size=4, stride=2, padding=1) #YOUR CODE HERE

        # self.conv3 is a convolutional layer, with 64 output channels, kernel size 4, stride 2,
        # and padding 1

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=4, stride=2, padding=1)

        # self.conv4 is a convolutional layer, with 64 output channels, kernel size 4, stride 2,
        # and padding 1
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=4, stride=2, padding=1)  #YOUR CODE HERE

        # self.conv5 is a convolutional layer, with 256 output channels, kernel size 4, stride 1,
        # and padding 0

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=256,
                               kernel_size=4, stride=1, padding=0)


        # self.linear is a linear layer with dimz*2 outputs
        self.linear = nn.Linear(256,dimz*2) #YOUR CODE HERE

        self.softplus = nn.Softplus()

    def forward(self,x):
        # The forward method to project and image into a 2dimz dimensional vector
        z = self.relu(self.conv1(x))
        z = self.relu(self.conv2(z)) #YOUR CODE HERE
        z = self.relu(self.conv3(z))
        z = self.relu(self.conv4(z)) #YOUR CODE HERE
        z = self.relu(self.conv5(z))
        # Transform z into a 256-dim vector
        z = z.view(-1,256) #YOUR CODE HERE
        z = self.linear(z)

        return z

    def encode_and_sample(self,x,flag_sample=True):
        # This methods compute both the posterior mean and variance
        # Also we obtain a sample from the posterior using the
        # reparameterization trick.

        # We obtain the encoder projection using the forward method
        z = self.forward(x) #YOUR CODE HERE

        # The mean is the first dimz components of the forward output

        mu = z[:, :self.dimz] #YOUR CODE HERE

        # We compute the variance from the last dimz components using a
        # soft plus
        var = self.softplus(0.5 * z[:, self.dimz:])

        sample = None

        if(flag_sample==True):

            eps = torch.randn_like(var)

            sample = mu + eps*(var**0.5)

        return mu,var,sample


class decoder(nn.Module):
    def __init__(self,dimz,channels=3,var_x=0.1):

        super().__init__()

        # We expand z into a 256 dimensional vector

        self.linear = nn.Linear(dimz,256)

        self.relu = nn.ReLU()

        self.tanh = nn.Tanh()

        # self.tconv1 is a convolutional layer, with 64 output channels, kernel size 4, stride 1,
        # and padding 0

        self.tconv1 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=1,padding=0)

        # self.tconv2 is a convolutional layer, with 64 output channels, kernel size 4, stride 2,
        # and padding 1

        self.tconv2 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2,padding=1) #YOUR CODE HERE

        # self.tconv3 is a convolutional layer, with 32 output channels, kernel size 4, stride 2,
        # and padding 1

        self.tconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2,padding=1)

        # self.tconv3 is a convolutional layer, with 32 output channels, kernel size 4, stride 2,
        # and padding 1

        self.tconv4 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2,padding=1) #YOUR CODE HERE

        # self.tconv3 is a convolutional layer, with channels output channels, kernel size 4, stride 2,
        # and padding 1

        self.tconv5 = nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2,padding=1)

    def forward(self,z):

        x = self.relu(self.linear(z).view(-1,256,1,1))
        x = self.relu(self.tconv1(x)) #YOUR CODE HERE
        x = self.relu(self.tconv2(x))
        x = self.relu(self.tconv3(x)) #YOUR CODE HERE
        x = self.relu(self.tconv4(x))
        x = self.tanh(self.tconv5(x)) #YOUR CODE HERE
        return x

    def decode(self,z):

        # This function simply calls the forward method

        return self.forward(z)


def getDataLoaders(batch_size, shuffle=True, device='cuda'):
        if not (os.path.exists('./data/train-ms-mnist-idx.pt')
                and os.path.exists('./data/train-ms-svhn-idx.pt')) :
           #     and os.path.exists('./data/test-ms-mnist-idx.pt')
            #    and os.path.exists('./data/test-ms-svhn-idx.pt')):
            raise RuntimeError('Generate transformed indices with the script in bin')
        # get transformed indices
        t_mnist = torch.load('./data/train-ms-mnist-idx.pt')
        t_svhn = torch.load('./data/train-ms-svhn-idx.pt')
        #s_mnist = torch.load('./data/test-ms-mnist-idx.pt')
        #s_svhn = torch.load('./data/test-ms-svhn-idx.pt')

        # load base datasets
        t1 = mnist_train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        t2 = shvn_train_dataset = datasets.SVHN(root='./data', split='train',transform=transforms.ToTensor(), download=True)

        train_mnist_svhn = TensorDataset([
            ResampleDataset(t1, lambda d, i: t_mnist[i], size=len(t_mnist)),
            ResampleDataset(t2, lambda d, i: t_svhn[i], size=len(t_svhn))
        ])
        print(type(t1))
        print(type(ResampleDataset(t1, lambda d, i: t_mnist[i], size=len(t_mnist))))
        # test_mnist_svhn = TensorDataset([
        #    ResampleDataset(s1.dataset, lambda d, i: s_mnist[i], size=len(s_mnist)),
        #    ResampleDataset(s2.dataset, lambda d, i: s_svhn[i], size=len(s_svhn))
        # ])

        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        print(type(train_mnist_svhn))
        train = DataLoader(train_mnist_svhn, batch_size=batch_size, shuffle=shuffle, **kwargs)
        print(type(train))
        #test = DataLoader(test_mnist_svhn, batch_size=batch_size, shuffle=shuffle, **kwargs)
        return train #, test
