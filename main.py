from utils import encoder, decoder, imshow

import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch import optim, sqrt
import time
from os import listdir
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np

def eval_Gaussian_LL(x,mu_x,var_x):
    # x is a mini-batch of images. It has dimension [Batch,3,dimx,dimx]
    # mu_x is a mini-batch of reconstructed images. It has dimension [Batch,3,dimx,dimx]
    # var_x is a torch constant
    D = x.shape[1] * x.shape[2] * x.shape[3]   # Dimension of the image

    x = x.reshape(-1, D)

    mu_x = mu_x.reshape(-1, D)

    var_x = torch.ones_like(mu_x) * var_x

    # Constant term in the gaussian distribution
    cnt = D * np.log(2 * np.pi) + torch.sum(torch.log(var_x), dim=-1)

    # log-likelihood per datapoint

    logp_data = -0.5 * (cnt + torch.sum((x - mu_x) * var_x ** -1 * (x - mu_x), dim=-1))

    logp = torch.sum(logp_data)

    return logp,logp_data


class VAE(nn.Module):
    def __init__(self,dimz,channels=3,var_x=0.1):

        super().__init__()

        self.var_x = var_x

        self.dimz = dimz

        # We create two encoder network

        self.encoder_1 = encoder(self.dimz,channels,var_x) #YOUR CODE HERE
        self.encoder_2 = encoder(self.dimz,channels,var_x) #YOUR CODE HERE

        # We create two decoder network

        self.decoder_1 = decoder(self.dimz,channels,var_x) #YOUR CODE HERE
        self.decoder_2 = decoder(self.dimz,channels,var_x) #YOUR CODE HERE


    def forward(self,x):

        # In the forward method, we return the mean and variance
        # given by the encoder network and also the reconstruction mean
        # given by the decoder network using a sample from the
        # encoder's posterior distribution.

        mu_1,var_1,_ = self.encoder_1.encode_and_sample(x) #YOUR CODE HERE
        mu_2,var_2,_ = self.encoder_2.encode_and_sample(x) #YOUR CODE HERE
        # Generate the joint latent space -> N(m, C)
        C = 1/(1/var_1 + 1/var_2)
        m = C(var_1*mu_1 + var_2*mu_2)
        # Sample from the latent space
        var = None
        eps = torch.randn_like(var)# TODO capire come far ottenere var del joint latent space
        sample_z = eps * sqrt(C) + m
        # Decoder provides the mean of the reconstruction
        mu_x_1 = self.decoder_1.decode(sample_z)
        mu_x_2 = self.decoder_2.decode(sample_z)
        return mu_x_1, mu_x_2 ,m,C #TODO ricordati di modificare la parte del training in modo che usi mu_x_2


    # Reconstruction + KL divergence losses summed over all elements and batch

    def loss_function(self, x, mu_x_1, mu_x_2, mu_z, var_z):

        # We evaluate the loglikelihood in the batch using the function provided above

        logp_1,_ = eval_Gaussian_LL(x,mu_x_1,self.var_x)  #YOUR CODE HERE
        logp_2,_ = eval_Gaussian_LL(x,mu_x_2,self.var_x)

        # KL divergence between q(z) and N()
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114

        KLz = -0.5 * torch.sum(1 + torch.log(var_z) - mu_z.pow(2) - var_z)

        # To maximize ELBO we minimize loss (-ELBO)
        return -logp_1 - logp_2 + KLz, -logp, KLz
