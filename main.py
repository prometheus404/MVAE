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


class MVAE(nn.Module):
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


    def forward(self,x_1, x_2):

        # In the forward method, we return the mean and variance
        # given by the encoder network and also the reconstruction mean
        # given by the decoder network using a sample from the
        # encoder's posterior distribution.

        mu_1,var_1,_ = self.encoder_1.encode_and_sample(x_1) #YOUR CODE HERE
        mu_2,var_2,_ = self.encoder_2.encode_and_sample(x_2) #YOUR CODE HERE
        # Generate the joint latent space -> N(m, C)
        C = 1/(1/var_1 + 1/var_2)
        m = C(var_1*mu_1 + var_2*mu_2)
        # Sample from the latent space
        eps = torch.randn_like(C)
        sample_z = eps * sqrt(C) + m
        # Decoder provides the mean of the reconstruction
        mu_x_1 = self.decoder_1.decode(sample_z)
        mu_x_2 = self.decoder_2.decode(sample_z)
        return mu_x_1, mu_x_2 ,m,C #TODO ricordati di modificare la parte del training in modo che usi mu_x_2


    # Reconstruction + KL divergence losses summed over all elements and batch

    def loss_function(self, x_1, x_2, mu_x_1, mu_x_2, mu_z, var_z):

        # We evaluate the loglikelihood in the batch using the function provided above

        logp_1,_ = eval_Gaussian_LL(x_1, mu_x_1,self.var_x)  #the first is for svhn the second one is for mnist
        logp_2,_ = eval_Gaussian_LL(x_2, mu_x_2,self.var_x)

        # KL divergence between q(z) and N()
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114

        KLz = -0.5 * torch.sum(1 + torch.log(var_z) - mu_z.pow(2) - var_z)

        # To maximize ELBO we minimize loss (-ELBO)
        return -logp_1 - logp_2 + KLz, -logp_1, -logp_2, KLz


class MVAE_extended(MVAE):

    def __init__(self, dimz=2,  channels=3, var_x=0.1,lr=1e-3,epochs=20,save_folder='./',restore=False):

        super().__init__(dimz,channels=3,var_x=0.1)

        self.lr = lr
        self.optim = optim.Adam(self.parameters(), self.lr)
        self.epochs = epochs

        self.save_folder = save_folder

        if(restore==True):
          state_dict = torch.load(self.save_folder+'VAE_checkpoint.pth')
          self.load_state_dict(state_dict)

        self.loss_during_training = []
        self.reconstruc_1_during_training = []
        self.reconstruc_2_during_training = []
        self.KL_during_training = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def trainloop(self,trainloader):

        nims = len(trainloader.dataset)

        self.train()

        for e in range(int(self.epochs)):

            train_loss = 0
            train_rec1 = 0
            train_rec2 = 0
            train_kl_l = 0

            idx_batch = 0

            for mnist, shvn in trainloader:
                
                #images = images.to(self.device)
                mnist = mnist.to(self.device)
                shvn = shvn.to(self.device)
                self.optim.zero_grad()
                
                mu_mnist, mu_shvn, mu_z, var_z = self.forward(mnist,shvn)

                loss, rec1, rec2, kl_l = self.loss_function(mnist, shvn, mu_mnist, mu_shvn, mu_z, var_z)

                loss.backward()

                train_loss += loss.item()
                train_rec1 += rec1.item()
                train_rec2 += rec2.item()

                train_kl_l += kl_l.item()

                self.optim.step()

                if(idx_batch%10==0):

                  torch.save(self.state_dict(), self.save_folder + 'VAE_checkpoint.pth')

                idx_batch += 1

            self.loss_during_training.append(train_loss/len(trainloader))
            self.reconstruc_1_during_training.append(train_rec1/len(trainloader))
            self.reconstruc_2_during_training.append(train_rec2/len(trainloader))
            self.KL_during_training.append(train_kl_l/len(trainloader))

            if(e%1==0):

                torch.save(self.state_dict(), self.save_folder + 'VAE_checkpoint.pth')
                print('Train Epoch: {} \tLoss: {:.6f}'.format(e,self.loss_during_training[-1]))


    def sample(self,num_imgs):

      with torch.no_grad():

        eps = torch.randn([num_imgs,self.dimz]).to(self.device)

        mnist_sample = self.decoder_1.decode(eps)
        shvn_sample = self.decoder_2.decode(eps)

        return mnist_sample.to("cpu").detach(), shvn_sample.to("cpu").detach()

