import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

os.makedirs("images", exist_ok=True)

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

class GAN:
    def __init__(self, latent_dim=100, img_size=28, channels=1, lr=0.0002, b1=0.5, b2=0.999):
        
        # Initialize Tensorboard writer
        logdir = './logs/linear/' + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir=logdir)
        
        self.latent_dim = latent_dim
        self.img_shape = (channels, img_size, img_size)
        self.lr = lr
        self.b1 = b1
        self.b2 = b2

        self.generator = Generator(latent_dim, self.img_shape)
        self.discriminator = Discriminator(self.img_shape)
        self.adversarial_loss = torch.nn.BCELoss()

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))

        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.adversarial_loss.cuda()

    def train(self, dataloader, n_epochs, sample_interval):
        logdir = 'linear/' + datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(f"images/{logdir}", exist_ok=True)
        
        for epoch in range(n_epochs):
            for i, (imgs, _) in enumerate(dataloader):
                    
                valid = torch.ones(imgs.size(0), 1, device='cuda' if self.cuda else 'cpu', requires_grad=False)
                fake = torch.zeros(imgs.size(0), 1, device='cuda' if self.cuda else 'cpu', requires_grad=False)

                real_imgs = imgs.to('cuda' if self.cuda else 'cpu')

                # Train Generator
                self.optimizer_G.zero_grad()
                z = torch.randn(imgs.size(0), self.latent_dim, device='cuda' if self.cuda else 'cpu')
                gen_imgs = self.generator(z)
                g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)
                g_loss.backward()
                self.optimizer_G.step()

                # Train Discriminator
                self.optimizer_D.zero_grad()
                real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
                fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.optimizer_D.step()

                # print losses to console
                print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

                # logging to tensorboard
                batches_done = epoch * len(dataloader) + i
                self.writer.add_scalar('Loss/Generator', g_loss.item(), batches_done)
                self.writer.add_scalar('Loss/Discriminator', d_loss.item(), batches_done)

                # save image samples
                if batches_done % sample_interval == 0:
                    grid = make_grid(gen_imgs.data[:25], nrow=5, normalize=True)
                    save_image(grid, f"images/{logdir}/{batches_done}.png")
                    # save_image(gen_imgs.data[:25], f"images/{batches_done}.png", nrow=5, normalize=True)
                    self.writer.add_image('Generated Images', grid, global_step=batches_done)
                    
        # shut down tensorboard
        self.writer.close()

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = DataLoader(
    datasets.MNIST("../../data/mnist", train=True, download=True,
                   transform=transforms.Compose([transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])),
    batch_size=64, shuffle=True)

# Create GAN instance and train
gan = GAN()
gan.train(dataloader, n_epochs=1, sample_interval=400)