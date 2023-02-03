import argparse
import os
import numpy as np
import math
import sys

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import utils



cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=False):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.sensitive_dim, 20, normalize=True),
            *block(20, 10, normalize=True),
            nn.Linear(10, 5),
            nn.Tanh(),
            #nn.Linear(20, 5),
            #nn.Tanh()
        )

    def forward(self, z, a):
        gen_input = torch.cat((a, z), -1)
        sam = self.model(gen_input)
        # img = img.view(img.shape[0], *img_shape)
        return sam


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(13, 64),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(512, 256),
            # nn.LeakyReLU(0.2, inplace=True),
            #nn.Linear(64, 20),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x, a):
        # Concatenate label embedding and image to produce input
        dis_in = torch.cat((a, x), -1)
        validity = self.model(dis_in)
        return validity


def train_cfgan(X_in, opt):
    # Initialize generator and discriminator
    generator = Generator(opt)
    discriminator = Discriminator()

    if cuda:
        generator.cuda()
        discriminator.cuda()

    # Configure data loader

    A = X_in[:, 0:8]
    X = torch.Tensor(X_in[:, 8:])
    A = torch.Tensor(A)
    D = torch.utils.data.TensorDataset(X, A)
    dataloader = torch.utils.data.DataLoader(D, batch_size=opt.batch_size, shuffle=True,)

    # Optimizers
    #optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
    #optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)
    # betas=(0.5, 0.999)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # # Adding noise for differential privacy (note that, clipping also needed)
    # if sigma is not None:
    #     for parameter in discriminator.parameters():
    #         parameter.register_hook(
    #             lambda grad: grad + (1 / opt.batch_size) * sigma * torch.randn(parameter.shape)
    #         )

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    batches_done = 0
    for epoch in range(opt.n_epochs):

        for i, x in enumerate(dataloader):

            # Configure input
            # real_sample = Variable(x.type(Tensor))
            real_sample = x[0]
            sen_att = x[1]
            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (len(x[0]), opt.latent_dim))))

            # Generate a batch of images

            fake_sample = generator(z, sen_att).detach()
            # Adversarial loss
            loss_D = -torch.mean(discriminator(real_sample, sen_att)) + torch.mean(discriminator(fake_sample, sen_att))

            loss_D.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)

            # Train the generator every n_critic iterations
            if i % opt.n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Generate a batch of images
                z = Variable(Tensor(np.random.normal(0, 1, (len(x[0]), opt.latent_dim))))
                gen_sample = generator(z, sen_att)
                # Adversarial loss
                sen_att_cf = utils.law_cf_sensitive(sen_att)
                fake_sample_cf = generator(z, sen_att_cf).detach()
                y_fake = fake_sample[:, -1]
                y_fake_cf = fake_sample_cf[:, -1]

                loss_G = -torch.mean(discriminator(gen_sample, sen_att)) + opt.lam * torch.mean((y_fake - y_fake_cf) ** 2)


                loss_G.backward()
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
                )

            # if batches_done % opt.sample_interval == 0:
            #    save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
            batches_done += 1

    z = Tensor(np.random.normal(0, 1, (X_in.shape[0], opt.latent_dim)))
    X_gen = generator(z, A)
    X_gen = torch.tensor(X_gen, requires_grad=False)
    X_gen = torch.cat([A, X_gen], -1)
    X_gen.detach().numpy()
    return X_gen
