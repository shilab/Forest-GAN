
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        
        img_size = args.img_size
        input_nc = args.latent_dim
        ngf = 128
        last_nc = 3

        conv_list = []
        K = np.log2(float(img_size)).astype(int) - 2
        out_channel = ngf * (2 ** (K - 1))

        if img_size == 32 or img_size == 64:
            self.first_k = 4
        elif img_size == 48:
            self.first_k = 6
        
        self.fc = nn.Sequential(
            nn.Linear(input_nc, out_channel * self.first_k * self.first_k),
            nn.BatchNorm1d(out_channel * self.first_k * self.first_k),
            nn.ReLU(inplace=True)
        )
        
        for i in range(K-1):
            in_channel = out_channel
            out_channel = in_channel // 2

            conv_list.append(nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, 5, 2, 2, 1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True))
            )
            
        conv_list.append(nn.Sequential(
            nn.ConvTranspose2d(ngf, last_nc, 5, 2, 2, 1),
            nn.Tanh())
        )

        self.conv = nn.Sequential(*conv_list)
    
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0)  
        
    def forward(self, input):
        
        features = self.fc(input)
        output = self.conv(features.view(input.size(0), -1, self.first_k, self.first_k))
        
        return output

class Discriminator(nn.Module):
    def __init__(self, args, is_train=True):
        super(Discriminator, self).__init__()
        
        self.n_disc = args.num_disc
        self.is_train = is_train

        img_size = args.img_size
        input_nc = 3
        ndf = 128

        conv_list = []
        K = np.log2(float(img_size)).astype(int) - 2
        last_k = img_size
        in_channel = input_nc
        out_channel = ndf 
           
        for i in range(K):
            
            conv_list.append(nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 5, 2, 2),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True))
            )

            in_channel = out_channel
            out_channel = in_channel * 2
            last_k = last_k // 2
            self.shared = nn.Sequential(nn.Sequential(*conv_list))
        
        self.branch_fc = nn.Linear(ndf * (2 ** (K - 1)) * last_k * last_k, 1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0)  


    def forward(self,input):
        
        shared = self.shared(input)
        features = shared.view(input.size(0), -1)
        output = self.branch_fc(features)
        # for i in range(self.n_disc -1):
        #     output = torch.cat([output, self.branch_fc[i+1](features)], 0)
        
        return output.squeeze(-1)
    

