import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg19
from glob import glob
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import random

## Define the FeatureExtractor class that extracts features from the VGG19 model
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:18]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, img):
        return self.feature_extractor(img)
    
## Define the ResidualBlock class that defines the residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)

## Define the Generator class that defines the generator network
class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(Generator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
        )

        self.res = nn.Sequential(
            *[ResidualBlock(64) for _ in range(n_residual_blocks)]
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )

        self.upsampling = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4)
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out  = self.res(out1)
        out  = self.conv2(out)
        out  = torch.add(out1, out)
        out  = self.upsampling(out)
        out  =  self.conv3(out)
        return out
    

## Define the Discriminator class that defines the discriminator network
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        def discriminator_block(in_filters, out_filters, stride=1, BN=True):
            block = []
            block.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=stride, padding=1))
            if BN:
                block.append(nn.BatchNorm2d(out_filters))
            block.append(nn.LeakyReLU(0.2))
            return block


        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, stride=1, BN=False),
            *discriminator_block(64, 64, stride=2),
            *discriminator_block(64, 128, stride=1),
            *discriminator_block(128, 128, stride=2),
            *discriminator_block(128, 256, stride=1),
            *discriminator_block(256, 256, stride=2),
            *discriminator_block(256, 512, stride=1),
            *discriminator_block(512, 512, stride=2),
            # nn.Flatten()
        )

        # self.classifier = nn.Sequential(
        #     nn.Linear(8*8*512, 1024),  # For 128x128 input
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(1024, 1),
        #     #nn.Sigmoid()
        # )
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        out = self.model(x)
        out = self.classifier(out)
        return out
 