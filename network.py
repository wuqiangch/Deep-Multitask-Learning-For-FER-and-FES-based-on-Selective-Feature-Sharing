import torch
import torch.nn as nn
import torch.nn.functional as F
from LeakyUnit import LeakyUnit

cfg = {
    'VGG11': [64, 'D', 128, 'D', 256, 256, 'D', 512, 512, 'D', 512, 512, 'D'],
    'VGG13': [64, 64, 'D', 128, 128, 'D', 256, 256, 'D', 512, 512, 'D', 512, 512, 'D'],
    'VGG16': [64, 64, 'D', 128, 128, 'D', 256, 256, 256, 'D', 512, 512, 512, 'D', 512, 512, 512, 'D'],
    'VGG19': [64, 64, 'D', 128, 128, 'D', 256, 256, 256, 256, 'D', 512, 512, 512, 512, 'D', 512, 512, 512, 512, 'D'],
}


class Decoder(nn.Module):
    def __init__(self, vgg_name='VGG13', out_channel=3):
        super(Decoder, self).__init__()
        self.module = self.make_layers(cfg[vgg_name], out_channel=out_channel)

    def forward(self, feature):
        y = self.module(feature)
        return y

    def make_layers(self, cfg, out_channel=3):
        layers = []
        in_channels = out_channel
        for x in cfg:
            if x == 'D':
                layers += [nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)]
            else:
                if in_channels == x:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.BatchNorm2d(x),
                               nn.ReLU(inplace=True)]
                else:
                    layers += [nn.Conv2d(x, in_channels, kernel_size=3, padding=1),
                               nn.BatchNorm2d(x),
                               nn.ReLU(inplace=True)]
                in_channels = x
        # layers += [nn.ConvTranspose2d(512, 512, kernel_size=3, padding=0)]
        layers = layers[::-1]
        return nn.Sequential(*layers)


class Transformer(nn.Module):
    def __init__(self, nz=7, nc=512, kernel_size=3):
        super(Transformer, self).__init__()
        self.transform_1d = nn.Sequential(
            nn.Linear(nz, nc),
            nn.ReLU(inplace=True)
        )
        self.transform_2d = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size, 1, 0, bias=False),
            nn.BatchNorm2d(nc),
            nn.ReLU(True),
            nn.Conv2d(nc, nc, 1, 1, 0, bias=False)
        )

    def forward(self, prob):
        f_label = self.transform_1d(prob).unsqueeze(2).unsqueeze(3)
        f_label = self.transform_2d(f_label)
        return f_label


class Classifier(nn.Module):
    def __init__(self, nz=7, nc=512, kernel_size=3):
        super(Classifier, self).__init__()
        self.pooling = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=kernel_size, padding=0),
            nn.BatchNorm2d(nc),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(nc, nz)

    def forward(self, feature):
        pooled = self.pooling(feature).view(feature.size(0), -1)
        out = self.classifier(pooled)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_shape=(1, 96, 96)):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)


class FERSNet(nn.Module):
    def __init__(self, vgg_name='VGG13', num_class=7, mem_size=512, kernel_size=3, k_channel=1):
        super(FERSNet, self).__init__()
        self.leakyunitxy = LeakyUnit(n_features=256)
        self.leakyunityx = LeakyUnit(n_features=256)
        self.in_c = nn.Sequential(
            nn.Conv2d(k_channel, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv1x = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv1y = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv2x = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(128),
        )
        self.conv2y = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv3x = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv3y = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv4x = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv4y = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.classifier = Classifier(nz=num_class, nc=512, kernel_size=kernel_size)
        self.decoder = Decoder(vgg_name=vgg_name, out_channel=k_channel)
        self.transformer = Transformer(nz=num_class, nc=mem_size, kernel_size=kernel_size)
        self.pooling = nn.MaxPool2d(stride=2, kernel_size=2)

    def forward(self, x, prob_t):
        f1 = self.pooling(self.in_c(x))

        f_x2, f_y2 = self.pooling(self.conv1x(f1)), self.pooling(self.conv1y(f1))
        f_x2_hat, r_xy2, z_xy2 = self.leakyunitxy(f_x2, f_y2)
        f_y2_hat, r_yx2, z_yx2 = self.leakyunityx(f_y2, f_x2)

        f_x3, f_y3 = self.pooling(self.conv2x(f_x2_hat)), self.pooling(self.conv2y(f_y2_hat))
        f_x3_hat, r_xy3, z_xy3 = self.leakyunitxy(f_x3, f_y3)
        f_y3_hat, r_yx3, z_yx3 = self.leakyunityx(f_y3, f_x3)

        f_x4, f_y4 = self.pooling(self.conv3x(f_x3_hat)), self.pooling(self.conv3y(f_y3_hat))
        f_x4_hat, r_xy4, z_xy4 = self.leakyunitxy(f_x4, f_y4)
        f_y4_hat, r_yx4, z_yx4 = self.leakyunityx(f_y4, f_x4)

        f_x5, f_y5 = self.pooling(self.conv4x(f_x4_hat)), self.pooling(self.conv4y(f_y4_hat))
        f_x5_hat, r_xy5, z_xy5 = self.leakyunitxy(f_x5, f_y5)
        f_y5_hat, r_yx5, z_yx5 = self.leakyunityx(f_y5, f_x5)

        prob5 = self.classifier(f_y5_hat)
        out = self.decoder(torch.cat((f_x5_hat, self.transformer(prob_t)), dim=1))

        return out, prob5


