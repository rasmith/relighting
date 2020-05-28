import torch
import torch.nn as nn
import models.small_precoder as p


class Identity(nn.Module):
    def forward(self, x):
        return x


class PrecodedDiscriminator(nn.Module):
    def __init__(self,
                 num_layers,
                 num_input_channels,
                 precoder_layer=p.SmallPrecoder,
                 normalization_layer=Identity):
        super(PrecodedDiscriminator, self).__init__()
        conv_layer = lambda i: nn.Conv2d(max(2**(i + 2), 6),
                                         2**(i + 3),
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
        relu_layer = lambda i: nn.LeakyReLU(0.2, inplace=True)
        norm_layer = lambda i: normalization_layer()
        self.precoder = precoder_layer(num_input_channels)
        self.layers = []
        for i in range(num_layers):
            self.layers += [conv_layer(i)]
            self.layers += [norm_layer(i)]
            self.layers += [relu_layer(i)]
        self.layers += [
            nn.Conv2d(2**(num_layers + 2),
                      1,
                      kernel_size=1,
                      stride=1,
                      padding=0)
        ]
        self.model = nn.Sequential(*self.layers)

    def forward(self, x, y):
        y = self.precoder(y)
        x = torch.cat((x, y), 1)
        x = self.model(x)
        return x
