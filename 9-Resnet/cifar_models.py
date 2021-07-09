import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomNet(nn.Module):
    def __init__(self, normalization='batchnorm'):
        super(CustomNet, self).__init__()

        valid_norm = ['groupnorm', 'layernorm', 'batchnorm']
     
        if normalization not in valid_norm:
            raise ValueError(f'Invalid normalization key: {normalization}')
        else:
            self.normalization = normalization

        self.conv1 = self.conv(3, (32, 32), padding=1) 
        self.conv2 = self.conv(32, (64, 32), padding=1) 
        self.conv3 = self.conv(64, (64, 32), padding=1) 
        self.conv4a = self.conv(64, (32, 32), kernel_size=1)
        self.conv4 = self.conv(32, (32, 32), dilation=2, padding=1)
        
        self.conv5 = self.conv(32, (64, 16), padding=1) 
        self.conv6 = self.depthwise_separable_conv(64, (64, 32)) 
        self.conv7 = self.conv(64, (32, 16), kernel_size=1)
        self.conv8 = self.conv(32, (32, 16), stride=2, padding=1) 

        self.conv9 = self.conv(32, (64, 8), padding=1) 

        self.conv11 = self.depthwise_separable_conv(64, (64, 16))

        self.gap = nn.AdaptiveAvgPool2d(1) 
        self.linear = nn.Linear(64, 10)

    def _get_normalization(self, out_channels):
        if self.normalization == 'batchnorm':
            return nn.BatchNorm2d(num_features=out_channels[0])
        elif self.normalization == 'groupnorm':
            return nn.GroupNorm(num_groups=2, num_channels=out_channels[0])
        return nn.LayerNorm((out_channels[0], out_channels[1], out_channels[1]))

    def conv(self, in_channels, out_channels, padding=0, stride=1, dilation=1, kernel_size=3, groups=1):
      return nn.Sequential(
          nn.Conv2d(in_channels=in_channels, out_channels=out_channels[0], kernel_size=kernel_size, stride=stride, padding=padding, bias=False, dilation=dilation, groups=1),
          nn.ReLU(),
          self._get_normalization(out_channels)      
          )
      
    def depthwise_separable_conv(self, in_channels, out_channels):
      return nn.Sequential(
          nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels),
          nn.Conv2d(in_channels=in_channels, out_channels=out_channels[0], kernel_size=1, bias=False),
          nn.ReLU(),
          self._get_normalization(out_channels)      
          )

    def forward(self, x):
        x = self.conv4(self.conv4a(self.conv3(self.conv2(self.conv1(x)))))
        x = self.conv8(self.conv7(self.conv6(self.conv5(x))))
        x = self.conv11(self.conv9(x))
        # x = self.conv15(self.conv14(self.conv13(x)))
        x = self.gap(x)
        # x = x.view(-1, 10)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return F.log_softmax(x)


class CustomResNet(nn.Module):
    def __init__(self, normalization='batchnorm'):
        super(CustomResNet, self).__init__()

        valid_norm = ['groupnorm', 'layernorm', 'batchnorm']
     
        if normalization not in valid_norm:
            raise ValueError(f'Invalid normalization key: {normalization}')
        else:
            self.normalization = normalization

        self.prep = self.conv(in_channels=3, out_channels=64, out_size=32, padding=1) # PrepLayer

        #   Layer 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )
        self.resblock1 = self.res_block(in_channels=128, out_channels=128)
        #   Add conv1 and resblock1

        #   Layer 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )

        #   Layer 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU()
        )
        self.resblock2 = self.res_block(in_channels=512, out_channels=512)
        #   Add conv3 and resblock2

        self.max_pool = nn.MaxPool2d(kernel_size=4, stride=4) 
        self.linear = nn.Linear(512, 10)

    def conv(self, in_channels, out_channels, out_size, padding=0, stride=1, dilation=1, kernel_size=3, groups=1):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, dilation=dilation, groups=1),
            self._get_normalization(out_channels, out_size),
            nn.ReLU(),      
            )

    def res_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def _get_normalization(self, out_channels, out_size):
        if self.normalization == 'batchnorm':
            return nn.BatchNorm2d(num_features=out_channels)
        elif self.normalization == 'groupnorm':
            return nn.GroupNorm(num_groups=2, num_channels=out_channels)
        return nn.LayerNorm((out_channels, out_size, out_size))
      
    def depthwise_separable_conv(self, in_channels, out_channels):
      return nn.Sequential(
          nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels),
          nn.Conv2d(in_channels=in_channels, out_channels=out_channels[0], kernel_size=1, bias=False),
          nn.ReLU(),
          self._get_normalization(out_channels)      
          )

    def forward(self, x):
        #   PREP LAYER
        x = self.prep(x)

        #   LAYER 1
        x = self.conv1(x)
        r1 = self.resblock1(x)
        x = torch.add(x, r1)

        #   LAYER 2
        x = self.conv2(x)

        #   LAYER 3
        x = self.conv3(x)
        r2 = self.resblock2(x)
        x = torch.add(x, r2)

        x = self.max_pool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return F.log_softmax(x)
