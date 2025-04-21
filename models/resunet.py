import torch.nn as nn
import torch
import snntorch as snn

class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(cin),
            snn.Leaky(beta=0.95),
            nn.Conv2d(
                cin, cout, kernel_size=kernel_size, stride=stride, padding=kernel_size//2
            ),
            nn.BatchNorm2d(cout),
            snn.Leaky(beta=0.95),
            nn.Conv2d(cout, cout, kernel_size=kernel_size, padding=kernel_size//2),
        )
        self.res = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=kernel_size, stride=stride, padding=kernel_size//2),
            nn.BatchNorm2d(cout),
        )

    def forward(self, x):
        return self.block(x) + self.res(x)


def Upsample(c):
    return nn.ConvTranspose2d(c, c, 2, 2)


class ResUnet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dim=32, conv_kernel_size=3):
        super().__init__()
        c = dim
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            snn.Leaky(beta=0.95),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=c, kernel_size=conv_kernel_size, padding=conv_kernel_size//2),
        )
        self.residual_conv_1 = ResBlock(c, 2*c, 2, conv_kernel_size)
        self.residual_conv_2 = ResBlock(2*c, 4*c, 2, conv_kernel_size)
        self.bridge = ResBlock(4*c, 8*c, 2, conv_kernel_size)
        self.upsample_1 = Upsample(8*c)
        self.up_residual_conv1 = ResBlock(8*c + 4*c, 4*c, 1, conv_kernel_size)
        self.upsample_2 = Upsample(4*c)
        self.up_residual_conv2 = ResBlock(4*c+2*c, 2*c, 1, conv_kernel_size)
        self.upsample_3 = Upsample(2*c)
        self.up_residual_conv3 = ResBlock(2*c + c, c, 1, conv_kernel_size)
        self.upsample_4 = Upsample(c)
        self.up_residual_conv4 = ResBlock(c, c, 1, conv_kernel_size)
        self.output_layer = nn.Sequential(
            nn.Conv2d(c, out_channels, 1, 1),
            # nn.Linear(),
        )

    def forward(self, x, time):
      mem = 0
      for time_step in range(time):
        # Encode
        xt = x[:,time_step,:,:]
        x1 = self.input_layer(xt) + self.input_skip(xt)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        # x11 = self.upsample_4(x10)
        # x12 = self.up_residual_conv4(x11)
        # x13 = TF.resize(x12, (1080, 1920), antialias=True)  # interpolation

        output = self.output_layer(x10)
        mem += output
      return mem / time
