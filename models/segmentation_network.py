#mini u_net

# max params 1 000 000

import  torch.nn as nn
import torch

import torch.nn.functional as F

class segmentationNet(nn.Module):

    def __init__(self, input_channels, n_classes):
        super(segmentationNet, self).__init__()

        #self.hidden = 32  # nn.Linear(256,1) #None

        # Down 1
        #self.conv_1_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)  # None
        #self.relu_1_1 = nn.ReLU()  # None
        #self.conv_1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)  # None
        #self.relu_1_2 = nn.ReLU()  # None

        # Down 2
        #self.maxpool_2 = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)  # None
        #self.conv_2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # None
        #self.relu_2_1 = nn.ReLU()  # None
        #self.conv_2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)  # None
        #self.relu_2_2 = nn.ReLU()  # None

        # Down 3
        #self.maxpool_3 = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)  # None
        #self.conv_3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # None
        #self.relu_3_1 = nn.ReLU()  # None
        #self.conv_3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)  # None
        #self.relu_3_2 = nn.ReLU()  # None

        # Down 4
        #self.maxpool_4 = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)  # None
        #self.conv_4_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)  # None
        #self.relu_4_1 = nn.ReLU()  # None
        #self.conv_4_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)  # None
        #self.relu_4_2 = nn.ReLU()  # None

        # Down 5
        #self.maxpool_5 = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)  # None
        #self.conv_5_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)  # None
        #self.relu_5_1 = nn.ReLU()  # None
        #self.conv_5_2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)  # None
        #self.relu_5_2 = nn.ReLU()  # None

        # Up 6
        #self.upsample_6 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, padding=0,
        #                                     stride=2)  # None
        #self.conv_6_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=1)  # None
        #self.relu_6_1 = nn.ReLU()  # None
        #self.conv_6_2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=1)  # None
        #self.relu_6_2 = nn.ReLU()  # None

        # Up 7
        #self.upsample_7 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, padding=0,
        #                                     stride=2)  # None
        #self.conv_7_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=1)  # None
        #self.relu_7_1 = nn.ReLU()  # None
        #self.conv_7_2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1)  # None
        #self.relu_7_2 = nn.ReLU()  # None

        # Up 8
        #self.upsample_8 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, padding=0,
        #                                     stride=2)  # None
        #self.conv_8_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1)  # None
        #self.relu_8_1 = nn.ReLU()  # None
        #self.conv_8_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1)  # None
        #self.relu_8_2 = nn.ReLU()  # None

        # Up 9
        #self.upsample_9 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, padding=0,
        #                                     stride=2)  # None
        #self.conv_9_1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1)  # None
        #self.relu_9_1 = nn.ReLU()  # None
        #self.conv_9_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)  # None
        #self.relu_9_2 = nn.ReLU()  # None

        #self.output_conv = nn.Conv2d(self.hidden, n_classes, kernel_size=1)

        #self.sigmoid = nn.Sigmoid()
        self.hidden = 13

        # Down 1
        self.conv_1_1 = nn.Conv2d(in_channels=input_channels, out_channels=13, kernel_size=3, stride=1, padding=1)
        self.relu_1_1 = nn.ReLU()
        self.conv_1_2 = nn.Conv2d(in_channels=13, out_channels=13, kernel_size=3, stride=1, padding=1)
        self.relu_1_2 = nn.ReLU()

        # Down 2
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
        self.conv_2_1 = nn.Conv2d(in_channels=13, out_channels=26, kernel_size=3, stride=1, padding=1)
        self.relu_2_1 = nn.ReLU()
        self.conv_2_2 = nn.Conv2d(in_channels=26, out_channels=26, kernel_size=3, stride=1, padding=1)
        self.relu_2_2 = nn.ReLU()

        # Down 3
        self.maxpool_3 = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
        self.conv_3_1 = nn.Conv2d(in_channels=26, out_channels=52, kernel_size=3, stride=1, padding=1)
        self.relu_3_1 = nn.ReLU()
        self.conv_3_2 = nn.Conv2d(in_channels=52, out_channels=52, kernel_size=3, stride=1, padding=1)
        self.relu_3_2 = nn.ReLU()

        # Down 4
        self.maxpool_4 = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
        self.conv_4_1 = nn.Conv2d(in_channels=52, out_channels=104, kernel_size=3, stride=1, padding=1)
        self.relu_4_1 = nn.ReLU()
        self.conv_4_2 = nn.Conv2d(in_channels=104, out_channels=104, kernel_size=3, stride=1, padding=1)
        self.relu_4_2 = nn.ReLU()

        # Down 5
        self.maxpool_5 = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
        self.conv_5_1 = nn.Conv2d(in_channels=104, out_channels=208, kernel_size=3, stride=1, padding=1)
        self.relu_5_1 = nn.ReLU()
        self.conv_5_2 = nn.Conv2d(in_channels=208, out_channels=104, kernel_size=3, stride=1, padding=1)
        self.relu_5_2 = nn.ReLU()

        # Up 6
        self.upsample_6 = nn.ConvTranspose2d(in_channels=104, out_channels=104, kernel_size=2, padding=0, stride=2)
        self.conv_6_1 = nn.Conv2d(in_channels=208, out_channels=104, kernel_size=3, padding=1, stride=1)
        self.relu_6_1 = nn.ReLU()
        self.conv_6_2 = nn.Conv2d(in_channels=104, out_channels=52, kernel_size=3, padding=1, stride=1)
        self.relu_6_2 = nn.ReLU()

        # Up 7
        self.upsample_7 = nn.ConvTranspose2d(in_channels=52, out_channels=52, kernel_size=2, padding=0, stride=2)
        self.conv_7_1 = nn.Conv2d(in_channels=104, out_channels=52, kernel_size=3, padding=1, stride=1)
        self.relu_7_1 = nn.ReLU()
        self.conv_7_2 = nn.Conv2d(in_channels=52, out_channels=26, kernel_size=3, padding=1, stride=1)
        self.relu_7_2 = nn.ReLU()

        # Up 8
        self.upsample_8 = nn.ConvTranspose2d(in_channels=26, out_channels=26, kernel_size=2, padding=0, stride=2)
        self.conv_8_1 = nn.Conv2d(in_channels=52, out_channels=26, kernel_size=3, padding=1, stride=1)
        self.relu_8_1 = nn.ReLU()
        self.conv_8_2 = nn.Conv2d(in_channels=26, out_channels=13, kernel_size=3, padding=1, stride=1)
        self.relu_8_2 = nn.ReLU()

        # Up 9
        self.upsample_9 = nn.ConvTranspose2d(in_channels=13, out_channels=13, kernel_size=2, padding=0, stride=2)
        self.conv_9_1 = nn.Conv2d(in_channels=26, out_channels=13, kernel_size=3, padding=1, stride=1)
        self.relu_9_1 = nn.ReLU()
        self.conv_9_2 = nn.Conv2d(in_channels=13, out_channels=13, kernel_size=3, padding=1, stride=1)
        self.relu_9_2 = nn.ReLU()

        self.output_conv = nn.Conv2d(self.hidden, n_classes, kernel_size=1)
    def forward(self, x):
        #x1 = self.relu_1_1(self.conv_1_1(x))
        #x1 = self.relu_1_2(self.conv_1_2(x1))

        # Down 2
        # To do
        #x2 = self.maxpool_2(x1)
        #x2 = self.relu_2_1(self.conv_2_1(x2))
        #x2 = self.relu_2_2(self.conv_2_2(x2))

        # Down 3
        # To do
        #x3 = self.maxpool_3(x2)
        #x3 = self.relu_3_1(self.conv_3_1(x3))
        #x3 = self.relu_3_2(self.conv_3_2(x3))

        # Down 4
        # To do
        #x4 = self.maxpool_4(x3)
        #x4 = self.relu_4_1(self.conv_4_1(x4))
        #x4 = self.relu_4_2(self.conv_4_2(x4))

        # Down 5
        # To do
        #x5 = self.maxpool_5(x4)
        #x5 = self.relu_5_1(self.conv_5_1(x5))
        #x5 = self.relu_5_2(self.conv_5_2(x5))

        # Up 6
        # To do
        #u6 = self.upsample_6(x5)
        #u6 = torch.cat([u6, x4], dim=1)
        #u6 = self.relu_6_1(self.conv_6_1(u6))
        #u6 = self.relu_6_2(self.conv_6_2(u6))

        # Up 7
        # To do
        #u7 = self.upsample_7(u6)

        ###a retirer####
        #u7 = F.interpolate(
        #    u7,
        #    size=x3.shape[2:],
        #    mode="bilinear",
        #    align_corners=False
        #)
        #####

        #u7 = torch.cat([u7, x3], dim=1)
        #u7 = self.relu_7_1(self.conv_7_1(u7))
        #u7 = self.relu_7_2(self.conv_7_2(u7))

        # Up 8
        # To do
        #u8 = self.upsample_8(u7)
        #u8 = torch.cat([u8, x2], dim=1)
        #u8 = self.relu_8_1(self.conv_8_1(u8))
        #u8 = self.relu_8_2(self.conv_8_2(u8))

        # Up 9
        # To do
        #u9 = self.upsample_9(u8)
        # Adapter u9 aux dimensions de x1
        #u9 = F.interpolate(
        #    u9,
        #    size=x1.shape[2:],
        #    mode="bilinear",
        #    align_corners=False
        #)

        # Skip connection
        #u9 = torch.cat([u9, x1], dim=1)
        #u9 = self.relu_9_1(self.conv_9_1(u9))
        #u9 = self.relu_9_2(self.conv_9_2(u9))

        # Out
        #out = self.output_conv(u9)
        #out = self.sigmoid(out)

        # Down 1
        x1 = self.relu_1_1(self.conv_1_1(x))
        x1 = self.relu_1_2(self.conv_1_2(x1))

        # Down 2
        x2 = self.maxpool_2(x1)
        x2 = self.relu_2_1(self.conv_2_1(x2))
        x2 = self.relu_2_2(self.conv_2_2(x2))

        # Down 3
        x3 = self.maxpool_3(x2)
        x3 = self.relu_3_1(self.conv_3_1(x3))
        x3 = self.relu_3_2(self.conv_3_2(x3))

        # Down 4
        x4 = self.maxpool_4(x3)
        x4 = self.relu_4_1(self.conv_4_1(x4))
        x4 = self.relu_4_2(self.conv_4_2(x4))

        # Down 5
        x5 = self.maxpool_5(x4)
        x5 = self.relu_5_1(self.conv_5_1(x5))
        x5 = self.relu_5_2(self.conv_5_2(x5))

        # Up 6
        u6 = self.upsample_6(x5)
        u6 = F.interpolate(u6, size=x4.shape[2:], mode="bilinear", align_corners=False)
        u6 = torch.cat([u6, x4], dim=1)
        u6 = self.relu_6_1(self.conv_6_1(u6))
        u6 = self.relu_6_2(self.conv_6_2(u6))

        # Up 7
        u7 = self.upsample_7(u6)
        u7 = F.interpolate(u7, size=x3.shape[2:], mode="bilinear", align_corners=False)
        u7 = torch.cat([u7, x3], dim=1)
        u7 = self.relu_7_1(self.conv_7_1(u7))
        u7 = self.relu_7_2(self.conv_7_2(u7))

        # Up 8
        u8 = self.upsample_8(u7)
        u8 = F.interpolate(u8, size=x2.shape[2:], mode="bilinear", align_corners=False)
        u8 = torch.cat([u8, x2], dim=1)
        u8 = self.relu_8_1(self.conv_8_1(u8))
        u8 = self.relu_8_2(self.conv_8_2(u8))

        # Up 9
        u9 = self.upsample_9(u8)
        u9 = F.interpolate(u9, size=x1.shape[2:], mode="bilinear", align_corners=False)
        u9 = torch.cat([u9, x1], dim=1)
        u9 = self.relu_9_1(self.conv_9_1(u9))
        u9 = self.relu_9_2(self.conv_9_2(u9))

        # Out
        out = self.output_conv(u9)

        return out