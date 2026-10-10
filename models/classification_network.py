import torch.nn as nn

###je dois faire un alex net custom

##max 200 000 params

#intrant 53 * 53
#niveau de gris
#3 classe

class classificationNet(nn.Module):

    def __init__(self):

        super(classificationNet,self).__init__()

        ###a down scale
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flat = nn.Flatten()

        self.lin1 = nn.Linear(256, 256)
        self.reluLin1 = nn.ReLU()
        self.lin2 = nn.Linear(256, 128)
        self.reluLin2 = nn.ReLU()
        self.lin3 = nn.Linear(128, 3)
        #self.soft = nn.Softmax()


    def forward(self, x):

        x1 = self.conv_1(x)
        r1 = self.relu1(x1)
        max1 = self.maxpool1(r1)
        x2 = self.conv_2(max1)
        r2 = self.relu2(x2)
        max2 = self.maxpool2(r2)
        x3 = self.conv_3(max2)
        r3 = self.relu3(x3)
        max3 = self.maxpool3(r3)

        flat = self.flat(max3)

        lin1 = self.lin1(flat)
        rlin1 = self.reluLin1(lin1)
        lin2 = self.lin2(rlin1)
        rlin2 = self.reluLin2(lin2)
        lin3 = self.lin3(rlin2)
        #soft = self.soft(lin3) ## multi sortie

        return lin3
