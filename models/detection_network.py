import torch
import torch.nn as nn


# ##400 000 params

# #implementer un yolo custom


class detectionNet(nn.Module):

    def __init__(self):
        super(detectionNet, self).__init__()

        ### Down scale
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=5, stride=1,
                                padding=2)
        #             nn.Conv2d(in_channels=1, out_channels=24, kernel_size=5, stride=2, padding=2)
        self.Lrelu1 = nn.LeakyReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_2 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1)
        self.Lrelu2 = nn.LeakyReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_3 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.Lrelu3 = nn.LeakyReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_4 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.Lrelu4 = nn.LeakyReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.Lrelu5 = nn.LeakyReLU()
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        ### Detection
        self.conv_6 = nn.Conv2d(in_channels=128, out_channels=72, kernel_size=3, stride=1, padding=1)
        self.Lrelu6 = nn.LeakyReLU()

        # self.conv_7 = nn.Conv2d(in_channels=72, out_channels=8, kernel_size=1, stride=1, padding=0)
        # nn.Conv2d(in_channels=72, out_channels=13, kernel_size=1, stride=1, padding=0)

        # self.grid = nn.AdaptiveAvgPool2d((7, 7))
        self.conv_7 = nn.Conv2d(in_channels=72, out_channels=21, kernel_size=1, stride=1, padding=0)

        self.grid = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x1 = self.conv_1(x)
        r1 = self.Lrelu1(x1)
        max1 = self.maxpool1(r1)

        x2 = self.conv_2(max1)
        r2 = self.Lrelu2(x2)
        max2 = self.maxpool2(r2)

        x3 = self.conv_3(max2)
        r3 = self.Lrelu3(x3)
        # max3 = self.maxpool3(r3)

        x4 = self.conv_4(r3)
        r4 = self.Lrelu4(x4)
        # max4 = self.maxpool4(r4)

        x5 = self.conv_5(r4)
        r5 = self.Lrelu5(x5)
        # max5 = self.maxpool5(r5)

        x6 = self.conv_6(r5)
        r6 = self.Lrelu6(x6)

        # ####a modifier

        x7 = self.conv_7(r6)

        x7 = self.grid(x7)

        output = x7.reshape(x7.shape[0], 3, 7)

        ### Prédictions
        confidence = torch.sigmoid(output[..., 0:1])
        boxes = torch.sigmoid(output[..., 1:4])
        classes = torch.sigmoid(output[..., 4:7])

        output = torch.cat([
            confidence,
            boxes,
            classes
        ], dim=-1)

        return output


## custom loss
# source https://medium.com/@mustafa.kamalzhd/writing-a-custom-loss-in-pytorch-part-1-7dd857934f48

##loss de yolo
# https://medium.com/@whyamit404/how-to-implement-a-yolo-object-detector-from-scratch-in-pytorch-e310829d92e6
class lossYolo(nn.Module):

    def __init__(self):
        super(lossYolo, self).__init__()

    def forward(self, output, target):
        target_yolo = torch.zeros_like(output)

        N = output.shape[0]

        for i in range(N):
            for j in range(target.shape[1]):

                if target[i, j, 0] == 1:
                    classe = int(target[i, j, 4].item())

                    target_yolo[i, classe, 0] = 1
                    target_yolo[i, classe, 1] = target[i, j, 1]
                    target_yolo[i, classe, 2] = target[i, j, 2]
                    target_yolo[i, classe, 3] = target[i, j, 3]
                    target_yolo[i, classe, 4 + classe] = 1

        #############################################################################

        pred_conf = output[..., 0]
        pred_boxes = output[..., 1:4]
        pred_classes = output[..., 4:7]

        target_conf = target_yolo[..., 0]
        target_boxes = target_yolo[..., 1:4]
        target_classes = target_yolo[..., 4:7]

        object_mask = target_conf > 0
        noobject_mask = target_conf == 0

        # Localization Loss
        Lxywh = torch.sum((pred_boxes[object_mask] - target_boxes[object_mask]) ** 2)

        # Confidence Loss
        Lco = torch.sum((pred_conf[object_mask] - target_conf[object_mask]) ** 2)
        Lcso = torch.sum(pred_conf[noobject_mask] ** 2)

        # Classification Loss
        # Lclasse = torch.sum((pred_classes[object_mask] - target_classes[object_mask]) ** 2)
        ###suggestion meilleur
        Lclasse = nn.functional.binary_cross_entropy(
            pred_classes[object_mask],
            target_classes[object_mask],
            reduction='sum'
        )

        Lyolo = 5 * Lxywh + Lco + 3 * Lcso + Lclasse

        return Lyolo
