import torch 
import torch.nn as nn
import numpy as np

class DAB_S(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_classes=10):
        super(DAB_S, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, dilation=1, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.InstanceNorm2d(in_channels, affine=True),
            nn.LeakyReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, dilation=2, kernel_size=(3,3), padding=(2,2)),
            nn.InstanceNorm2d(in_channels, affine=True),
            nn.LeakyReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, dilation=3, kernel_size=(3,3), stride=(1,1), padding=(3,3)),
            nn.InstanceNorm2d(in_channels, affine=True),
            nn.LeakyReLU()
        )
        self.layer5 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        )
        self.layer6 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=3*2*in_channels, out_channels=in_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.InstanceNorm2d(in_channels, affine=True),
            nn.LeakyReLU()
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        )
        self.layer9 = nn.Sequential(
            nn.Softmax(dim=1)
        )
        self.layer10 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU()
        )
        
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = torch.cat([x2, x3, x4], axis=1)
        x6 = self.layer5(x5)
        x7 = self.layer6(x5)
        x8 = torch.cat([x6, x7], axis=1)
        x9 = self.layer7(x8)
        x10 = self.layer8(x9)
        x11 = self.layer9(x10)
        x12 = self.layer10(x11 * x)
        return x12
    
class DA_SNet(nn.Module):
    def __init__(self):
        super(DAB_SNet, self).__init__()
        n = 8
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=n, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = DAB_S(n, n)
        self.maxpool1 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = DAB_S(n, 2*n)
        self.layer3 = DAB_S(2*n, 2*n)
        self.maxpool2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = DAB_S(2*n, 4*n)
        self.layer5 = DAB_S(4*n, 4*n)
        self.maxpool3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer6 = DAB_S(4*n, 8*n)
        self.layer7 = DAB_S(8*n, 8*n)
        self.maxpool4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer8 = DAB_S(8*n, 16*n)
        self.layer9 = DAB_S(16*n, 16*n)
        self.layer10 = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=2)
        )
        self.layer11 = nn.Flatten()
        self.fc1 = nn.Linear(16*n, 16*n//2)
        self.layer12 = nn.Sequential(nn.ReLU())
        self.fc2 = nn.Linear(16*n//2, 16*n//4)
        self.layer13 = nn.Sequential(
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x1 = self.layer0(x)
        x2 = self.layer1(x1)
        x3 = self.maxpool1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x6 = self.maxpool2(x5)
        x7 = self.layer4(x6)
        x8 = self.layer5(x7)
        x9 = self.maxpool3(x8)
        x10 = self.layer6(x9)
        x11 = self.layer7(x10)
        x13 = self.maxpool4(x11)
        x14 = self.layer8(x13)
        x15 = self.layer9(x14)
        x16 = self.layer10(x15)
        x17 = self.layer11(x16)
        x18 = self.fc1(x17)
        x19 = self.layer12(x18)
        x20 = self.fc2(x19)
        x21 = self.layer13(x20)

        return x21






class DAB_H(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_classes=10):
        super(DAB_H, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, dilation=1, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.LazyBatchNorm2d(),
            nn.PReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, dilation=2, kernel_size=(3,3), padding=(2,2)),
            nn.LazyBatchNorm2d(),
            nn.PReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, dilation=3, kernel_size=(3,3), stride=(1,1), padding=(3,3)),
            nn.LazyBatchNorm2d(),
            nn.PReLU()
        )
        self.layer5 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        )
        self.layer6 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=3*2*in_channels, out_channels=in_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.LazyBatchNorm2d(),
            nn.PReLU()
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        )
        self.layer9 = nn.Sequential(
            nn.Softmax(dim=1)
        )
        self.layer10 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.LazyBatchNorm2d(),
            nn.PReLU()
        )
        
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = torch.cat([x2, x3, x4], axis=1)
        x6 = self.layer5(x5)
        x7 = self.layer6(x5)
        x8 = torch.cat([x6, x7], axis=1)
        x9 = self.layer7(x8)
        x10 = self.layer8(x9)
        x11 = self.layer9(x10)
        x12 = self.layer10(x)
        return x12 * x11

class DA_HNet(nn.Module):
    def __init__(self):
        super(DAB_HNet, self).__init__()
        n = 8
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=n, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = DAB_H(n, n)
        self.maxpool1 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = DAB_H(n, 2*n)
        self.layer3 = DAB_H(2*n, 2*n)
        self.maxpool2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = DAB_H(2*n, 4*n)
        self.layer5 = DAB_H(4*n, 4*n)
        self.maxpool3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer6 = DAB_H(4*n, 8*n)
        self.layer7 = DAB_H(8*n, 8*n)
        self.maxpool4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer8 = DAB_H(8*n, 16*n)
        self.layer9 = DAB_H(16*n, 16*n)
        self.layer10 = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=2)
        )
        self.layer11 = nn.Flatten()
        self.fc1 = nn.Linear(16*n, 16*n//4)
        self.layer12 = nn.Sequential(nn.PReLU())
        self.fc2 = nn.Linear(16*n//4, 16*n//16)
        self.layer13 = nn.Sequential(
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x1 = self.layer0(x)
        x2 = self.layer1(x1)
        x3 = self.maxpool1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x6 = self.maxpool2(x5)
        x7 = self.layer4(x6)
        x8 = self.layer5(x7)
        x9 = self.maxpool3(x8)
        x10 = self.layer6(x9)
        x11 = self.layer7(x10)
        x13 = self.maxpool4(x11)
        x14 = self.layer8(x13)
        x15 = self.layer9(x14)
        x16 = self.layer10(x15)
        x17 = self.layer11(x16)
        x18 = self.fc1(x17)
        x19 = self.layer12(x18)
        x20 = self.fc2(x19)
        x21 = self.layer13(x20)

        return x21
