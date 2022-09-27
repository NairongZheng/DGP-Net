import os

import numpy
import torch
import torch.nn as nn
import torchvision as tv

from PIL import Image
from sklearn.manifold import TSNE

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CifarSEBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, reduction=16):
        super(CifarSEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out += residual
        out = self.relu(out)

        return out


class SEResNet(nn.Module):
    def __init__(self, block=CifarSEBasicBlock, n_size=6, reduction=16):  # n_size=3
        super(SEResNet, self).__init__()
        self.inplane = 16
        self.conv1 = nn.Conv2d(
            1, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            block, 16, blocks=n_size, stride=1, reduction=reduction)
        self.layer2 = self._make_layer(
            block, 32, blocks=n_size, stride=2, reduction=reduction)
        self.layer3 = self._make_layer(
            block, 64, blocks=n_size, stride=2, reduction=reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(64, num_classes)
        # self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, reduction):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride, reduction))
            self.inplane = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x
    # def load(self, file_name):
    #     self.load_state_dict(torch.load(file_name, map_location=lambda storage, loc: storage))
    def save(self, file_name):
        torch.save(self.state_dict(), file_name)
if __name__ == '__main__':
    model_dir = os.path.join('model', 'pretrain_model.pth')
    model = SEResNet()
    # model =torch.load(model_dir)
    model.load_state_dict(torch.load(model_dir))
    img_path = os.path.join('data1', '6')
    transform_valid = tv.transforms.Compose([
        tv.transforms.Grayscale(1),  # 单通道
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(
            (0.1307,), (0.3081,))
    ]
    )
    imgg = Image.open(os.path.join('data1', '0', 'HB14931.JPG'))
    imgg_ = transform_valid(imgg).unsqueeze(0)  # 拓展维度
    outt = model(imgg_)
    for file in os.listdir(img_path):
        # 判断是否是文件
        if os.path.isfile(os.path.join(img_path, file)) == True:
            img = Image.open(os.path.join(img_path, file))
            img_ = transform_valid(img).unsqueeze(0)  # 拓展维度
            out = model(img_)
            outt = torch.cat((outt, out), 0)

    with open("cnn_dist.csv",'ab') as f:
        numpy.savetxt(f, outt.detach().numpy(), delimiter=',')
    # print(outt.size())

    #
    # tsne.fit_transform(out.detach().numpy())
    # print(tsne.embedding_)



