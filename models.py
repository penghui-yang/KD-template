import torch.nn as nn
from torchvision import models


class LeNet(nn.Module):

    def __init__(self, out_dim=10, in_channel=1, img_sz=28):
        super(LeNet, self).__init__()
        feat_map_sz = img_sz//4
        self.n_feat = 50 * feat_map_sz * feat_map_sz
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 20, 5, padding=2),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 50, 5, padding=2),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.linear = nn.Sequential(
            nn.Linear(self.n_feat, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(inplace=True),
        )
        self.last = nn.Linear(500, out_dim)

    def features(self, x):
        x = self.conv(x)
        x = self.linear(x.view(-1, self.n_feat))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_feature, out_feature, cell_count):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(in_feature, cell_count)
        self.layer1_activation = nn.ReLU()
        self.layer2 = nn.Linear(cell_count, cell_count)
        self.layer2_activation = nn.ReLU()
        self.layer3 = nn.Linear(cell_count, cell_count)
        self.layer3_activation = nn.ReLU()
        self.layer4 = nn.Linear(cell_count, out_feature)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer1_activation(out)
        out = self.layer2(out)
        out = self.layer2_activation(out)
        out = self.layer3(out)
        out = self.layer3_activation(out)
        out = self.layer4(out)
        return out


def mlp(num_classes):
    net = MLP(3*224*224, num_classes, 300)
    return net


def lenet(num_classes):
    net = LeNet(num_classes, 3, 224)
    return net


def alexnet(num_classes):
    net = models.alexnet(pretrained=False)
    net.classifier[6] = nn.Linear(4096, num_classes)
    return net


def vgg11(num_classes):
    net = models.vgg11(pretrained=False)
    net.classifier[6] = nn.Linear(4096, num_classes)
    return net


def resnet50(num_classes):
    net = models.resnet50(pretrained=False)
    num_features = net.fc.in_features
    net.fc = nn.Linear(num_features, num_classes)
    return net
