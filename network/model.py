import torch
from torchvision.models import vgg16, resnet50, densenet121, mobilenet_v2
from torch.nn import functional as F
import torch.nn as nn
import torchvision.models as models

class OCRDenseNet(nn.Module):
    def __init__(self, num_letters, finetune=False, lr=1e-3, weight_decay=0.001):
        super(OCRDenseNet, self).__init__()
        self.backbone = densenet121(weights='DEFAULT')

        for param in self.backbone.parameters():
            param.requires_grad = finetune

        self.fc1 = nn.Linear(5120, 512)
        self.bi_lstm = nn.LSTM(512, 512, batch_first=True, bidirectional=True)
        self.fc3 = nn.Linear(1024, num_letters)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        self.loss_fn = nn.CTCLoss(blank=0)

        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x, target=None, target_length=None):
        x = self.backbone.features(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.relu(self.fc1(x))
        x, _ = self.bi_lstm(x)

        if target is not None and target_length is not None:
            x = self.log_softmax(self.fc3(x))
            x = x.permute(1, 0, 2)
            input_length = torch.full(size=(x.size(1),), fill_value=x.size(0), dtype=torch.long)
            loss = self.loss_fn(x, target, input_length, target_length)
            return x, loss

        return self.softmax(self.fc3(x)), None


class OCRMobile(nn.Module):
    def __init__(self, num_letters, finetune=False, lr=1e-3, weight_decay=0.001):
        super(OCRMobile, self).__init__()
        self.backbone = mobilenet_v2(weights='DEFAULT')

        for param in self.backbone.parameters():
            param.requires_grad = finetune

        self.fc1 = nn.Linear(6400, 512)
        self.bi_lstm = nn.LSTM(512, 512, batch_first=True, bidirectional=True)
        self.fc3 = nn.Linear(1024, num_letters)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        self.loss_fn = nn.CTCLoss(blank=0)

        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x, target=None, target_length=None):
        x = self.backbone.features(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.relu(self.fc1(x))
        x, _ = self.bi_lstm(x)

        if target is not None and target_length is not None:
            x = self.log_softmax(self.fc3(x))
            x = x.permute(1, 0, 2)
            input_length = torch.full(size=(x.size(1),), fill_value=x.size(0), dtype=torch.long)
            loss = self.loss_fn(x, target, input_length, target_length)
            return x, loss

        return self.softmax(self.fc3(x)), None


class VietOCRResNet50(nn.Module):
    def __init__(self, num_letters, finetune=False):
        super(VietOCRResNet50, self).__init__()
        self.backbone = resnet50(pretrained=True)

        for param in self.backbone.parameters():
            param.requires_grad = finetune

        self.fc1 = nn.Linear(2048, 512)
        self.bi_lstm = nn.LSTM(512, 512, batch_first=True, bidirectional=True)
        self.fc3 = nn.Linear(1024, num_letters)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        self.loss_fn = nn.CTCLoss(blank=0)

    def forward(self, x, target=None, target_length=None):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x, _ = self.bi_lstm(x)

        if target is not None and target_length is not None:
            x = self.fc3(x)
            input_length = torch.full(size=(x.size(0),), fill_value=x.size(1), dtype=torch.long)
            loss = self.loss_fn(x, target, input_length, target_length)
            return x, loss

        return self.softmax(self.fc3(x)), None


class VietOCRVGG16(nn.Module):
    def __init__(self, num_letters, finetune=False):
        super(VietOCRVGG16, self).__init__()
        self.backbone = vgg16(weights = 'DEFAULT')

       
        for param in self.backbone.parameters():
            param.requires_grad = finetune

        self.fc1 = nn.Linear(2560, 512)
        self.bi_lstm = nn.LSTM(512, 512, batch_first=True, bidirectional=True)
        self.fc3 = nn.Linear(1024, num_letters)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        self.loss_fn = nn.CTCLoss(blank=0)

    def forward(self, x, target=None, target_length=None):
        x = self.backbone.features(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.relu(self.fc1(x))
        x, _ = self.bi_lstm(x)

        if target != None and target_length != None:
            x = self.log_softmax(self.fc3(x))
            x = x.permute(1, 0, 2)
            input_length = torch.full(size=(x.size(1),), fill_value=x.size(0), dtype=torch.long)
            loss = self.loss_fn(x, target, input_length, target_length)
            return x, loss

        return self.softmax(self.fc3(x)), None