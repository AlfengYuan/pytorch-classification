import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

class AlexNet(nn.Module):
    """AlexNet model architecture from the
        origin paper: <http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>
       `"One weird trick..." <https://arxiv.org/abs/1202.2745>`_ paper.
    """
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

 
def alexnet(pretrained=False, checkpoints=None, progress=True, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        if checkpoints is not None:
            model.load_state_dict(torch.load(checkpoints), True)
            return model
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth")
        model.load_state_dict(state_dict)
        return model
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)
    return model

