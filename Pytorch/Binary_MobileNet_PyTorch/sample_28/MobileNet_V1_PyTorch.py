import torch
import torch.nn as nn
from torchsummary import summary
from io import StringIO
import sys

import random
import numpy as np


# Set seed.
# seed = 180
# seed = 90
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)


class MobileNetV1(nn.Module):
    def __init__(self, ch_in, n_classes):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
                )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                # pw
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
                )

        self.model = nn.Sequential(
            conv_bn(ch_in, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1) # target output size of 1x1 (square)
        )
        self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model check
    # model = MobileNetV1(ch_in=3, n_classes=1000)
    model = MobileNetV1(ch_in=441, n_classes=1)
    # model = MobileNetV1(ch_in=1, n_classes=1)
    model.to(device)  # Move the model to the same device as the input data
    summary(model, input_size=(441, 128, 384), device=device)
    # summary(model, input_size=(1, 128, 384), device=device)

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    # Capture the printed summary
    string_buffer = StringIO()
    sys.stdout = string_buffer
    try:
        summary(model, input_size=(441, 128, 384), device=device)
    finally:
        sys.stdout = sys.__stdout__

    # Write the captured summary to a file
    summary_path = '/mnt/ceph/tco/TCO-Students/Homes/ISRAT/PyTorch_MobileNetV1_model_summary.txt'  # Replace with your desired path
    with open(summary_path, 'w') as f:
        f.write(string_buffer.getvalue())
