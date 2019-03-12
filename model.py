import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43  # GTSRB as 43 classes


class Net(nn.Module):
    def __init__(self, gray=True):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1 if gray else 3, 100, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(100, 150, 3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(150)
        self.conv3 = nn.Conv2d(150, 250, 2, padding=1)
        self.fc1 = nn.Linear(250 * 4 * 4, 300)
        self.fc1_bn = nn.BatchNorm1d(300)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(300, 43)

    def forward(self, x):
        x = self.pool(F.elu(self.conv1(x)))
        x = self.pool(F.elu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.elu(self.conv3(x)))
        x = x.view(-1, 250 * 4 * 4)
        x = F.elu(self.fc1_bn(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class StnNet(nn.Module):
    def __init__(self, gray=True):
        super(StnNet, self).__init__()
        input_chan = 1 if gray else 3
        self.conv1 = nn.Conv2d(input_chan, 100, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(100, 150, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(150)
        self.conv3 = nn.Conv2d(150, 250, kernel_size=2, padding=1)
        self.fc1 = nn.Linear(250 * 4 * 4, 300)
        self.fc1_bn = nn.BatchNorm1d(300)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(300, nclasses)

        # Spatial transformer localization-network
        self.loc_net = nn.Sequential(
            nn.Conv2d(input_chan, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor(
            [1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.loc_net(x)
        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        x = self.stn(x)
        x = self.pool(F.elu(self.conv1(x)))
        x = self.pool(F.elu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.elu(self.conv3(x)))
        x = x.view(-1, 250 * 4 * 4)
        x = F.elu(self.fc1_bn(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
