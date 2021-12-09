from dataLoader import *
import torch.nn as nn
import torch.nn.functional as F

from plot_utils import plot_loss, plot_output
from test_model import test_model
from train_model import train_model


class SmokeNeXt(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5))
        self.fc1 = nn.Linear(in_features=16 * 61 * 61, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=len(classes))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    # from torchsummary import summary

    model = SmokeNeXt()
    # summary(model.cuda(), (3, 256, 256))

    # early stopping patience; how long to wait after last time validation loss improved.
    patience = 20

    model, train_loss, valid_loss = train_model(model, patience, 100)
    if train_loss and valid_loss:
        plot_loss(train_loss, valid_loss)

    ###############################################################################################################################
    test_model(model)

    # obtain one batch of test images
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    images = images.to(gpu)
    labels = labels.to(gpu)
    # get sample outputs
    output = model(images)
    images = images.cpu()
    # convert output probabilities to predicted class
    _, preds = torch.max(output, 1)

    plot_output(images, labels, preds)
