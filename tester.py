import torch.optim as optim
import torch.nn as nn
import torch
from dataLoader import test_loader, WEIGHTS_PATH
from smokeNeXt import SmokeNeXt

net = SmokeNeXt()
net.load_state_dict(torch.load(WEIGHTS_PATH))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))