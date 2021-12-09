import torch
from dataLoader import test_loader, WEIGHTS_PATH
from resnet.ResNet import *

net = resnet18(3, 2)
net.load_state_dict(torch.load(WEIGHTS_PATH))
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(gpu)
net.to(gpu)

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        # images, labels = data
        images, labels = data[0].to(gpu), data[1].to(gpu)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %.2f %%' % (100.0 * correct / total))