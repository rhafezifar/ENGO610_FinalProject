import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

WEIGHTS_PATH = 'net_weights.pth'

batch_size = 16
test_data_percent = 20  # take ~20% for test
validation_data_percent = 15  # take ~15% for test

reverse_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0))])

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

data = torchvision.datasets.ImageFolder(root='../dataset3', transform=transform)
n = len(data)  # total number of examples
n_test = int(test_data_percent * n / 100.0)
n_valid = int(validation_data_percent * n / 100.0)
test_set, valid_set, train_set = torch.utils.data.random_split(data, [n_test, n_valid, n - n_valid - n_test], generator=torch.Generator().manual_seed(42))


train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

# classes = ('Cloud', 'Dust', 'Haze', 'Land', 'Seaside', 'Smoke')
classes = ('Non-Smoke', 'Smoke')

if __name__ == '__main__':
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # get some random training images
    dataiter = iter(test_loader)
    for images, labels in dataiter:
        # images, labels = dataiter.next()

        # show images
        # imshow(torchvision.utils.make_grid(images))
        # print labels
        print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))