import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

WEIGHTS_PATH = 'net_weights/smoke_next.pth'

batch_size = 16
test_data_percent = 20

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

data = torchvision.datasets.ImageFolder(root='dataset2', transform=transform)
n = len(data)  # total number of examples
n_test = int(test_data_percent * n / 100.0)  # take ~10% for test
test_set, train_set = torch.utils.data.random_split(data, [n_test, n - n_test], generator=torch.Generator().manual_seed(42))


train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
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