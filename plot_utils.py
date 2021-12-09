import numpy as np
from matplotlib import pyplot as plt

from dataLoader import reverse_transform, classes, batch_size


def plot_loss(train_loss, valid_loss):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 0.5)  # consistent scale
    plt.xlim(0, len(train_loss) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # fig.savefig('loss_plot.png', bbox_inches='tight')


def plot_output(images, labels, predictions):
    # prep images for display
    images = images.numpy()

    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(4, 4))
    for idx in np.arange(batch_size):
        ax = fig.add_subplot(4, 4, idx + 1, xticks=[], yticks=[])
        images[idx] = reverse_transform(np.transpose(images[idx]))
        ax.imshow(np.transpose(images[idx]))
        ax.set_title("{} ({})".format(str(classes[predictions[idx].item()]), str(classes[labels[idx].item()])), color=("green" if predictions[idx] == labels[idx] else "red"))

    plt.show()