import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms



cifar10_NAME = 'cifar10'
cifar10_MEAN = [0.49139968, 0.48215827, 0.44653124]
cifar10_STD = [0.24703233, 0.24348505, 0.26158768]

mnist_NAME = 'mnist'
mnist_MEAN = [0.13066051707548254]
mnist_STD = [0.30810780244715075]

fashionmnist_NAME = 'fashionmnist'
fashionmnist_MEAN = [0.28604063146254594]
fashionmnist_STD = [0.35302426207299326]

DATASET = {
    cifar10_NAME:       {'mean':[0.49139968, 0.48215827, 0.44653124],
                         'std':[0.24703233, 0.24348505, 0.26158768],
                         'transforms': [transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip()]},

    mnist_NAME:         {'mean':[0.13066051707548254],
                         'std':[0.30810780244715075],
                         'transforms':[transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1)]},

    fashionmnist_NAME:  {'mean':[0.28604063146254594],
                         'std':[0.35302426207299326],
                         'transforms':[transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
                                        transforms.RandomVerticalFlip()]}

}

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):

        try:
            h, w = img.size(1), img.size(2)
        except TypeError:
            h, w = img.shape[1], img.shape[2]

        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(np.array(mask))
        mask = mask.expand_as(torch.tensor(img))
        img = torch.tensor(img).float()*mask

        return img

def data_transforms(dataset, cutout_length):
    dataset = dataset.lower()

    if not dataset in DATASET.keys():
        raise ValueError('not expected dataset = {}'.format(dataset))

    MEAN, STD, transf = None, None, None
    for dataset_name in DATASET.keys():
        # Search for the dataset information
        if dataset is dataset_name:
            MEAN = DATASET[dataset_name]['mean']
            STD = DATASET[dataset_name]['std']
            transf = DATASET[dataset_name]['transforms']
            break

    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]

    train_transform = transforms.Compose(transf + normalize)
    valid_transform = transforms.Compose(normalize)

    if cutout_length > 0:
        train_transform.transforms.append(Cutout(cutout_length))

    return train_transform, valid_transform

# Test the above functions
import matplotlib.pyplot as plt

if __name__ == '__main__':


    # Test Cutout class
    img = np.random.random([265, 265, 3])

    cutout = Cutout(3)
    img = cutout(img)

    plt.imshow(img)
    plt.show()