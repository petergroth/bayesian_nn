import torch
import torch.utils.data

from torchvision import datasets, transforms

def mnist():
    # Create transform object to convert data to normalised tensors
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])

    # Download train data
    train = datasets.MNIST('data/processed/MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

    # Download and load the test data
    test = datasets.MNIST('data/processed/MNIST_data/', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=False)

    return trainloader, testloader


