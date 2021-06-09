from torchvision import datasets, transforms

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    # Create transform object to convert data to normalised tensors
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])

    # Download train data
    datasets.MNIST('data/processed/MNIST_data/', download=True, train=True, transform=transform)

    # Download and load the test data
    datasets.MNIST('data/processed/MNIST_data/', download=True, train=False, transform=transform)


if __name__ == '__main__':
    main()
