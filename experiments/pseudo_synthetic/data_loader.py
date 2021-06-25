import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def data_loader(data):
    if data == 'mnist':
        transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])
        trainset = datasets.MNIST('./data', train=True, download=True,
                            transform=transform)
        testset = datasets.MNIST('./data', train=False,
                            transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    elif data == 'cifar10':
        transform_train = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])

        transform_test = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])
        trainset = torchvision.datasets.CIFAR10(
            root='./cifar10', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=1, shuffle=False)

        testset = torchvision.datasets.CIFAR10(
            root='./cifar10', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=1, shuffle=False)

    elif data == 'cifar100':
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
        train_dataset = torchvision.datasets.CIFAR100(
                    root='./cifar100', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,]))
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)

        test_dataset = torchvision.datasets.CIFAR100(
                    root='./cifar100', train=False, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        normalize,]))
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    return trainloader, testloader
