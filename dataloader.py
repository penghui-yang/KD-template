import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


def CIFAR10(batch_sz, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    ])

    train_dataset = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=num_workers)
    train_loader.num_classes = 10

    test_dataset = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    eval_loader = DataLoader(test_dataset, batch_size=batch_sz, shuffle=True, num_workers=num_workers)
    eval_loader.num_classes = 10

    return train_loader, eval_loader
