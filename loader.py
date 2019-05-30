from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms


def get_loaders(root, batch_size, resolution, num_workers=32):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = ImageFolder(
        root + "/train",
        transforms.Compose([
            transforms.Resize([resolution, resolution]),
            transforms.RandomResizedCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )

    val_dataset = ImageFolder(
        root + "/val",
        transforms.Compose([
            transforms.Resize([resolution, resolution]),
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_loader = DataLoader(train_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(val_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader
