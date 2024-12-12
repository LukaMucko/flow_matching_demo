import torchvision.transforms as tfs
from torchvision.datasets import ImageFolder
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
from einops import rearrange

MEAN = torch.tensor([0.4188, 0.4207, 0.2801])
STD = torch.tensor([0.2712, 0.2463, 0.2515])


def loader(path):
    img = Image.open(path)
    return img.convert("RGB")


def show_images(batch_images, titles=None, reverse_dict=None):
    batch_size, C, H, W = batch_images.shape

    #batch_images = rearrange(STD, "c -> 1 c 1 1") * batch_images + rearrange(
    #    MEAN, "c -> 1 c 1 1"
    #)
    batch_images = batch_images.clamp(0, 1)

    fig, axes = plt.subplots(1, batch_size, figsize=(batch_size * 2, 2))

    for i in range(batch_size):
        img = batch_images[i].permute(1, 2, 0).numpy()
        if batch_size == 1:
            axes.imshow(img)
            axes.axis("off")
            if titles is not None:
                axes.set_title(reverse_dict[titles[i]], fontsize=12, loc="left")
        else:
            axes[i].imshow(img)
            axes[i].axis("off")
            if titles is not None:
                axes[i].set_title(reverse_dict[titles[i]], fontsize=12, loc="left")

    plt.tight_layout()
    plt.show()


class ImageDataset(Dataset):
    def __init__(self, dataset, image_size=256):
        self.dataset = dataset
        self.transform = tfs.Compose(
            [
                tfs.Resize((image_size, image_size)),
                tfs.ToTensor(),
                #tfs.Normalize(mean=MEAN, std=STD),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = self.transform(img)
        return img


def get_dataloader(image_folder = "animals10/", batch_size=32, num_workers=4, image_size=256):
    dataset = ImageFolder(root=image_folder, loader=loader)
    image_dataset = ImageDataset(dataset, image_size=image_size)

    class_dict = dataset.class_to_idx
    reverse_dict = {v: k for k, v in class_dict.items()}

    return (
        DataLoader(
            image_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        ),
        class_dict,
        reverse_dict,
    )
