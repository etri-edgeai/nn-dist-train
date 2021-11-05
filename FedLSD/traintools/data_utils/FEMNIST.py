import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os, json

__all__ = ["FEMNIST"]


class FEMNIST(Dataset):
    """
    Data Download link:
    >>> https://www.dropbox.com/s/ri3riws7us84ibw/fedlsd_femnist.zip?dl=0
    """

    def __init__(self, root, transform=None, split="train"):
        self.root, self.split = root, split
        self.transforms = {
            "train": transforms.Compose(
                [
                    transforms.RandomCrop(28, 3),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize(0.9618, 0.1644),
                ]
            ),
            "test": transforms.Compose([transforms.Normalize(0.9618, 0.1644)]),
        }

        self.images, self.labels = [], []

        with open(os.path.join(root, split + ".json")) as f:
            self.data_dict = json.load(f)

        if self.split == "train":
            self.num_samples = self.data_dict["num_samples"]

            for i in range(100):
                user_key = self.data_dict["users"][i]
                self.images += self.data_dict["user_data"][user_key]["x"]
                self.labels += self.data_dict["user_data"][user_key]["y"]

        else:
            for values in self.data_dict["user_data"].values():
                for i in range(len(values["y"])):
                    self.images.append(values["x"][i])
                    self.labels.append(values["y"][i])

        self.labels = torch.LongTensor(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.FloatTensor(self.images[idx]).view(1, 28, 28)
        image = self.transforms[self.split](image)
        label = self.labels[idx]

        return image, label
