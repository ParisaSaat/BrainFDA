import os

import nibabel as nib
import torch
from torch.utils.data import Dataset


class VS(Dataset):
    def __init__(self, img_dir, mask_dir=None, transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform
        self.images_list = os.listdir(img_dir)
        self.masks_list = [] if mask_dir is None else os.listdir(mask_dir)
        self.images = []
        self.masks = []

        for img in sorted(self.images_list):
            image = nib.load(os.path.join(img_dir, img)).get_data()
            self.images.append(image)

        if mask_dir is not None:
            for msk in sorted(self.masks_list):
                mask = nib.load(os.path.join(mask_dir, msk)).get_data()
                self.masks.append(mask)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[:, :, idx] if self.masks else None

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        sample = {'image': torch.tensor(image), 'mask': torch.tensor(mask)}
        return sample
