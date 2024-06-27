import os
import pandas as pd
import torchvision.transforms as T

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100, CIFAR10
from torchvision.io import read_image
from PIL import Image


class FlickrSubset(Dataset):

    def __init__(self, label_path, img_path, transform, target_transform, adversarial, is_test_data=False):
        super().__init__()
        self.labels = pd.read_csv(label_path)
        self.img_dir = img_path
        if adversarial:
            self.getitem_func = self.getitem_adversarial
        elif is_test_data:
            self.getitem_func = self.getitem_withpath
        else:
            self.getitem_func = self.getitem
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tup = self.getitem_func(idx)
        return tup

    def getitem_adversarial(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, -1])
        image = read_image(img_path)
        label = self.labels.iloc[idx, -3]
        image = self.transform(image, label)
        label = self.target_transform(label)
        return image, label, img_path

    def getitem_withpath(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, -1])
        image = read_image(img_path)
        label = self.labels.iloc[idx, -3]
        image = self.transform(image)
        label = self.target_transform(label)
        return image, label, img_path

    def getitem(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, -1])
        image = read_image(img_path)
        label = self.labels.iloc[idx, -3]
        image = self.transform(image)
        label = self.target_transform(label)
        return image, label

class FlickrSubsetWithPath(Dataset):

    def __init__(self, label_path, img_path, transform, target_transform, adversarial, is_test_data=False):
        super().__init__()
        self.labels = pd.read_csv(label_path)
        self.img_dir = img_path
        if adversarial:
            self.getitem_func = self.getitem_adversarial
        elif is_test_data:
            self.getitem_func = self.getitem_withpath
        else:
            self.getitem_func = self.getitem
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image, label, img_path = self.getitem_func(idx)
        return image, label, img_path

    def getitem_adversarial(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, -1])
        image = read_image(img_path)
        label = self.labels.iloc[idx, -3]
        image = self.transform(image, label)
        label = self.target_transform(label)
        return image, label, img_path

    def getitem(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, -1])
        image = read_image(img_path)
        label = self.labels.iloc[idx, -3]
        image = self.transform(image)
        label = self.target_transform(label)
        return image, label, img_path


class AugmentedFlickrSubset(FlickrSubset):

    def __init__(self, *args, **kwargs):
        # Augmentations as in https://arxiv.org/pdf/1912.11035.pdf
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, -1])
        image = read_image(img_path)
        label = self.labels.iloc[idx, -3]
        image = self.transform(image)
        label = self.target_transform(label)
        return image, label


class Nips17Subset(Dataset):
    
    def __init__(self, img_path, label_path, transform, target_transform, adversarial, is_test_data=False):
        super().__init__()
        self.labels = pd.read_csv(label_path)
        self.img_dir = img_path
        if adversarial:
            self.getitem_func = self.getitem_adversarial
        elif is_test_data:
            self.getitem_func = self.getitem_withpath
        else:
            self.getitem_func = self.getitem
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tup = self.getitem_func(idx)
        return tup

    def getitem_adversarial(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = read_image(img_path + '.png')
        label = self.labels.iloc[idx, 6] - 1
        image = self.transform(image, label)
        label = self.target_transform(label)
        return image, label, img_path

    def getitem_withpath(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = read_image(img_path + '.png')
        label = self.labels.iloc[idx, 6] - 1
        image = self.transform(image)
        label = self.target_transform(label)
        return image, label, img_path
    
    def getitem(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = read_image(img_path + '.png')
        label = self.labels.iloc[idx, 6] - 1
        image = self.transform(image, label)
        label = self.target_transform(label)
        return image, label

class CustomCIFAR10(CIFAR10):
    
    #helps us to overwrite CIFAR10 obj for our purposes
    
    def __init__(self, adversarial, is_test_data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pil_to_tensor = T.PILToTensor()
        if adversarial:
            self.getitem_func = self.getitem_adversarial
        else:
            self.getitem_func = self.getitem

    def __getitem__(self, index: int):
        img, target, path = self.getitem_func(index)
        return img, target, path
    
    def getitem_adversarial(self, index: int):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.pil_to_tensor(img)

        if self.transform is not None:
            img = self.transform(img, target)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, 'nopathgiven' # to make this work with our version

    def getitem(self, index: int):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        img = self.pil_to_tensor(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, 'nopathgiven' # to make this work with our version

class CustomCIFAR100(CIFAR100):
    
    #helps us to overwrite CIFAR10 obj for our purposes
    
    def __init__(self, adversarial, is_test_data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pil_to_tensor = T.PILToTensor()
        if adversarial:
            self.getitem_func = self.getitem_adversarial
        else:
            self.getitem_func = self.getitem

    def __getitem__(self, index: int):
        img, target, path = self.getitem_func(index)
        return img, target, path
    
    def getitem_adversarial(self, index: int):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.pil_to_tensor(img)

        if self.transform is not None:
            img = self.transform(img, target)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, 'nopathgiven' # to make this work with our version

    def getitem(self, index: int):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        img = self.pil_to_tensor(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, 'nopathgiven' # to make this work with our version
        
    


if __name__ == '__main__':
    pass