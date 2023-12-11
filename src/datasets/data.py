import csv

from torch.utils.data import DataLoader
from src.datasets.subsets import FlickrSubset, FlickrSubsetWithPath, AugmentedFlickrSubset, Nips17Subset
from src.datasets.data_transforms.img_transform import IMGTransforms
from torch.utils.data import ConcatDataset
from torchvision.datasets import MNIST, CIFAR10

class Data:
    
    def __init__(self, dataset_name, *args, **kwargs):
        self.dataset = self.loader(dataset_name, *args, **kwargs)
    
    def loader(self, dataset_name, *args, **kwargs):
        if dataset_name in ['140k_flickr_faces', 'debug']:
            dataset = SynDataset(*args, **kwargs)
        elif dataset_name == 'nips17':
            dataset = Nips17ImgNetData(*args, **kwargs)
        elif dataset_name == 'mnist':
            dataset == MNISTDataset(*args, **kwargs)
        elif dataset_name == 'cifar10':
            dataset == CIFAR10Dataset(*args, **kwargs)
        else:
            raise ValueError('Dataset not recognized')
        return dataset

class BaseDataset:
    
    def __init__(self,
                dataset_name,
                model,
                device,
                batch_size,
                transform,
                adversarial_opt,
                adversarial_training_opt,
                greyscale_opt,
                target_transform=None,
                input_size=224):


        self.transform_type = transform
        self.transforms = IMGTransforms(transform,
                                   device=device,
                                   target_transform=target_transform, 
                                   input_size=input_size, 
                                   adversarial_opt=adversarial_opt,
                                   greyscale_opt=greyscale_opt,
                                   dataset_type=dataset_name,
                                   model=model)
        self.greyscale_opt = greyscale_opt
        self.adversarial_opt = adversarial_opt
        self.adversarial_training_opt = adversarial_training_opt
        self.device = device
        self.batch_size = batch_size
        self.x, self.y = input_size, input_size
        

class SynDataset(BaseDataset):
    
    # for 140k_flickr_faces

    def __init__(self,
                syndataset_type,
                lrf_visualization=False,
                *args,
                **kwargs):
        super().__init__(syndataset_type, *args, **kwargs)
        
        self.dataset_type = syndataset_type
        self.train_data, self.val_data, self.test_data =  self.get_data(syndataset_type, lrf_visualization)
        
        if self.adversarial_training_opt.adversarial_training:
            train_dev_data = ConcatDataset([self.train_data, self.val_data])
            self.train = DataLoader(train_dev_data, batch_size=self.batch_size, shuffle=True)
        else:
            self.train = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
            self.validation = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=True)
            
        self.test = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

    def get_data(self, dataset, lrf_visualization):
        if dataset == '140k_flickr_faces':
            path = './data/140k_flickr_faces'

            train_labels = path + '/train.csv'
            val_labels = path + '/valid.csv'
            test_labels = path + '/test.csv'
            data_path = path + '/real_vs_fake/real-vs-fake/'

            train = FlickrSubset(label_path=train_labels, img_path=data_path, transform=self.transforms.transform_train, target_transform=self.transforms.target_transform, adversarial=self.adversarial_opt.adversarial)
            val = FlickrSubset(label_path=val_labels, img_path=data_path, transform=self.transforms.transform_val, target_transform=self.transforms.target_transform, adversarial=self.adversarial_opt.adversarial)
            if lrf_visualization:
                test = FlickrSubsetWithPath(label_path=test_labels, img_path=data_path, transform=self.transforms.transform_val, target_transform=self.transforms.target_transform, adversarial=self.adversarial_opt.adversarial)
            else:
                test = FlickrSubset(label_path=test_labels, img_path=data_path, transform=self.transforms.transform_val, target_transform=self.transforms.target_transform, adversarial=self.adversarial_opt.adversarial)


            return train, val, test
        
        elif dataset == 'Augmented140k_flickr_faces':
            path = './data/140k_flickr_faces'

            train_labels = path + '/train.csv'
            val_labels = path + '/valid.csv'
            test_labels = path + '/test.csv'
            data_path = path + '/real_vs_fake/real-vs-fake/'
    
            train = AugmentedFlickrSubset(label_path=train_labels, img_path=data_path, transform=self.transforms.transform_train, target_transform=self.transforms.target_transform)
            val = AugmentedFlickrSubset(label_path=val_labels, img_path=data_path, transform=self.transforms.transform_val, target_transform=self.transforms.target_transform)
            test = AugmentedFlickrSubset(label_path=test_labels, img_path=data_path, transform=self.transforms.transform_val, target_transform=self.transforms.target_transform)

            return train, val, test

        elif dataset == 'debug':

            path = './data/debug'

            train_labels = path + '/train.csv'
            val_labels = path + '/valid.csv'
            test_labels = path + '/test.csv'
            data_path = path + '/real_vs_fake/real-vs-fake/'
            train = FlickrSubset(label_path=train_labels, 
                                img_path=data_path, 
                                transform=self.transforms.transform_train, 
                                target_transform=self.transforms.target_transform, 
                                adversarial=self.adversarial_opt.adversarial)
            val = FlickrSubset(label_path=val_labels, 
                               img_path=data_path, 
                               transform=self.transforms.transform_val, 
                               target_transform=self.transforms.target_transform, 
                               adversarial=self.adversarial_opt.adversarial)
            if lrf_visualization:
                test = FlickrSubsetWithPath(label_path=test_labels, 
                                            img_path=data_path, 
                                            transform=self.transforms.transform_val, 
                                            target_transform=self.transforms.target_transform, 
                                            adversarial=self.adversarial_opt.adversarial, 
                                            is_test_data=True)
            else:
                test = FlickrSubset(label_path=test_labels, 
                                    img_path=data_path, 
                                    transform=self.transforms.transform_val, 
                                    target_transform=self.transforms.target_transform, 
                                    adversarial=self.adversarial_opt.adversarial, 
                                    is_test_data=True)


            return train, val, test

    def get_dim(self, dataset):
        x, y =  dataset[0][0][0].shape
        return x, y


class Nips17ImgNetData(BaseDataset):

    def __init__(self, *args,**kwargs):
        super().__init__('nips17', *args, **kwargs)
        
        self.categories = self.get_categories()
        self.dataset_type = 'nips17'

        self.test_data = self.get_data(transform_val=self.transforms.transform_val, 
                                    target_transform=self.transforms.target_transform)
        self.test = self.train = self.validation =  DataLoader(self.test_data, batch_size=self.batch_size)

    def get_data(self, transform_val, target_transform,):
        path_test = './data/nips17/'
        path_labels = path_test + 'images.csv'
        path_images = path_test + 'images/'
        test = Nips17Subset(label_path=path_labels, 
                            img_path=path_images, 
                            transform=transform_val, 
                            target_transform=target_transform, 
                            adversarial=self.adversarial_opt.adversarial, 
                            is_test_data=True)
        return test
        
    def get_categories(self):
        categories = {}
        path = './data/nips17/categories.csv'
        with open(path, 'r') as cats:
            filereader = csv.reader(cats)
            next(filereader)
            for ind, cat in filereader:
                categories[int(ind) - 1] = cat
        return categories

class MNISTDataset:
    
    def __init__(self,
                *args,
                **kwargs):
        super().__init__('mnist', *args, **kwargs)
        
        self.train_val_data, self.test_data =  self.get_data()
        self.dataset_type = 'mnist'
        
        if self.adversarial_training_opt.adversarial_training:
            self.train = DataLoader(self.train_val_data, batch_size=self.batch_size, shuffle=True)
        else:
            self.train_data, self.val_data = self.train_val_data.split_random(self.train_val_data, lengths=[0.8, 0.2])
            self.train_data = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
            self.validation = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)
        self.test = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
    
    def get_data(self):
        train_val_data = MNIST(root='./data', train=True, download=True, transform=self.transforms)
        test_data = MNIST(root='./data', train=False, download=True, transform=self.transforms)
        return train_val_data, test_data
    

class CIFAR10Dataset:
    
    def __init__(self,
                *args,
                **kwargs):
        super().__init__('cifar10', *args, **kwargs)
        
        self.train_val_data, self.test_data =  self.get_data()
        self.dataset_type = 'cifar10'

        if self.adversarial_training_opt.adversarial_training:
            self.train = DataLoader(self.train_val_data, batch_size=self.batch_size, shuffle=True)
        else:
            self.train_data, self.val_data = self.train_val_data.split_random(self.train_val_data, lengths=[0.8, 0.2])
            self.train_data = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
            self.validation = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)
        self.test = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

    def get_data(self):
        train_val_data = CIFAR10(root='./data', train=True, download=True, transform=self.transforms)
        test_data = CIFAR10(root='./data', train=False, download=True, transform=self.transforms)
        return train_val_data, test_data
        
if __name__ == '__main__':
    pass
