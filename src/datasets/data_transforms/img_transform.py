from torchvision import transforms as T
from src.datasets.data_transforms.spatial_transform import SpatialTransforms
from src.datasets.data_transforms.spectral_transform import SpectralTransforms
from src.adversarial.spatial import Augmenter, AttackLoader


class CustomCompose(T.Compose):
    
    # This class extends the torch Compose class
    # by having the option of passing the label
    # to specific transforms in the list
    
    # accepts a boolean list "target_required" (ie. [0,1,0])
    # that stores for which transforms the label
    # should be passed
    
    def __init__(self, transforms, target_required=False ):
        self.transforms = transforms
        if not target_required:
            self.target_required = self.check_for_custom_composes([0 for i in range(len(transforms))])
        else:
            self.target_required = self.check_for_custom_composes(target_required)
        
    def check_for_custom_composes(self, lst):
        for e, t in enumerate(self.transforms):
            if type(t) == CustomCompose:
                lst[e] = 1
        return lst
                
    def __call__(self, img, tar):
        for t, req in zip(self.transforms, self.target_required):
            if req:
                img = t(img, tar)
            else:
                img = t(img)
        return img


class IMGTransforms:

    def __init__(self,
                transform, 
                target_transform, 
                input_size, 
                device, 
                adversarial_opt, 
                greyscale_opt, 
                dataset_type,
                model,
                *args, 
                **kwargs):
        
        self.device = device
        self.input_size = input_size
        self.adversarial_opt = adversarial_opt
        self.greyscale_opt = greyscale_opt
        self.dataset_type = dataset_type

        # (1) get basis transforms
        
        train_base_transform, val_base_transform= self.get_base_trm()
        
        # (2) get main trainsform
        
        transform_train, transform_val = self.get_main_trm(transform, *args, **kwargs)
        
        # (3) compose base and main transform. adversarial trms get CustomCompose for label access
        
        self.transform_train = T.Compose([train_base_transform, transform_train])
        if adversarial_opt.adversarial:
            self.transform_val = CustomCompose([val_base_transform, transform_val])
        else:
            self.transform_val = T.Compose([val_base_transform, transform_val])

        # (4) if adversarial, define adversarial transform which change self.transform_val and train
        
        if adversarial_opt.adversarial:
            self.transform_train, self.transform_val = self.get_adv_trm(val_base_transform, transform_val, model)

        # (5) define target transforms
        
        if target_transform == None:
            self.target_transform = self.identity
        
    def identity(self, x):
        return x
    
    
    def get_base_trm(self):
        # method to get base trms that are shared across all models
        post_resize = None

        if self.input_size == 299:
            resize = T.Resize(299)
            crop_size = 299 
        else:
            resize = T.Resize(256)
            crop_size = 224
        
        if self.greyscale_opt.greyscale_processing:
            
            
            train_base_transform = T.Compose([resize,
                                                T.RandomResizedCrop(crop_size, scale=(0.5,1)),
                                                T.RandomHorizontalFlip(),
                                                T.Grayscale()])

            val_base_transform = T.Compose([resize,
                                            T.CenterCrop(crop_size), 
                                            T.Grayscale()])            
        else:
            train_base_transform = T.Compose([resize,
                                            T.RandomResizedCrop(crop_size, scale=(0.5,1)),
                                            T.RandomHorizontalFlip()])
            if self.adversarial_opt.adversarial:
                val_base_transform = CustomCompose([resize,
                                                    T.CenterCrop(crop_size)])
            else:
                val_base_transform = T.Compose([resize,
                                                    T.CenterCrop(crop_size)])
        
        return train_base_transform, val_base_transform
    
    def get_main_trm(self, transform, *args, **kwargs):
        if transform in ['standard', 
                        'augmented', 
                        'pretrained_imgnet',
                        'augmented_pretrained_imgnet', 
                        'band_cooccurrence', 
                        'cross_cooccurrence', 
                        'basic_pix_attn_cnn', 
                        'compute_sdn',
                        'ycbcr_transform',
                        'calc_avg_attack_norm']:
            
            transforms = SpatialTransforms(transform,
                                           self.greyscale_opt.greyscale_processing, 
                                           self.input_size, 
                                           *args, 
                                           **kwargs)

            transform_train = transforms.transform_train
            transform_val = transforms.transform_val

        elif transform in ['real_nd_fourier', 'augmented_nd_fourier', 'bi_hpf_transform', 'basic_fr_attn_cnn']:
            transforms = SpectralTransforms(transform, 
                                            self.greyscale_opt, 
                                            self.adversarial_opt, 
                                            self.input_size, 
                                            self.dataset_type, 
                                            *args, 
                                            **kwargs)

            transform_train = transforms.transform_train
            transform_val = transforms.transform_val

        else:
            raise ValueError('WRONG TRANSFORM INPUT. change in options')
        
        return transform_train, transform_val
    
    def get_adv_trm(self, val_base_transform, transform_val, model):
        
        # define spectral adv
        
        """if self.adversarial_opt.spectral_adv:
            self.transform_val_spectral = T.Compose([val_base_transform,
                                                    SpectrumNorm(num_features=40,
                                                                greyscale_processing=self.adversarial_opt.greyscale_processing,
                                                                img_size=self.input_size,
                                                                path_power_dict=self.adversarial_opt.power_dict_path, 
                                                                path_delta=self.adversarial_opt.spectral_delta_path,
                                                                dataset_type=self.dataset_type),
                                                    transform_val])"""
            

        # define spatial adv 
        
        self.transform_val_spatial = None
        if self.adversarial_opt.adversarial:
            if self.adversarial_opt.spatial_adv_type == 'augmenter':
                self.transform_val_spatial = T.Compose([val_base_transform, Augmenter()])
            else:
                surrogate_model_name = self.adversarial_opt.spatial_attack_params.surrogate_model
                surrogate_input_size = 299 if surrogate_model_name in ['inception', 'xception', 'adv-inception'] else 224
                if surrogate_input_size != val_base_transform.transforms[1].size[0]:
                    surrogate_post_resize = T.Resize(surrogate_input_size)
                    
                surrogate_trm_name = 'pretrained_imgnet'
                surrogate_transforms = SpatialTransforms(surrogate_trm_name,
                                            self.greyscale_opt.greyscale_processing, 
                                            surrogate_input_size)
                surrogate_trm = surrogate_transforms.transform_val
                if surrogate_input_size != val_base_transform.transforms[1].size[0]:
                    surrogate_trm = T.Compose([surrogate_post_resize, surrogate_trm])
                attack_loader = AttackLoader(attack_type=self.adversarial_opt.spatial_adv_type,
                                            device=self.device,
                                            dataset_type=self.dataset_type,
                                            model_trms=transform_val,
                                            model=model,
                                            surrogate_model_trms=surrogate_trm,
                                            input_size=self.input_size,
                                            **self.adversarial_opt.spatial_attack_params.__dict__)
                self.attack = attack_loader.attack
                if self.adversarial_opt.attack_compression:
                    augmenter = Augmenter(compression_rate=self.adversarial_opt.compression_rate)
                    self.transform_val_spatial = CustomCompose([val_base_transform, self.attack, augmenter.jpeg_compression, transform_val], [0, 1, 0, 0])
                else:
                    self.transform_val_spatial = CustomCompose([val_base_transform, self.attack, transform_val], [0, 1, 0])
            #self.transform_val_spatial = CustomCompose([self.transform_val_spatial, transform_val])
            
        # put them in AdversarialAttack obj and turn that into self.transform_val
                    
        transform_train_adv = self.identity
        transform_val_adv = AdversarialAttackTransform(trm_bypass=self.transform_val,
                                                dataset_type=self.dataset_type,
                                                attack_transform=self.transform_val_spatial)
        
        return transform_train_adv, transform_val_adv
        


class AdversarialAttackTransform:
    
    def __init__(self, trm_bypass, dataset_type, attack_transform=None):
        self.attack_transform = attack_transform
        self.trm_bypass = trm_bypass
        self.call_fn = self.attack_binary if dataset_type == '140k_flickr_faces' else self.attack
    
    def __call__(self, x, y):
        img = self.call_fn(x, y)
        return img
    
    def attack_binary(self, x, y):
        # this is a special case where we only want to perturb synthetic images
        if y == 0:
            return self.attack_transform(x, y)
        else:
            #self.adversarial_decider.transforms[1].l2_norm.append(0.0)
            return self.trm_bypass(x, y)
    
    def attack(self, x, y):
        return self.attack_transform(x, y)
        
        

if __name__ == '__main__':
    pass