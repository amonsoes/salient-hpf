import torch
import torchattacks
import torchvision

from torchvision import transforms as T
from torchvision.io import encode_jpeg, decode_image
from src.model.pretrained import IMGNetCNNLoader
from src.adversarial.hpf_mask import HPFMasker
from src.adversarial.black_box.boundary_attack import BoundaryAttack, HPFBoundaryAttack
from src.adversarial.black_box.nes_attack import NESAttack
from src.adversarial.black_box.square_attack import SquareAttack, HPFSquareAttack 
from src.adversarial.black_box.pg_rgf import PriorRGFAttack, HPFPriorRGFAttack
from src.adversarial.iqm import ImageQualityMetric


class AttackLoader:
    
    def __init__(self, 
                attack_type, 
                device,
                model,
                surrogate_model, 
                dataset_type, 
                model_trms,
                surrogate_model_trms,
                hpf_mask_params,
                input_size,
                *args, 
                **kwargs):
        self.device = device
        self.dataset_type = dataset_type
        self.model = model
        self.model_trms = model_trms
        self.surrogate_model = self.get_model(surrogate_model)
        self.surrogate_model_trms = surrogate_model_trms
        self.attack = self.load_attack(attack_type, hpf_mask_params, input_size, *args, **kwargs)
        
    def load_attack(self, attack_type, hpf_mask_params, input_size, *args, **kwargs):
        # if hpf version create hpf_masker
        if attack_type in ['hpf_fgsm',
                            'hpf_bim',
                            'hpf_pgd',
                            'hpf_pgdl2',
                            'hpf_vmifgsm',
                            'ycbcr_hpf_fgsm']:
            hpf_masker = HPFMasker(self.device, input_size=input_size, **hpf_mask_params)
        elif attack_type in ['hpf_boundary_attack', 'hpf_square_attack', 'hpf_pg_rgf']:
            hpf_masker = HPFMasker(self.device, input_size=input_size, is_black_box=True, **hpf_mask_params)
        else:
            hpf_masker = None
        # load white or black box
        if attack_type in ['fgsm',
                           'bim',
                           'pgd',
                           'pgdl2',
                           'vmifgsm',
                           'hpf_fgsm',
                           'hpf_bim',
                           'hpf_pgd',
                           'hpf_pgdl2',
                           'hpf_vmifgsm',
                           'ycbcr_hpf_fgsm',
                           'varsinifgsm']:
            attack = WhiteBoxAttack(attack_type=attack_type,
                                    surrogate_model=self.surrogate_model,
                                    device=self.device,
                                    input_size=self.input_size,
                                    dataset_type=self.dataset_type,
                                    surrogate_model_trms=self.surrogate_model_trms,
                                    hpf_masker=hpf_masker,
                                    *args, 
                                    **kwargs)
        elif attack_type in ['boundary_attack',
                            'nes', 
                            'hpf_boundary_attack',
                            'square_attack',
                            'hpf_square_attack',
                            'pg_rgf',
                            'hpf_pg_rgf']:
            attack = BlackBoxAttack(attack_type,
                                    model=self.model,
                                    surrogate_model=self.surrogate_model,
                                    device=self.device, 
                                    model_trms=self.model_trms,
                                    surrogate_model_trms=self.surrogate_model_trms,
                                    hpf_masker=hpf_masker,
                                    input_size=input_size,
                                    *args, 
                                    **kwargs)
        else:
            raise ValueError('ATTACK NOT RECOGNIZED. Change spatial_adv_type in options')
        return attack

    def get_model(self, surrogate_model):
        if self.dataset_type == 'nips17':
            if surrogate_model == 'adv_resnet_fbf':
                surrogate_path = './saves/models/Adversarial/fbf_models/imagenet_model_weights_2px.pth.tar'
                adv_training_protocol = 'fbf'
            elif surrogate_model == 'adv_resnet_pgd':
                surrogate_path = './saves/models/Adversarial/pgd_models/imagenet_linf_4.pt'
                adv_training_protocol = 'pgd'
            else:
                surrogate_path = ''
                adv_training_protocol = None
            n_classes = 1000
        elif self.dataset_type == 'cifar100':
            surrogate_path = ''
            adv_training_protocol = None
            n_classes = 100
                
        loader = IMGNetCNNLoader(surrogate_path, adv_training_protocol)
        cnn, self.input_size = loader.transfer(surrogate_model, n_classes, feature_extract=False, device=self.device)
        cnn.model_name = surrogate_model
        model = cnn
        model.eval()
        return model 

    def get_l2(self, orig_img, adv_img):
        distance = (orig_img - adv_img).pow(2).sum().sqrt()
        return (distance / orig_img.max()).item()

class BlackBoxAttack:
    
    def __init__(self, 
                attack_type, 
                model, 
                surrogate_model, 
                model_trms, 
                surrogate_model_trms, 
                hpf_masker,
                input_size,
                *args, 
                **kwargs):
        self.black_box_attack = self.load_attack(attack_type,
                                                model,
                                                model_trms,
                                                surrogate_model,
                                                surrogate_model_trms,
                                                hpf_masker,
                                                input_size,
                                                *args, 
                                                **kwargs)
        self.to_float = T.ConvertImageDtype(torch.float32)
        self.l2_norm = []
        self.num_queries_lst = []
        self.mse_list = []
        self.n = 1
    
    def load_attack(self, 
                    attack_type, 
                    model, 
                    model_trms,
                    surrogate_model,
                    surrogate_model_trms,
                    hpf_masker,
                    input_size,
                    *args, 
                    **kwargs):
        if attack_type == 'boundary_attack':
            attack = BoundaryAttack(model, *args, **kwargs)
        elif attack_type == 'hpf_boundary_attack':
            attack = HPFBoundaryAttack(model, hpf_masker, *args, **kwargs)
        elif attack_type == 'nes':
            attack = NESAttack(model, *args, **kwargs)
        elif attack_type == 'square_attack':
            attack = SquareAttack(model, *args, **kwargs)
        elif attack_type == 'hpf_square_attack':
            attack = HPFSquareAttack(model, hpf_masker, *args, **kwargs)
        elif attack_type == 'pg_rgf':
            attack = PriorRGFAttack(model=model,
                                    model_trms=model_trms, 
                                    surrogate_model=surrogate_model, 
                                    surrogate_model_trms=surrogate_model_trms,
                                    input_size=input_size,
                                    num_classes=surrogate_model.n_classes,
                                    *args, 
                                    **kwargs)
        elif attack_type == 'hpf_pg_rgf':
            attack = HPFPriorRGFAttack(hpf_masker,
                                    model=model,
                                    model_trms=model_trms, 
                                    surrogate_model=surrogate_model, 
                                    surrogate_model_trms=surrogate_model_trms,
                                    input_size=input_size,
                                    num_classes=surrogate_model.n_classes,
                                    *args, 
                                    **kwargs)
            
        return attack

    def get_l2(self, orig_img, adv_img):
        distance = (orig_img - adv_img).pow(2).sum().sqrt()
        return (distance / orig_img.max()).item()
    
    def __call__(self, x, target_y):
        orig_x = x
        x = self.to_float(x)
        perturbed_x, num_queries, mse = self.black_box_attack(x, target_y)
        perturbed_x = perturbed_x.squeeze(0).cpu()
        self.l2_norm.append(self.get_l2(orig_x.cpu(), (perturbed_x*255).to(torch.uint8)))
        self.num_queries_lst.append(num_queries)
        self.mse_list.append(mse)
        return perturbed_x
    
    def set_up_log(self, path_log_dir):
        self.path_log_dir = path_log_dir
    
    def get_attack_metrics(self):
        avg_num_queries = sum(self.num_queries_lst) / len(self.num_queries_lst)
        avg_mse = sum(self.mse_list) / len(self.mse_list)
        return avg_num_queries, avg_mse
        
        
class WhiteBoxAttack:
    
    def __init__(self, attack_type, surrogate_model, device, input_size, dataset_type, surrogate_model_trms, hpf_masker,  *args, **kwargs):
        self.model = surrogate_model
        self.device = torch.device(device)
        surrogate_loss = torch.nn.CrossEntropyLoss()
        self.model_trms = surrogate_model_trms
        self.input_size = input_size
        self.attack = self.load_attack(self.model, attack_type, surrogate_loss, hpf_masker, *args, **kwargs)
        self.image_metric = ImageQualityMetric(['mad'])
        self.save_dir = f"./data/survey_data/{attack_type.split('_')[0]}/vanilla" if len(attack_type.split('_')) == 1 else f"./data/survey_data/{attack_type.split('_')[1]}/hpf"
        self.orig_save_dir = "./data/survey_data/orig"
        self.l2_norm = []
        self.n = 1
        self.call_fn = self.attack_sample
        """if dataset_type =='nips17':
            self.call_fn = self.attack_imgnet
        elif dataset_type == 'cifar100':
            self.call_fn = self.attack_cifar"""
            
    def __call__(self, x, y):
        return self.call_fn(x, y)
    
    def attack_sample(self, x, y):
        with torch.enable_grad():
            orig_x = x.clone().detach() / 255
            x = x.to(self.device)
            self.model.zero_grad()
            x = x.unsqueeze(0)
            y = torch.LongTensor([y])
            perturbed_x = self.attack(x, y)
            perturbed_x = perturbed_x.squeeze(0).cpu()
            self.l2_norm.append(self.get_l2(orig_x, perturbed_x))
            mad_score = self.image_metric(orig_x, perturbed_x)
            #torchvision.utils.save_image(orig_x, f'{self.orig_save_dir}/{self.n}.png', format='PNG')
            #torchvision.utils.save_image(perturbed_x, f'{self.save_dir}/{self.n}.png', format='PNG')
            self.n += 1
        return perturbed_x
    
    def load_attack(self, model, attack_type, surrogate_loss, hpf_masker, *args, **kwargs):
        if attack_type in ['vmifgsm', 'hpf_vmifgsm', 'mvmifgsm', 'varsinifgsm']:
            attack = self.load_blackbox(model, attack_type, surrogate_loss, hpf_masker, *args, **kwargs)
        else:
            attack = self.load_whitebox(model, attack_type, surrogate_loss, hpf_masker, *args, **kwargs)
        return attack
    
    def load_blackbox(self, model, attack_type, surrogate_loss, hpf_masker, *args, **kwargs):
        if attack_type == 'vmifgsm':
            attack = torchattacks.attacks.vmifgsm.VMIFGSM(model, surrogate_loss, model_trms=self.model_trms, *args, **kwargs)
        elif attack_type == 'hpf_vmifgsm':
            attack = torchattacks.attacks.vmifgsm.HpfVMIFGSM(model, surrogate_loss=surrogate_loss, model_trms=self.model_trms, hpf_masker=hpf_masker, *args, **kwargs)
        elif attack_type == 'mvmifgsm':
            attack = torchattacks.attacks.vmifgsm.MVMIFGSM(model, surrogate_loss, *args, **kwargs)
        elif attack_type == 'varsinifgsm':
            attack = torchattacks.attacks.sinifgsm.VarSINIFGSM(model, surrogate_loss, *args, **kwargs)
        else:
            raise ValueError('ADVERSARIAL ATTACK NOT RECOGNIZED FROM TYPE. Change spatial_adv_type in options')
        return attack        
    
    def load_whitebox(self, model, attack_type, surrogate_loss, hpf_masker, *args, **kwargs):
        if attack_type == 'fgsm':
            attack = torchattacks.attacks.fgsm.FGSM(model, surrogate_loss, model_trms=self.model_trms, *args, **kwargs)
        elif attack_type == 'hpf_fgsm':
            attack = torchattacks.attacks.fgsm.HpfFGSM(model, surrogate_loss=surrogate_loss, model_trms=self.model_trms, hpf_masker=hpf_masker, *args, **kwargs)
        elif attack_type == 'ycbcr_hpf_fgsm':
            attack = torchattacks.attacks.fgsm.YcbcrHpfFGSM(model, surrogate_loss=surrogate_loss, model_trms=self.model_trms, hpf_masker=hpf_masker, *args, **kwargs)
        elif attack_type == 'bim':
            attack = torchattacks.attacks.bim.BIM(model, surrogate_loss, model_trms=self.model_trms, *args, **kwargs)
        elif attack_type == 'hpf_bim':
            attack = torchattacks.attacks.bim.HpfBIM(model, surrogate_loss=surrogate_loss, model_trms=self.model_trms, hpf_masker=hpf_masker, *args, **kwargs)
        elif attack_type == 'pgd':
            attack = torchattacks.attacks.pgd.PGD(model, surrogate_loss, model_trms=self.model_trms, *args, **kwargs)
        elif attack_type == 'hpf_pgd':
            attack = torchattacks.attacks.pgd.HpfPGD(model, surrogate_loss=surrogate_loss, model_trms=self.model_trms, hpf_masker=hpf_masker, *args, **kwargs)
        elif attack_type == 'pgdl2':
            attack = torchattacks.attacks.pgdl2.PGDL2(model, surrogate_loss, model_trms=self.model_trms, *args, **kwargs)
        elif attack_type == 'hpf_pgdl2':
            attack = torchattacks.attacks.pgdl2.HpfPGDL2(model, surrogate_loss=surrogate_loss, model_trms=self.model_trms, hpf_masker=hpf_masker, *args, **kwargs) #diff for 299(inception) and 224(rest)
        elif attack_type == 'ycbcr_hpf_pgdl2':
            attack = torchattacks.attacks.pgdl2.YcbcrHpfPGDL2(model, input_size=self.input_size, surrogate_loss=surrogate_loss, *args, **kwargs) #diff for 299(inception) and 224(rest)
        elif attack_type == 'sinifgsm':
            attack = torchattacks.attacks.sinifgsm.SINIFGSM(model, surrogate_loss, *args, **kwargs)
        else:

            raise ValueError('ADVERSARIAL ATTACK NOT RECOGNIZED FROM TYPE. Change spatial_adv_type in options')
        return attack

    def get_l2(self, orig_img, adv_img):
        if orig_img.max() > 1:
            raise ValueError('original image is not 0 < x < 1')
        if adv_img.max() > 1:
            raise ValueError('adv image is not 0 < x < 1')
        distance = (orig_img - adv_img).pow(2).sum().sqrt()
        return (distance / orig_img.max()).item()
        
    # Deprecated
    def get_l2norm(self, orig_x, perturbed_x):
        orig_x = orig_x / 255
        perturbed_x = perturbed_x / 255
        
        return torch.linalg.norm(orig_x - perturbed_x)
        
    def get_avg_l2norm(self):
        return self.l2_norm / self.n
    
class Augmenter:
    
    def __init__(self, kernel_size=5, compression_rate=40):
        self.blur = T.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 3.0))
        self.compression_rate = compression_rate
        
    def __call__(self, img):
        img = self.blur(img)
        img = self.jpeg_compression(img)
        return img

    def jpeg_compression(self, img):
        adjusted_for_jpeg = False
        ori_img = img.clone().detach()
        if img.dtype != torch.uint8:
            ig_max, ig_min = img.max().item(), img.min().item()
            img = (img - ig_min) / (ig_max - ig_min)
            img = (img * 255).to(torch.uint8)
            adjusted_for_jpeg = True
        compressed = encode_jpeg(img, self.compression_rate)
        compressed_img = decode_image(compressed)
        if adjusted_for_jpeg:
            compressed_img = compressed_img / 255
            compressed_img = compressed_img*(ig_max-ig_min)+ig_min
        return compressed_img


class Patchify:
    
    def __init__(self, img_size, patch_size, n_channels):
        self.img_size = img_size
        self.n_patches = (img_size // patch_size) ** 2

        assert (img_size // patch_size) * patch_size == img_size
        
    def __call__(self, x):
        p = x.unfold(1, 8, 8).unfold(2, 8, 8).unfold(3, 8, 8) # x.size() -> (batch, model_dim, n_patches, n_patches)
        self.unfold_shape = p.size()
        p = p.contiguous().view(-1,8,8)
        return p
    
    def inverse(self, p):
        if not hasattr(self, 'unfold_shape'):
            raise AttributeError('Patchify needs to be applied to a tensor in ordfer to revert the process.')
        x = p.view(self.unfold_shape)
        output_h = self.patchify.unfold_shape[1] * self.patchify.unfold_shape[4]
        output_w = self.patchify.unfold_shape[2] * self.patchify.unfold_shape[5]
        x = x.permute(0,1,4,2,5,3).contiguous()
        x = x.view(3, output_h, output_w)
        return x 
        
        