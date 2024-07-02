import torch

from IQA_pytorch import MAD
#
import torch
import csv

from math import log10
from IQA_pytorch import MAD, SSIM
from torch.nn import functional as F
from DISTS_pytorch import DISTS
from torchmetrics.image import VisualInformationFidelity as VIF
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM


class ImageQualityMetric:
    
    def __init__(self, metrics, filepath, device='cpu', n_channels=3):
        if not isinstance(metrics, list):
            raise ValueError('Input for ImageQualityMetric should be list.')
        self.mapping = {
            'mad' : 0,
            'ssim' : 1,
            'psnr' : 2,
            'dists' : 3,
            'vif' : 4,
            'msssim' : 5,
        }
        self.metrics_dict = {}
        self.device = device
        self.filepath = filepath
        if self.filepath:
            self.filepath = "/".join(filepath.split('/')[:-2]) + '/' + 'iqa_results.csv' # put results in this csv
            with open(self.filepath, 'w') as results_file:
                results_obj = csv.writer(results_file)
                results_obj.writerow([key for key, _ in sorted(self.mapping.items(), key=lambda x:x[1])])
        for metric in metrics:
            if metric == 'mad':
                self.mad = MAD(channels=n_channels)
                self.metrics_dict['mad'] = self.mad
            elif metric == 'ssim':
                self.ssim = SSIM(channels=n_channels)
                self.metrics_dict['ssim'] = self.ssim
            elif metric == 'psnr':
                self.psnr = PSNR(channels=n_channels)
                self.metrics_dict['psnr'] = self.psnr
            elif metric == 'dists':
                self.dists = DISTS().to(self.device)
                self.metrics_dict['dists'] = self.dists
            elif metric == 'vif':
                self.vif = VIF()
                self.metrics_dict['vif'] = self.vif
            elif metric == 'msssim':
                self.msssim = MSSSIM(data_range=1.0).to(self.device)
                self.metrics_dict['msssim'] = self.msssim
        self.n_channels = n_channels
        self.mad_total = 0
        self.n = 0
        
    def __call__(self, ref_image, adv_image):
        row = ['n/a'] * len(self.mapping)
        if len(ref_image.shape) < 4:
            ref_image = ref_image.unsqueeze(0)
        if len(adv_image.shape) < 4:
            adv_image = adv_image.unsqueeze(0)
        for name, iqa_fn in self.metrics_dict.items():
            result = iqa_fn(ref_image, adv_image)
            row[self.mapping[name]] = float(result)
            if name == 'mad':
                mad_r = torch.tanh((1/10)*result)
            elif name == 'ssim':
                ssim_r = result
        if self.filepath:
            with open(self.filepath, 'a') as results_file:
                results_obj = csv.writer(results_file)
                results_obj.writerow(row)
        self.n += 1
        return mad_r.item(), ssim_r.item()
    
    def get_avg_mad(self):
        return self.mad_total / self.n

def gaussian_filter(input, win):
    out = F.conv2d(input, win, stride=1, padding=0, groups=input.shape[1])
    return out
    
    

class PSNR(torch.nn.Module):
    def __init__(self, channels=3):
    
        super(PSNR, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, X, Y, as_loss=False):
        assert X.shape == Y.shape
        if as_loss:
            score = self.psnr(X, Y)
            norm_score = score / 255 # to better balance the other loss
            return 1 - norm_score # return for min-optimization
        else:
            with torch.no_grad():
                score = self.psnr(X, Y)
            return score
    
    def psnr(self, X, Y):
        """
        This returns the PSNR in decibels. The higher the better
        so it needs to be inverted for optimization (minimization)
        https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        the maximum for pixel vals is 255 but normalized to 1.0.
        
        For color images with three RGB values per pixel, the definition 
        of PSNR is the same except that the MSE is the sum over all 
        squared value differences, divided by image 
        size and by three. For 8bit the maximum PSNR is 255
        """
        mse_score = self.mse(X, Y)
        #psnr_score = 20*log10(1) - 10*torch.log10((mse_score/3)/224)
        psnr_score = 20*log10(1) - 10*torch.log10(mse_score)
        return psnr_score
    
    
class HPFL2(torch.nn.Module):
    
    def __init__(self, channels=3):
    
        super(HPFL2, self).__init__()
        self.mse = torch.nn.MSELoss()
        self.ig_size = None

    def forward(self, X, Y, mask):
        assert X.shape == Y.shape
        score = self.mask_guided_L2(X, Y, mask)
        return score
    
    def mask_guided_L2(self, X, Y, mask):
        """
        This method computes MSE and weights differences based on a mask

        Args:
            X : torch.Tensor target  
            Y : torch.Tensor reference
            mask : mask holding coefficients for weighting
        """
        if self.ig_size == None:
            self.ig_size = X.numel()
        diff = torch.square(X - Y)
        diff = diff * mask # weight differences by mask
        return diff.sum() / self.ig_size
        
class DWTL2(torch.nn.Module):
    
    # take L2 on the LL band only
    
    def __init__(self, device, wavelet_type='db2',n_dwt=1,targeted=False):
    
        super(DWTL2, self).__init__()
        self.device = device
        self.hpf_dwt = HpfDWT(device, 
                            wavelet_type=wavelet_type,
                            n_dwt=n_dwt,
                            targeted=targeted)
        self.mse = torch.nn.MSELoss()
        self.ig_size = None

    def forward(self, X_ll, Y_ll):
        assert X_ll.shape == Y_ll.shape
        score = self.mse(X_ll, Y_ll)
        return score
    

class ImageQualityMetric:
    
    def __init__(self, metrics, n_channels=3):
        if not isinstance(metrics, list):
            raise ValueError('Input for ImageQualityMetric should be list.')
        self.mad = MAD(channels=n_channels)
        self.psnr = PSNR(channels=n_channels)
        self.n_channels = n_channels
        
        self.mad_total = 0
        self.psnr_total = 0
        self.n = 0
        
    def __call__(self, ref_image, adv_image):
        '''
        expects [0, 255]
        '''
        """if ref_image.dtype == torch.float32:
            ref_image = (ref_image * 255).to(torch.uint8)
        if adv_image.dtype == torch.float32:
            adv_image = (adv_image * 255).to(torch.uint8)"""
        
        if len(ref_image.shape) < 4:
            ref_image = ref_image.unsqueeze(0)
        if len(adv_image.shape) < 4:
            adv_image = adv_image.unsqueeze(0)
            
        mad_for_img = self.mad(ref_image, adv_image)
        self.mad_total += mad_for_img.item()
        psnr_for_img = self.psnr(ref_image, adv_image, as_loss=False)
        self.psnr_total += psnr_for_img.item()
        
        self.n += 1
        return mad_for_img, psnr_for_img
    
    def get_avg_mad(self):
        return self.mad_total / self.n

    def get_avg_psnr(self):
        return self.psnr_total / self.n
        
                
        