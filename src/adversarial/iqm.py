import torch

from IQA_pytorch import MAD


class ImageQualityMetric:
    
    def __init__(self, metrics, n_channels=3):
        if not isinstance(metrics, list):
            raise ValueError('Input for ImageQualityMetric should be list.')
        for metric in metrics:
            if metric == 'mad':
                self.mad = MAD(channels=n_channels)
        self.n_channels = n_channels
        self.mad_total = 0
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
        self.n += 1
        return mad_for_img
    
    def get_avg_mad(self):
        return self.mad_total / self.n
        
                
        