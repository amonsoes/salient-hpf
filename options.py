import argparse
import sys
import numpy as np

from src.utils.argutils import str2dict_conv
from src.utils.argutils import set_up_args

def build_args():

    parser = argparse.ArgumentParser()



    # ========= SHARED OPTIONS =========



    #script options
    parser.add_argument('--optimization', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='if True performs genetic algorithm')
    parser.add_argument('--device', type=str, default='cpu', help='set gpu or cpu')
    parser.add_argument('--optim', type=str, default='sgd', help='set to adam or sgd. sgd: sgd-nesterov, adam: radam')
    parser.add_argument('--log_result', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='log result to wandb')

    #adversarial options
    parser.add_argument('--adversarial', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='activate adversarial processing')
    parser.add_argument('--spatial_adv_type', type=str, default='fgsm', help='choose available spatial attack')
    parser.add_argument('--eps', type=float, default=0.004, help='set epsilon for attack boundary')
    
    # adv: white-box
    parser.add_argument('--surrogate_model', type=str, default='resnet', help='set the Image Net model you want to transfer for FGSM')
    parser.add_argument('--alpha', type=float, default=2/255, help='set alpha for step size of iterative fgsm methods')
    parser.add_argument('--log_mu', type=float, default=0.4, help='determines the weight of the LoG mask for the final HPF mask')
    parser.add_argument('--N', type=int, default=10, help='set number of samples to be drawn from the eps-neighborhood of adversary gradient for mean calc')
    parser.add_argument('--diagonal', type=int, default=-5, help='set how much low frequency information should be added to dct coefficient comutation. middle_ground:0, less:>0 more:<0')
    parser.add_argument('--lf_boosting', type=float, default=0.0, help='set lf_boosting to boost low frequencies by the amount in hpf settings')
    parser.add_argument('--mf_boosting', type=float, default=0.0, help='set mf_boosting to boost low frequencies by the amount in hpf settings')
    parser.add_argument('--use_sal_mask', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=True, help='set to False to disable salient mask in HPF computation')
    parser.add_argument('--sal_mask_only', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='set to False to disable salient mask in HPF computation')
    parser.add_argument('--hpf_mask_tau', type=float, default=0.7, help='set binary variable to define hpf mask and saliency mask tradeoff')
    
    # adv: compression and counter-compression
    parser.add_argument('--attack_compression', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='activate JPEG compression on spatial attack')
    parser.add_argument('--attack_compression_rate', type=int, default=40, help='set rate of JPEG compression on attack')
    parser.add_argument('--gaussian', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='set to True if you want to apply gaussian after attack computation')
    parser.add_argument('--gauss_kernel', type=int, default=15, help='size of gaussian kernel. should be odd')
    parser.add_argument('--gauss_sigma', type=float, default=2.5, help='set sigma for gaussian kernel')
    
    # adv: black-box: boundary
    parser.add_argument('--max_queries', type=int, default=2, help='boundary attack, pg_rgf: set max queries')
    parser.add_argument('--p', type=str, default='l2', help='boundary/nes/pg_rgf: l-norm')
    parser.add_argument('--steps', type=int, default=5000, help='boundary attack: set nr of steps')
    parser.add_argument('--spherical_step', type=float, default=0.008, help='boundary attack: spherical step size')
    parser.add_argument('--source_step', type=float, default=0.0016, help='boundary attack: step size towards source')
    parser.add_argument('--source_step_convergence', type=float, default=0.000001, help='boundary attack: threshold for convergence (eps)')
    parser.add_argument('--step_adaptation', type=int, default=1000, help='boundary attack: if step size should be adapted')
    parser.add_argument('--update_stats_every_k', type=int, default=30, help='boundary attack: every k times epherical and source step are updated')
    
    # adv: black-box: PG-RGF
    # max_queries
    parser.add_argument('--samples_per_draw', type=int, default=10, help='pg-rgf: number of samples (rand vecs) to estimate the gradient.')
    parser.add_argument('--method', type=str, default='fixed_biased', help='pg-rgf: methods used in the attack. uniform: RGF, biased: P-RGF (\lambda^*), fixed_biased: P-RGF (\lambda=0.5)')
    parser.add_argument('--dataprior', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='pg-rgf: whether to use data prior in the attack.')
    parser.add_argument('--sigma', type=float, default=0.0224, help='pg-rgf: float value sampling variance')
    parser.add_argument('--learning_rate', type=float, default=2.0, help='pg-rgf: adjustment rate of adversarial sample')
    
    # adv: black-box: NES
    parser.add_argument('--max_loss_queries', type=int, default=1000, help='nes attack: maximum nr of calls allowed to approx. grad ')
    parser.add_argument('--fd_eta', type=float, default=0.001, help='nes attack: step size of forward difference step')
    parser.add_argument('--nes_lr', type=float, default=0.005, help='nes attack: learning rate of NES step')
    parser.add_argument('--q', type=int, default=20, help='number of noise samples per NES step')
    
    # adv: black-box: SquareAttack
    parser.add_argument('--p_init', type=float, default=0.008, help='square attack: percentage of pixels to be attacked')
    
    # adv: spectral attack
    parser.add_argument('--power_dict_path', type=str, default='./src/adversarial/adv_resources/', help='set from where you want to load the power dict')
    parser.add_argument('--spectral_delta_path', type=str, default='./src/adversarial/adv_resources/', help='set from where you want to load the delta')

    # adv: adversarial training options
    parser.add_argument('--adversarial_training', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='activate adversarial training')
    parser.add_argument('--adv_training_type', type=str, default='', help='set type of adv training')
    parser.add_argument('--attacks_for_training', type=lambda x: x.split(','), default='bim', help='comma-separate attack you want to use for training')
    parser.add_argument('--training_eps', type=float, default=8.0, help='set epsilon for attack computation during training')
        
    # transform options
    parser.add_argument('--cross_offset', type=lambda x: tuple(map(int, x.split(', '))), default=(0,0), help='which cross-band pixel correlation offset to use. enter in form x,x')

    # Dataset options
    parser.add_argument('--dataset', type=str, default='140k_flickr_faces', help='which dataset to use')
    parser.add_argument('--greyscale_fourier', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='Fourier transform with 1(True) channel or 3(False)')
    parser.add_argument('--greyscale_processing', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='Image transform with 1(True) channel or 3(False)')
    

    filename = sys.argv[0].split('/')[-1]


    # ========= BASECNN OPTIONS =========


    if filename == 'run_basecnn.py':
        
        #basecnn options
        parser.add_argument('--pretrained', type=str, default='', help='set wether to load weights')
        parser.add_argument('--conv1', type=str2dict_conv, default={'size':5, 'out':17, 'pad':1, 'stride':1, 'dil':1}, help='set input, size, output, padding, stride and dilation in this order by comma-separated values')
        parser.add_argument('--conv2', type=str2dict_conv, default={'size':6, 'out':10, 'pad':4, 'stride':1, 'dil':1}, help='set input, size, output, padding, stride and dilation in this order by comma-separated values')
        parser.add_argument('--pool1', type=int, default=3, help='set pool2 size')
        parser.add_argument('--pool2', type=int, default=3, help='set pool1 size')
        parser.add_argument('--fc2', type=int, default=400, help='set size of dense layer')
        parser.add_argument('--dropout', type=float, default=0.4, help='set dropout')
        parser.add_argument('--batchsize', type=int, default=2, help='set batch size')
        parser.add_argument('--lr', type=float, default=0.0005, help='set learning rate')
        parser.add_argument('--epochs', type=int, default=7, help='set number of trainig epochs')
        parser.add_argument('--transform', type=str, default='pretrained_imgnet', help='which transform to perform on imgs')
        parser.add_argument('--input_size', type=int, default=224, help='set input size for BaseCnn')


    # ========= PRETRAINED OPTIONS =========


    elif filename == 'run_pretrained.py':

        #transfer model options
        parser.add_argument('--model_name', type=str, default='resnet', help='set the Image Net model you want to transfer')
        parser.add_argument('--as_ft_extractor', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='wether to freeze layers of transferred model')
        parser.add_argument('--pretrained', type=str, default='', help='set wether to load weights')
        parser.add_argument('--batchsize', type=int, default=8, help='set batch size')
        parser.add_argument('--lr', type=float, default=0.00005, help='set learning rate')
        parser.add_argument('--epochs', type=int, default=15, help='set number of trainig epochs')
        parser.add_argument('--model_out', type=str, default='pretrained', help='set the name of the output model')
        parser.add_argument('--transform', type=str, default='pretrained_imgnet', help='which transform to perform on imgs')
        
        # adversarial pretrained options
        parser.add_argument('--adversarial_pretrained', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='transfer adversarially pretrained')
        parser.add_argument('--adv_pretrained_protocol', type=str, default='fbf', help='transfer adversarially pretrained')


    # ========= COATNET OPTIONS =========


    elif filename == 'run_coatnet.py':

        parser.add_argument('--num_blocks', type=lambda x: x.split(','), default=[2, 2, 3, 5, 2], help='set block iterations for CoAtNet modules')
        parser.add_argument('--num_channels', type=lambda x: x.split(','), default=[64, 96, 192, 384, 768], help='set block iterations for CoAtNet modules')    
        parser.add_argument('--pretrained', type=str, default='', help='set wether to load weights')
        parser.add_argument('--batchsize', type=int, default=16, help='set batch size')
        parser.add_argument('--lr', type=float, default=0.0005, help='set learning rate')
        parser.add_argument('--epochs', type=int, default=20, help='set number of trainig epochs')
        parser.add_argument('--model_out', type=str, default='coatnet', help='set the name of the output model')
        parser.add_argument('--transform', type=str, default='pretrained_imgnet', help='which transform to perform on imgs')
        parser.add_argument('--input_size', type=int, default=224, help='set input size of images')


    args = parser.parse_args()
    args = set_up_args(args, filename)
    
    return args

args = build_args()

if __name__ == '__main__':
    pass
