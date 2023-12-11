from src.model.pretrained import IMGNetCNNLoader
from src.model.trainer import Trainer
from src.datasets.data import Data
from options import args



if __name__ == '__main__':
    
    if args.dataset == '140k_flickr_faces' or args.dataset == 'debug':
        n_classes = 2
    elif args.dataset == 'nips17':
        n_classes = 1000
    elif args.dataset == 'mnist':
        n_classes = 10
    elif args.dataset == 'cifar10':
        n_classes = 10
    

    loader = IMGNetCNNLoader(args.pretrained, args.adversarial_pretrained_opt)
    cnn, input_size = loader.transfer(args.model_name,
                                      n_classes, 
                                      feature_extract=args.as_ft_extractor, 
                                      device=args.device)
    cnn.model_name = args.model_name

    data = Data(dataset_name=args.dataset,
                device=args.device,
                batch_size=args.batchsize,
                transform=args.transform,
                model=cnn,
                input_size=input_size,
                adversarial_opt=args.adversarial_opt,
                adversarial_training_opt=args.adversarial_training_opt,
                greyscale_opt=args.greyscale_opt)
    
    """if args.dataset == '140k_flickr_faces':
        data = SynDataset(dataset=args.dataset,
                    device=args.device,
                    batch_size=args.batchsize,
                    transform=args.transform,
                    input_size=input_size,
                    adversarial_opt=args.adversarial_opt,
                    adversarial_training_opt=args.adversarial_training_opt,
                    greyscale_opt=args.greyscale_opt) 
    elif args.dataset == 'nips17':
        data = Nips17ImgNetData(device=args.device,
                    batch_size=args.batchsize,
                    transform=args.transform,
                    input_size=input_size,
                    adversarial_opt=args.adversarial_opt,
                    adversarial_training_opt=args.adversarial_training_opt,
                    greyscale_opt=args.greyscale_opt) 
    elif args.dataset == 'debug':
        data = SynDataset(dataset=args.dataset,
                    device=args.device,
                    batch_size=args.batchsize,
                    transform=args.transform,
                    input_size=input_size,
                    adversarial_opt=args.adversarial_opt,
                    adversarial_training_opt=args.adversarial_training_opt,
                    greyscale_opt=args.greyscale_opt)  
    else:
        raise ValueError('DATASET NOT SUPPORTED')"""
    
    args.model_dir_name += '_' + cnn.model_name
    trainer = Trainer(model=cnn,
                    data=data.dataset,
                    model_name=args.model_dir_name,
                    num_classes=n_classes,
                    optim_args=args.optim,
                    epochs=args.epochs,
                    model_type=args.model_out,
                    log_result=args.log_result,
                    adversarial_training_opt=args.adversarial_training_opt)

    if not args.pretrained and data.dataset.dataset_type != 'nips17':
        #print(f'\nrunning training for: \n{trainer.training.model}\n')
        best_acc = trainer.train_model(args.save_opt)
    
    trainer.test_model(args.save_opt)

print('done')