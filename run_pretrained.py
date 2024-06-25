from src.model.pretrained import IMGNetCNNLoader
from src.model.trainer import Trainer
from src.datasets.data import Data
from options import args



if __name__ == '__main__':
    
    if args.dataset == 'debug':
        n_classes = 2
        model_dir_name = 'ImgNetCNN'
    elif args.dataset == 'nips17':
        n_classes = 1000
        model_dir_name = 'ImgNetCNN'
    elif args.dataset == 'cifar10':
        n_classes = 10
        model_dir_name = 'CIFAR10CNN'
    elif args.dataset == 'cifar100':
        n_classes = 100
        model_dir_name = 'CIFAR100CNN'
    

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

    if not args.pretrained and data.dataset.dataset_type not in ['nips17', 'cifar100']:
        #print(f'\nrunning training for: \n{trainer.training.model}\n')
        best_acc = trainer.train_model(args.save_opt)
    
    trainer.test_model(args.save_opt)

print('done')