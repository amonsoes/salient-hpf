import torch

from torch import nn
from tqdm import tqdm
from src.utils.trainutils import TrainUtils, get_optim
from src.adversarial.spatial import BlackBoxAttack

class Training:
    
    def __init__(self, model, model_name, data, num_classes, optim_args, epochs, model_type, log_result, lr_gamma=0.9):
        self.model = model
        self.data = data
        self.criterion = nn.CrossEntropyLoss()
        self.device = self.model.device
        print(f'.....running on: {self.device}')
        self.model_type = model_type
        self.optim = get_optim(self.model, optim_args)
        self.epochs = epochs
        self.dataset_type = data.dataset_type
        self.num_classes = num_classes
        self.utils = TrainUtils(data=self.data,
                                optim=self.optim,
                                model_name=model_name,
                                model_type=model_type,
                                lr=optim_args['lr'],
                                lr_gamma=lr_gamma,
                                epochs=epochs,
                                num_classes=self.num_classes,
                                device=self.device,
                                log_result=log_result,
                                adversarial_opt=data.adversarial_opt,
                                adversarial_training_opt=data.adversarial_training_opt)

    def report(self, x_hat, y, paths):
        if self.data.adversarial_opt.adversarial:
            if self.data.adversarial_opt.attack_compression:
                l2_norms = self.data.transforms.transform_val.attack_transform.transforms[1].l2_norm
                self.data.transforms.transform_val.attack_transform.transforms[1].l2_norm = []
                self.utils.logger.write_to_report(paths, x_hat, y, l2_norms)
            else:
                l2_norms = self.data.transforms.transform_val.attack_transform.transforms[1].l2_norm
                self.data.transforms.transform_val.attack_transform.transforms[1].l2_norm = []
                self.utils.logger.write_to_report(paths, x_hat, y, l2_norms)    
        else:
            self.utils.logger.write_to_report(paths, x_hat, y)


class CNNTraining(Training):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_model(self, save_opt):
        print(f'\nINITIALIZE TRAINING. \n\nparameters:\n  optimizer:{self.optim}\n  epochs:{self.epochs}\n  batch size:{self.data.batch_size}\n\n')
        best_acc = 0.0
        len_train = len(self.data.train)
        len_val = len(self.data.validation)
        for e in range(self.epochs):
            print(f'\n EPOCH {e}:\n')
            self.utils.metrics.reset()
            self.model.train()
            running_loss = 0.0
            for x, y, _ in tqdm(self.data.train):
                self.optim.zero_grad()
                x, y = x.to(self.device), y.to(self.device)
                x_hat = self.model(x)
                loss = self.criterion(x_hat, y)
                running_loss += float(loss.item() * x.size(0))
                loss.backward()
                self.optim.step()
            # delete  training tensors from device

            del x
            del x_hat
            del loss

            epoch_train_loss = running_loss/len_train
            epoch_result, epoch_val_loss = self.evaluate_model(e, len_val)
            epoch_accuracy = epoch_result['MulticlassAccuracy']
            self.utils.logger.log_eval_results(epoch_train_loss, epoch_val_loss, epoch_result)

            if epoch_accuracy > best_acc:
                best_acc = epoch_accuracy
                save_opt.epoch = e
                self.utils.save_model(model=self.model, save_opt=save_opt, result=epoch_result)

            if self.utils.stopper(epoch_val_loss, self.utils.scheduler):
                break

            self.utils.scheduler.step(epoch_val_loss)
                
        #self.utils.logger.exit()
        return self.model, best_acc
    
    def evaluate_model(self, e, len_val):
        with torch.no_grad():
            self.model.eval()
            print(f'\n Evaluating at epoch {e}:\n')
            running_loss = 0.0
            for x, y, _ in tqdm(self.data.validation):
                x, y = x.to(self.device), y.to(self.device)
                x_hat = self.model(x)
                loss = self.criterion(x_hat, y)
                running_loss += float(loss.item() * x.size(0))
                self.utils.metrics(x_hat, y)
            epoch_val_loss = running_loss/len_val
            result = self.utils.metrics.compute()
            print(f'\nEvaluation on epoch {e}:\n')
            print(result)
            print('\n\n')
            return result, epoch_val_loss
        
    """def test_model(self):
        self.test_model_fn()
    
    def test_model_binary(self):
        with torch.no_grad():
            self.model.eval()
            self.utils.metrics.reset()
            print(f'\nTest Model\n')
            for x, y, paths in tqdm(self.data.test):
                x, y = x.to(self.device), y.to(self.device)
                x_hat = self.model(x)
                #self.utils.metrics(x_hat, y)
                self.utils.metrics(x_hat, y.unsqueeze(1))
                self.report(x_hat, y, paths)
            #high_freq_mean = self.data.transforms.transform_val.adversarial_decider.transforms[1].attack.observer.dct.tile_mean / self.data.transforms.transform_val.adversarial_decider.transforms[1].attack.observer.dct.n
            result = self.utils.metrics.compute()
            #self.utils.logger.log_test_results(result, high_freq_mean)
            self.utils.logger.log_test_results(result)
            print('\nTEST COMPLETED. RESULTS:\n')
            print(result)
            #print(f'HFM: {high_freq_mean}')
            print('\n\n')
            self.utils.logger.exit()
            return result['BinaryAccuracy'].item()"""

    def test_model(self, save_opt):
        with torch.no_grad():
            self.model.eval()
            self.utils.metrics.reset()
            print(f'\nTest Model\n')
            for x, y, paths in tqdm(self.data.test):
                x, y = x.to(self.device), y.to(self.device)
                x_hat = self.model(x)
                self.utils.metrics(x_hat, y)
                self.report(x_hat, y, paths)
            result = self.utils.metrics.compute()
            self.utils.logger.log_test_results(result)
            if hasattr(self.data.transforms, 'attack'):
                self.utils.logger.log_mad(self.data.transforms.attack)
                if isinstance(self.data.transforms.attack, BlackBoxAttack):
                    self.utils.logger.log_black_box_metrics(self.data.transforms.attack)
            print('\nTEST COMPLETED. RESULTS:\n')
            print(result)
            print('\n\n')
            self.utils.logger.exit()
            return 0.8 #result['MulticlassAccuracy'].item()
                

if __name__ == '__main__':
    pass