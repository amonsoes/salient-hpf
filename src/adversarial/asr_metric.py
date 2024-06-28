import csv

class ASR:
    
    def __init__(self, path, basepath, is_targeted=False):
        if is_targeted:
            self.target_report = path + '/' + 'target_labels.csv'
        used_attack_compression, compression_value = self.check_for_compression(path)
        self.path = path + '/' + 'report.csv'
        if used_attack_compression:
            basepath += f'_compr-{compression_value}'
        self.base_path = basepath + '/' + 'report.csv'
        self.n = 0
        self.success = 0
        self.eps = 1e-7
        self.call_fn = self.compute_targeted if is_targeted else self.compute_untargeted

    def check_len(self, results_f, base_f):
        with open(self.path, 'r') as results_f:
            with open(self.base_path) as base_f:
                results_obj = csv.reader(results_f)
                base_obj = csv.reader(base_f)
                row_count_results = sum(1 for row in results_obj)
                row_count_base = sum(1 for row in base_obj)
                if row_count_base != row_count_results:
                    print(f'####\nWARNING: reports have different lenght base:{row_count_base} report:{row_count_results}\n####')
        return row_count_results
        
    def check_for_compression(self, path):
        used_attack_compression = False
        compression_value = 0
        with open(path + '/' + 'run_params.txt', 'r') as run_params_f:
            next(run_params_f)
            for line in run_params_f:
                if line != '\n':
                    param, value = line.split(':')
                    if param.strip() == 'attack_compression' and value.strip() == 'True':
                        used_attack_compression = True
                    elif param.strip() == 'compression_rate':
                        compression_value = value.strip()
        return used_attack_compression, compression_value
                
    def __call__(self):
        asr = self.call_fn()
        return asr
    
    def compute_untargeted(self):
        with open(self.path, 'r') as results_f:
            with open(self.base_path, 'r') as base_f:
                row_count_results = self.check_len(results_f, base_f)
                results_obj = csv.reader(results_f)
                base_obj = csv.reader(base_f)
                next(results_obj)
                next(base_f)
                for r_line, b_line in zip(results_obj, base_obj):
                    if b_line[1] == b_line[2]: # check if base model predicted correctly
                        if r_line[-1] != '0.0': # check if an attack actually happened 
                            if r_line[1] != r_line[2]: # check if adv model forced misclassification
                                self.success +=1
                            self.n += 1
                return self.success / (self.n + self.eps)
        
    def compute_targeted(self):
        with open(self.path, 'r') as results_f:
            with open(self.base_path, 'r') as base_f:
                with open(self.target_report, 'r') as target_f:
                    row_count_results = self.check_len(results_f, base_f)
                    results_obj = csv.reader(results_f)
                    base_obj = csv.reader(base_f)
                    targets_obj = csv.reader(target_f)
                    next(results_obj)
                    next(base_f)
                    next(targets_obj)
                    for r_line, b_line, t_line in zip(results_obj, base_obj, targets_obj):
                        if b_line[1] == b_line[2]: # check if base model predicted correctly
                            if r_line[-1] != '0.0': # check if an attack actually happened 
                                if r_line[1] != r_line[2]: # check if adv model forced misclassification
                                    if r_line[1] == t_line[0]: # check if model predicted target class from attack
                                        self.success += 1
                                self.n += 1
                    return self.success / (self.n + self.eps)

class ConditionalAverageRate:
    
    def __init__(self, path, basepath):
        self.path = path + '/' + 'report.csv'
        self.base_path = basepath + '/' + 'report.csv'
        self.acc_dist = 0.0
        self.n = 0
        eps, dim = self.get_epsilon(path)
        #delta_eps = torch.full((3, dim, dim), eps)
        #self.max_pert_dist = delta_eps.pow(2).sum().sqrt()
        self.mu = 0.000001
        
    
    def get_epsilon(self, path):
        dim = 224 # a std value to avoid chrashing
        with open(path + '/' + 'run_params.txt', 'r') as f:
            for line in f.readlines():
                if line.startswith('eps'):
                    _, value = line.strip().split(':')
                    eps = value
                elif line.startswith('surrogate_model'):
                    _, value = line.strip().split(':')
                    if value == 'resnet':
                        dim = 224
                    elif value == 'inception':
                        dim = 299
                
        return float(value), dim
    
    def check_len(self, results_f, base_f):
        with open(self.path, 'r') as results_f:
            with open(self.base_path) as base_f:
                results_obj = csv.reader(results_f)
                base_obj = csv.reader(base_f)
                row_count_results = sum(1 for row in results_obj)
                row_count_base = sum(1 for row in base_obj)
                if row_count_base != row_count_results:
                    print(f'####\nWARNING: reports have different lenght base:{row_count_base} report:{row_count_results}\n####')
        
    def __call__(self):
        with open(self.path, 'r') as results_f:
            with open(self.base_path) as base_f:
                self.check_len(results_f, base_f)
                results_obj = csv.reader(results_f)
                base_obj = csv.reader(base_f)
                self.check_len(results_obj, base_obj)
                next(results_obj)
                next(base_f)
                for r_line, b_line in zip(results_obj, base_obj):
                    if r_line[-1] != '0.0': # check if adv attack was applied
                        if b_line[1] == b_line[2]: # check if base model predicted correctly
                            if r_line[1] != r_line[2]: # check if adv model forced misclassification
                                if r_line[-1] != 'nan': # some bug in l2 computation we have to figure out
                                    self.acc_dist += float(r_line[-1]) #/ self.max_pert_dist)
                                    self.n += 1
                return self.acc_dist / (self.n + self.mu)

    
    
if __name__ == '__main__':
    
    path = '/home/amon/git_repos/adv-attacks/saves/reports/intra_model_detection/2023-07-11_ImgNetCNN_xception_bim_0.2_2'
    basepath = '/home/amon/git_repos/adv-attacks/saves/reports/intra_model_detection/ImgNetCNN_xception_base'
    
    asr = ASR(path, basepath)
    asr_result = asr()
    print(asr_result)