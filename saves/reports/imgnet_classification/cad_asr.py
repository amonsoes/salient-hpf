import csv
import argparse
import os

class ASR:
    
    def __init__(self, path, basepath):
        self.path = path + '/' + 'report.csv'
        self.base_path = basepath + '/' + 'report.csv'
        self.n = 0
        self.success = 0

    def check_len(self, results_f, base_f):
        with open(self.path, 'r') as results_f:
            with open(self.base_path) as base_f:
                results_obj = csv.reader(results_f)
                base_obj = csv.reader(base_f)
                row_count_results = sum(1 for row in results_obj)
                row_count_base = sum(1 for row in base_obj)
                if row_count_base != row_count_results:
                    raise ValueError(f'ERROR: reports have different lenght base:{row_count_base} report:{row_count_results}')
        
    
    def __call__(self):
        with open(self.path, 'r') as results_f:
            with open(self.base_path) as base_f:
                self.check_len(results_f, base_f)
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
                return self.success / self.n

class ConditionalAverageRate:
    
    def __init__(self, path, basepath):
        self.path = path + '/' + 'report.csv'
        self.base_path = basepath + '/' + 'report.csv'
        self.acc_dist = 0.0
        self.n = 0
    
    def check_len(self, results_f, base_f):
        with open(self.path, 'r') as results_f:
            with open(self.base_path) as base_f:
                results_obj = csv.reader(results_f)
                base_obj = csv.reader(base_f)
                row_count_results = sum(1 for row in results_obj)
                row_count_base = sum(1 for row in base_obj)
                if row_count_base != row_count_results:
                    raise ValueError(f'ERROR: reports have different lenght base:{row_count_base} report:{row_count_results}')
        
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
                                self.acc_dist += float(r_line[-1])
                            self.n += 1
                return self.acc_dist / self.n


def get_base(res_path):
    run_name = res_path.split('/')[-1]
    base_path = res_path.split('/')[:-1]
    run_base_ls = run_name.split('_')
    run_base_str = "_".join(run_base_ls[1:3])
    run_base_str += '_base'
    return "/".join(base_path) + '/' +run_base_str
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('res_path', type=str, default='', help='results path')
    args = parser.parse_args()
    
    res_path = args.res_path
    #res_path= '2023-05-03_ImgNetCNN_resnet_hpf_vmifgsm_0.0004'
    
    res_path = os.path.abspath(res_path)
    
    base_path = get_base(res_path)
    
    asr_metric = ASR(res_path, base_path)
    cad_metric = ConditionalAverageRate(res_path, base_path)
    
    
    print('#####################################\n\nRESULTS:\n')
    print(f'ASR : {asr_metric()}\n')
    print(f'CAD : {cad_metric()}\n')
    print('\n#####################################')
    
    
    