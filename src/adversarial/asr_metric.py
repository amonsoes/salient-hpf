import csv

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
        self.eps = 0.000001
    
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
                return self.acc_dist / (self.n + self.eps)

    
    
if __name__ == '__main__':
    
    path = '/home/amon/git_repos/adv-attacks/saves/reports/intra_model_detection/2023-07-11_ImgNetCNN_xception_bim_0.2_2'
    basepath = '/home/amon/git_repos/adv-attacks/saves/reports/intra_model_detection/ImgNetCNN_xception_base'
    
    asr = ASR(path, basepath)
    asr_result = asr()
    print(asr_result)