import numpy as np
import torch
import random
import os 

def make_longtailed_imb(max_num, class_num, gamma):
    mu = np.power(1/gamma, 1/(class_num))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / gamma))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))

    return np.array(class_num_list)

def set_random_seed(args):
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    random.seed(args.random_seed)
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    