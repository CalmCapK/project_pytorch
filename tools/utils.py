import csv

from numpy.core.records import array
import horovod.torch as hvd
import numpy as np
import os
import random
import shutil
import torch
import warnings

#preds: [batchsize x 3]xn  - > [n x batchsize x 3] x nprocs
#labels: [batchsize]xn - > [n x batchsize] x nprocs
def all_gather_tensor(labels, preds, device, nprocs):
    preds = torch.cat(preds, 0)
    labels = torch.cat(labels, 0)
    #效果同上
    #preds = [x.cpu().detach().numpy() for x in preds]
    #labels = [x.cpu().detach().numpy() for x in labels]
    #preds = np.concatenate((preds),axis=0)
    #labels = np.concatenate((labels),axis=0)
    #preds = torch.from_numpy(preds).to(device)
    #labels = torch.from_numpy(labels).to(device)

    preds_list = [preds.clone() for i in range(nprocs)]
    labels_list = [labels.clone() for i in range(nprocs)]
    torch.distributed.all_gather(preds_list, preds)
    torch.distributed.all_gather(labels_list, labels)
    return labels_list, preds_list

def all_gather_tensor_hvd(labels, preds, device):
    preds = torch.cat(preds, 0)
    labels = torch.cat(labels, 0)
    preds_total = hvd.allgather(preds, name='barrier')
    labels_total = hvd.allgather(labels, name='barrier')
    preds_list = list(preds_total.chunk(2, 0))
    labels_list = list(labels_total.chunk(2, 0))
    return labels_list, preds_list

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.     SUM)
    rt /= nprocs
    return rt

def reduce_mean_hvd(tensor):
    rt = tensor.clone()
    rt = hvd.allreduce(rt, name='barrier') #op=None 返回平均值
    return rt

def init_seed(seed):
    random.seed(seed) #Python本身的随机因素
    np.random.seed(seed) #numpy随机因素
    torch.manual_seed(seed) #pytorch cpu随机因素
    torch.cuda.manual_seed(seed)  #pytorch gpu随机因素
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True  #True: 每次返回的卷积算法将是确定的
    torch.backends.cudnn.benchmark = False #GPU，将其值设置为 True，就可以大大提升卷积神经网络的运行速度

    warnings.warn('You have chosen to seed training. '
                    'This will turn on the CUDNN deterministic setting, '
                    'which can slow down your training considerably! '
                    'You may see unexpected behavior when restarting '
                    'from checkpoints.')

def read_data(path):
    lines = map(str.strip, open(path).readlines())
    data = []
    for line in lines:
        if len(line.split()) == 1:
            data.append(line)
        else:
            #sample_path, label = line.split()
            sample_path, label, _, _ = line.split()
            label = int(label)
            data.append((sample_path, label))
    return data

@profile
def record_epoch(mode, epoch, total_epoch, record, record_path):
    print('\n[%s] Epoch [%d/%d]' %
          (mode, epoch, total_epoch), end='')
    for k, v in record.items():
        print(', %s: %.4f' % (k, v), end='')
    print()
    if not os.path.exists(record_path):
        with open(record_path, 'a+') as f:
            record['epoch'] = epoch
            fieldnames = [k for k, v in record.items()]
            csv_write = csv.DictWriter(f, fieldnames=fieldnames)
            csv_write.writeheader()
            csv_write.writerow(record)
    else:
        with open(record_path, 'a+') as f:
            record['epoch'] = epoch
            fieldnames = [k for k, v in record.items()]
            csv_write = csv.DictWriter(f, fieldnames=fieldnames)
            #data_row = [epoch]
            #data_row.extend(['%.4f' % (v) for k, v in record.items()])
            #csv_write.writerow(data_row)
            csv_write.writerow(record)

def write_result_csv(path, datas):
    with open(path, 'w') as f:
        csv_write = csv.writer(f, delimiter=' ')
        for data in datas:
            data_row = [x for x in data]
            csv_write.writerow(data_row)

def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {}".format(num_params))


def load_model(model, model_path, device):
    checkpoint = torch.load(
        model_path, map_location=device)
    #checkpoint = torch.load(
    #    model_path, map_location=lambda storage, loc: storage)
    state_dict = checkpoint['state_dict']
    pretrained_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}	
    model.load_state_dict(pretrained_dict)
    return checkpoint['epoch'], checkpoint['best_score']


def save_model(epoch, state, model_type, save_model_path, isBetter):
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    if isBetter:
        print(save_model_path + '/' + model_type +
              "_epoch_{}_best.pth".format(epoch))
        torch.save(state, save_model_path + "/" +
                   model_type + "_epoch_{}_best.pth".format(epoch))
    else:
        print(save_model_path + '/' + model_type +
              "_epoch_{}.pth".format(epoch))
        torch.save(state, save_model_path + "/" +
                   model_type + "_epoch_{}.pth".format(epoch))


def remove_makedir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print('Remove path - %s' % dir_path)
    os.makedirs(dir_path)
    print('Create path - %s' % dir_path)


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()

# use:
# progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, top1, top5],
#                             prefix="Epoch: [{}]".format(epoch))
class ProgressMeter(object):
    def __init__(self, num_batches, meters=[], prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, meters=[]):
        self.meters = meters
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

if __name__ == '__main__':
    import argparse
    from argparse import Namespace
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,
                        default='./configs/config.yaml')
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = yaml.load(f)

    record_epoch('train', 1, 100, {'a': 1, 'b': 2},
                 config['save']['train_checkpoint_file'])
