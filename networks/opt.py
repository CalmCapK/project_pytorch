from torch import optim
from torch.optim.lr_scheduler import _LRScheduler

class PolyLR(_LRScheduler):
    """Sets the learning rate of each parameter group according to poly learning rate policy
    """
    def __init__(self, optimizer, max_iter=90000, power=0.9, last_epoch=-1,cycle=False):
        self.max_iter = max_iter
        self.power = power
        self.cycle = cycle
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        self.last_epoch_div = (self.last_epoch + 1) % self.max_iter
        scale = (self.last_epoch + 1) // self.max_iter + 1.0 if self.cycle else 1
        return [(base_lr * ((1 - float(self.last_epoch_div) / self.max_iter) ** (self.power))) / scale for base_lr in self.base_lrs]


def get_optimizer(model, optimizer_type, optimizer_params):
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=optimizer_params['lr'], 
        momentum=optimizer_params['momentum'], weight_decay=float(optimizer_params['weight_decay']))
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=optimizer_params['lr'])
    return optimizer
          
def get_schedule(optimizer, schedule_type, schedule_params):
    if schedule_type == "poly":
        scheduler = PolyLR(optimizer, **schedule_params['params'])
    else:
        scheduler = None
    return scheduler

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    new_lr = lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr