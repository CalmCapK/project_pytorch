import argparse
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
import yaml

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
    elif schedule_type == 'cosineAnnWarm':
        scheduler = CosineAnnealingWarmRestarts(optimizer, **schedule_params['params'])
        #(optimizer, 5, T_mult=1, eta_min=0, last_epoch=-1, verbose=False)
    else:
        scheduler = None
    return scheduler

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    new_lr = lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,
                        default='./configs/config.yaml')
    args = parser.parse_args()
    import torchvision.models as models
    net = models.__dict__["resnet50"](pretrained=True)
    with open(args.config_file) as f:
        config = yaml.load(f)
        
    optimizer = get_optimizer(net, "SGD", config["optimizer_params"]["SGD"])
    scheduler = get_schedule(optimizer, "cosineAnnWarm", config["schedule_params"]["cosineAnnWarm"])
    #scheduler = get_schedule(optimizer, "poly", config["schedule_params"]["poly"])
                       
    import numpy as np
    y = []
    epochs = 15
    steps = 20
    for epoch in range(0, epochs):
        for step in range(0, steps):
            optimizer.zero_grad()
            optimizer.step()
            print("第%d个batch的学习率：%f" % (step, optimizer.param_groups[0]['lr']))
            y.append(optimizer.param_groups[0]['lr'])
            #scheduler.step(step + epoch * steps)
            scheduler.step(epoch + step / steps)
            #scheduler.step()
        #scheduler.step()
    import matplotlib.pyplot as plt
    print(len(y))
    x = np.arange(0,len(y),1)
    print(len(x))
    plt.plot(x,y)
    plt.show()
    #plt.save('lr.jpg')