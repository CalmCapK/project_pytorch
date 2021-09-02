from torch import optim

def get_optimizer(model, optimizer_type, optimizer_params):
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=optimizer_params['lr'], 
        momentum=optimizer_params['momentum'], weight_decay=float(optimizer_params['weight_decay']))
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=optimizer_params['lr'])
    return optimizer
          
def get_schedule(model, type):
    return None

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    new_lr = lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr