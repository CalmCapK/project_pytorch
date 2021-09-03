import torch

def get_loss(loss_type):
    #每个概率值和为1，BCEWithLogitsLoss的每个概率没有关系
    if loss_type == 'CrossEntropyLoss':
        return torch.nn.CrossEntropyLoss()