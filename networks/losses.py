import torch

def get_loss(loss_type):
    #每个概率值和为1，BCEWithLogitsLoss的每个概率没有关系
    if loss_type == 'CrossEntropyLoss':
        return torch.nn.CrossEntropyLoss()


#preds: torch.Size([batchsize x num_classes , 2])
#labels: torch.Size([batchsize x num_classes]) 
def cal_balance_loss(criterion, preds, labels):
    fake_loss = 0
    real_loss = 0
    fake_idx = labels > 0.5
    real_idx = labels <= 0.5
    if torch.sum(fake_idx * 1) > 0:
        fake_loss = criterion(preds[fake_idx], labels[fake_idx])
    if torch.sum(real_idx * 1) > 0:
        real_loss = criterion(preds[real_idx], labels[real_idx])
    if fake_loss > real_loss:
        loss = (1.2 * fake_loss * sum(fake_idx) + real_loss * sum(real_idx)) / len(labels)
    elif fake_loss < real_loss:
        loss = (fake_loss * sum(fake_idx) + 1.2 * real_loss * sum(real_idx)) / len(labels)
    else:
        loss = (fake_loss * sum(fake_idx) + real_loss * sum(real_idx)) / len(labels)
    return loss