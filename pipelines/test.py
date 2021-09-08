import numpy as np
import torch
from tqdm import tqdm
from tools.custom import AverageMeter
from tools.eval import accuracy, cal_multiclass_metric, cal_binary_metric
from tools.utils import reduce_mean, reduce_mean_hvd, all_gather_tensor, all_gather_tensor_hvd


#@profile
def test_epoch(current_epoch, model, data_loader, device, criterion, parallel_type, nprocs):
    model.train(False)
    model.eval()

    labels = []
    preds = []
    infos = []
    with torch.no_grad():
        epoch_loss = AverageMeter('loss', ':.4e')
        epoch_acc = AverageMeter('acc', ':.3f')
        for image, label, info in tqdm(data_loader, ncols=80):
            #len(data_loader.dataset): dataset size
            #image: batchsize x n x 3 x 244 x 244
            #label: batchsize x n
            #info: [(..., len(batchsize x n))]
            image = image.reshape(-1, image.size(-3), image.size(-2), image.size(-1))
            label = label.reshape(-1)
            image = image.to(device)
            label = label.to(device)
            out = model(image)
            #需要torch gpu
            loss = criterion(out, label)
            acc = accuracy(label, out, topk=(1,))

            label = label.cpu().detach().numpy()  #(36, )
            out = out.cpu().detach().numpy()      #(36, 2)
            
            preds.append(out)
            labels.append(label)
            infos.append(info)
            
            
            if parallel_type == 'Distributed' or parallel_type == 'Distributed_Apex':
                #Distributed_7: 强制同步
                torch.distributed.barrier()
                reduced_loss = reduce_mean(loss, nprocs)
                reduced_acc = reduce_mean(acc[0], nprocs)
                epoch_loss.update(reduced_loss.data.item(), image.size(0))
                epoch_acc.update(reduced_acc, image.size(0))
            elif parallel_type == 'Horovod':
                #各算各的
                reduced_loss = reduce_mean_hvd(loss)
                reduced_acc = reduce_mean_hvd(acc[0])
                epoch_loss.update(reduced_loss.data.item(), image.size(0))
                epoch_acc.update(reduced_acc, image.size(0))
            else:
                epoch_loss.update(loss.data.item(), image.size(0))
                epoch_acc.update(acc[0], image.size(0))

            # epoch_loss.append(loss.data.item() / len(data_loader))
            # epoch_acc.append((out.data.max(1)[1] == label.data).sum().item())


        #梯度  ？
        #Loss   各自
        #Epoch_loss 平均
        #Epoch acc 平均
        #Metic  总 或 平均
        #未修改numpy调用reduce!!!
        # cal total metric
        if parallel_type == 'Distributed' or parallel_type == 'Distributed_Apex':
            torch.distributed.barrier()
            labels_list, preds_list = all_gather_tensor(labels, preds, device, nprocs)
            total_metric = cal_binary_metric(labels_list, preds_list)
            #total_metric = cal_multiclass_metric(labels_list, preds_list)
        elif parallel_type == 'Horovod':
            labels_list, preds_list = all_gather_tensor_hvd(labels, preds, device)
            total_metric = cal_binary_metric(labels_list, preds_list)
            #total_metric = cal_multiclass_metric(labels_list, preds_list)
        else:
            #total_metric = {'auc':1,'bcc':2}
            total_metric = cal_binary_metric(labels, preds)
            #total_metric = cal_multiclass_metric(labels, preds)
        # cal average metric 两个进程取平均值
        average_metric = cal_binary_metric(labels, preds)
        #average_metric = {'auc':1,'bcc':2}
        #average_metric = cal_multiclass_metric(labels, preds)
        if parallel_type == 'Distributed' or parallel_type == 'Distributed_Apex':
            torch.distributed.barrier()
            for k, v in average_metric.items():
                reduced_v = reduce_mean(torch.tensor([v]).to(device), nprocs)  
                average_metric[k] = reduced_v.item()
        elif parallel_type == 'Horovod':
            for k, v in average_metric.items():
                reduced_v = reduce_mean_hvd(torch.tensor([v]).to(device))
                average_metric[k] = reduced_v.item() 
        #print(total_metric)
        #print(average_metric)
        
        #epoch_loss = sum(epoch_loss)
        #accuracy = sum(accuracy) / float(len(data_loader.dataset))
        results = {
            'epoch_loss': epoch_loss.avg,
            'accuracy': epoch_acc.avg.item(),
            'total_score': total_metric['auc'],
            'average_score': average_metric['auc'],
        }
        #results.update(total_metric)
        #for k,v in average_metric.items():
        #    results['avg_'+k] = v
        # epochs x n x 3 x 244 x 244
        return results, (preds, labels, infos), total_metric, average_metric


def test_without_label(current_epoch, model, data_loader, device, criterion):
    model.train(False)
    model.eval()

    infos = []
    preds = []
    with torch.no_grad():
        for image, info in tqdm(data_loader, ncols=80):
            #len(data_loader.dataset): dataset size
            #image: batchsize x n x 3 x 244 x 244, info: [(..., len(batchsize x n))]
            image = image.reshape(-1, image.size(-3), image.size(-2), image.size(-1))
            image = image.to(device)
            out = model(image)

            preds.append(out)
            infos.append(info)

        # epochs x n x 3 x 244 x 244
        return (preds, infos)
