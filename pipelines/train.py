from apex import amp
import torch
from torch.cuda.amp import autocast as autocast
from tqdm import tqdm
from tools.custom import AverageMeter
from tools.eval import accuracy, cal_multiclass_metric, cal_binary_metric
from tools.utils import reduce_mean, reduce_mean_hvd, all_gather_tensor, all_gather_tensor_hvd
from networks.losses import cal_balance_loss

@profile
def train_epoch(current_epoch, model, data_loader, device, criterion, optimizer, scheduler, schedule_params, parallel_type, nprocs, scaler, use_amp, summary_writer):
    model.train(True)
    labels = []
    infos = []
    preds = []
    with torch.enable_grad():
        epoch_loss = AverageMeter('loss', ':.4e')
        epoch_acc = AverageMeter('acc', ':.3f')
        total_step = len(data_loader)
        for i, (image, label, info) in enumerate(tqdm(data_loader, ncols=80)):
            #len(data_loader.dataset): dataset size
            #image: batchsize x n x 3 x 244 x 244
            #label: batchsize x n
            #info: [(..., len(batchsize x n))]
            image = image.reshape(-1, image.size(-3), image.size(-2), image.size(-1))
            label = label.reshape(-1)
            image = image.to(device)
            label = label.to(device)
            
            if use_amp:# 前向过程(model + loss)开启 autocast
                with autocast(): 
                    out = model(image)
                    #loss = criterion(out, label)
                    loss = cal_balance_loss(criterion, out, label)
            else: 
                out = model(image)
                #loss = criterion(out, label)
                loss = cal_balance_loss(criterion, out, label)

            preds.append(out)
            infos.append(infos)
            labels.append(label)
                           
            
            acc = accuracy(label, out, topk=(1,))

            if parallel_type == 'Distributed' or parallel_type == 'Distributed_Apex':
                #Distributed_7: 强制同步
                torch.distributed.barrier() #每个进程进入这个函数后都会被阻塞，当所有进程都进入这个函数后，阻塞解除
                reduced_loss = reduce_mean(loss, nprocs)
                reduced_acc = reduce_mean(acc[0], nprocs)
                epoch_loss.update(reduced_loss.data.item(), image.size(0))
                epoch_acc.update(reduced_acc, image.size(0))
            elif parallel_type == 'Horovod':
                reduced_loss = reduce_mean_hvd(loss)
                reduced_acc = reduce_mean_hvd(acc[0])
                epoch_loss.update(reduced_loss.data.item(), image.size(0))
                epoch_acc.update(reduced_acc, image.size(0))
            else:
                epoch_loss.update(loss.data.item(), image.size(0))
                epoch_acc.update(acc[0], image.size(0))

            #epoch_loss.append(loss.data.item() / len(data_loader))
            # epoch_acc.append((out.data.max(1)[1] == label.data).sum().item())
            #print("s1:", loss.item())
            optimizer.zero_grad()
            #计算出每个参数的梯度，并存储在parameter.grad中
            if parallel_type == 'Distributed_Apex':
                #反向传播时需要调用 amp.scale_loss，用于根据loss值自动对精度进行缩放
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            elif use_amp:  # Scales loss. 为了梯度放大.
                scaler.scale(loss).backward()
            else:
                loss.backward()
            #print("s2:", loss.item())

            if use_amp:
                # scaler.step() 首先把梯度的值unscale回来.
                # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
                # 否则，忽略step调用，从而保证权重不更新（不被破坏）
                scaler.step(optimizer)
                # 准备着，看是否要增大scaler
                scaler.update()
            else:
                optimizer.step()

            if scheduler is not None:
                if schedule_params['mode'] == 'step':
                    if schedule_params['type'] == 'poly':
                        schedule_params['params']['max_iter'] = total_step
                        scheduler.step(i + current_epoch * schedule_params['params']['max_iter'])
                        #if i == schedule_params['params']['max_iter'] - 1:
                        #    print("i: ", i)
                        #    break
                    elif schedule_params['type'] == 'cosineAnnWarm':
                        scheduler.step(current_epoch + i / total_step)
                summary_writer.add_scalar('train_step/lr_scheduler', float(scheduler.get_lr()[-1]), i + current_epoch * total_step)
                summary_writer.add_scalar('train_step/lr_optimizer', optimizer.param_groups[0]['lr'], i + current_epoch * total_step)
                
        if scheduler is not None:  
            if schedule_params['mode'] == 'epoch':
                if schedule_params['type'] == 'poly':
                    scheduler.step()
                elif schedule_params['type'] == 'cosineAnnWarm':
                    scheduler.step()
        #梯度  ？
        #Loss   各自
        #Epoch_loss 平均
        #Epoch acc 平均
        #Metic  总 或 平均
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
            total_metric = cal_binary_metric(labels, preds)
            #total_metric = {'auc':1, 'acc':2}
            #total_metric = cal_multiclass_metric(labels, preds)
        # cal average metric 两个进程取平均值
        average_metric = cal_binary_metric(labels, preds)
        #average_metric = {'auc':1, 'acc':2}
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
            'lr': optimizer.param_groups[0]['lr'],
            'accuracy': epoch_acc.avg.item(),
            'total_score': total_metric['auc'],
            'average_score': average_metric['auc'],
        }
        #results.update(total_metric)
        #for k,v in average_metric.items():
        #    results['avg_'+k] = v
        # epochs x n x 3 x 244 x 244
        return results, (preds, labels, infos), total_metric, average_metric
