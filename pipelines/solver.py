import apex
from apex import amp
import os
import horovod.torch as hvd
import time 
from tensorboardX import SummaryWriter
import torch
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
import torch.nn.functional as F



from dataset.data_loader import get_loader, get_infer_image
from networks.losses import get_loss
from networks.networks import build_model
from networks.opt import adjust_learning_rate, get_optimizer, get_schedule
from pipelines.train import train_epoch
from pipelines.test import test_epoch, test_without_label
from tools.utils import init_seed, record_epoch, load_model, save_model, write_result_csv


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # 按照PCI_BUS_ID顺序从0开始排列GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

#@profile
class Solver(object):
    @profile
    def __init__(self, config):
        # seed
        if config.seed is not None:
            init_seed(config.seed)
        else:
            torch.backends.cudnn.benchmark = True
        
        self.nprocs = None

        # save
        for k, v in config.save.items():
            if k != 'result_path':
                config.save[k] = config.save['result_path'] + '/' +  v
                
        #Distributed_2: 初始化进程组, 设置batchsize
        if config.parallel_type == 'Distributed' or config.parallel_type == 'Distributed_Apex':
            #没有 torch.distributed.launch 读取的默认环境变量作为配置，我们需要手动为 init_process_group 指定参数
            #torch.distributed.init_process_group(backend='nccl')
            torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', world_size=config.world_size, rank=config.local_rank)
            self.nprocs = config.world_size
            config.model_params[config.model_type]['batch_size'] = int(config.model_params[config.model_type]['batch_size']  / self.nprocs)
            print("local_rank: {}, nprocs: {}".format(config.local_rank,self.nprocs))
        elif config.parallel_type == 'Horovod':
            self.nprocs = config.world_size
            config.model_params[config.model_type]['batch_size'] = int(config.model_params[config.model_type]['batch_size']  / self.nprocs)
            print("local_rank: {}, nprocs: {}".format(config.local_rank,self.nprocs))
     
        # gpu or cpu
        if config.gpu and torch.cuda.is_available():
            if config.parallel_type == 'Single' or config.parallel_type == 'DataParallel':
                self.device = torch.device('cuda:'+str(config.gpu[0]))
                torch.cuda.set_device('cuda:'+str(config.gpu[0]))
            #Distributed_3: 设置gpu
            if config.parallel_type == 'Distributed' or config.parallel_type == 'Distributed_Apex' or config.parallel_type == 'Horovod':
                self.device = torch.device('cuda:'+str(config.local_rank))
                torch.cuda.set_device('cuda:'+str(config.local_rank))
        else:
            self.device = torch.device('cpu')
        
        self.best_score = 0.0
        self.start_epoch = 0

        # model
        self.model = build_model(config.model_type, config.model_params[config.model_type], config.pretrained)
        if config.is_load_model:
            #??? 分布式需要加载几次模型
            print("=> load model: {}".format(config.load['load_model_path']))
            pre_epoch, pre_best = load_model(self.model, config.load['load_model_path'], self.device)
            if config.from_interruption:
                print("=> from_interruption: epoch:{} best_acc:{}".format(pre_epoch, pre_best))
                self.start_epoch = pre_epoch
                self.best_score = pre_best      
            
        self.model.to(self.device)
       
        # Data loader
        self.train_loader, self.valid_loader, self.test_loader = self.get_data_loaders(
            config)

        # criterion & optimizer
        #Distributed_4: 为了进一步加快训练速度，可以把损失函数也进行分布式计算???
        if config.parallel_type == 'Distributed' or config.parallel_type == 'Distributed_Apex' or config.parallel_type == 'Horovod':
            self.criterion = get_loss(config.loss_type).cuda(config.local_rank)
        else:
            self.criterion = get_loss(config.loss_type)
        
        if config.parallel_type == 'Horovod':
            self.optimizer = get_optimizer(self.model, config.optimizer_type, config.optimizer_params[config.optimizer_type])
            hvd.broadcast_optimizer_state(self.optimizer, root_rank=0) #？？？
            compression = hvd.Compression.fp16
            self.optimizer = hvd.DistributedOptimizer(self.optimizer, named_parameters=self.model.named_parameters(), compression=compression)
        else:
            self.optimizer = get_optimizer(self.model, config.optimizer_type, config.optimizer_params[config.optimizer_type])
            # ???
            # self.optimizer = torch.nn.DataParallel(self.optimizer, device_ids=config.gpu)
        if config.schedule_type == 'no':
            self.schedule_params = None
            self.scheduler = None
        else:
            self.schedule_params = config.schedule_params[config.schedule_type]
            if config.schedule_type == 'step':
                schedule_params['params']['max_iter'] = len(data_loader)
            self.scheduler = get_schedule(self.optimizer, config.schedule_type, self.schedule_params)


        if config.use_amp:
            self.scalar = GradScaler()
        else:
            self.scalar = None

        # model paralle
        if torch.cuda.device_count() > 1:
            print("=> use parallel type: {}".format(config.parallel_type))
            if config.parallel_type == 'DataParallel':
                self.model = torch.nn.DataParallel(
                    self.model, device_ids=config.gpu,  output_device=config.gpu[0])
            #Distributed_5: 分布式训练需要将bn换成sync_batchnorm进行多卡同步
            if config.parallel_type == 'Distributed':
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)  #???
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[config.local_rank])
            if config.parallel_type == 'Distributed_Apex':
                self.model = apex.parallel.convert_syncbn_model(self.model)
                self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")
                self.model = apex.parallel.DistributedDataParallel(self.model, delay_allreduce=True) #增加时间???
            if config.parallel_type == 'Horovod':
                hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        
        # log
        self.summary_writer = SummaryWriter(log_dir=config.save['log_path'])




    def get_data_loaders(self, config):
        dataset_params = config.datasets[config.data_type]
        model_params = config.model_params[config.model_type]
        train_loader = None
        valid_loader = None
        test_loader = None
        if config.mode == 'train':
            train_loader = get_loader(data_list_path=dataset_params['train_list_path'],
                                      batch_size=model_params['batch_size'],
                                      shuffle=True,
                                      num_workers=config.workers,
                                      mode='train',
                                      balance=True,
                                      model_params=model_params,
                                      dataset_params=dataset_params,
                                      parallel_type=config.parallel_type)
            valid_loader = get_loader(data_list_path=dataset_params['valid_list_path'],
                                      batch_size=model_params['batch_size'],
                                      shuffle=True,
                                      num_workers=config.workers,
                                      mode='valid',
                                      balance=False,
                                      model_params=model_params,
                                      dataset_params=dataset_params,
                                      parallel_type=config.parallel_type)
        elif config.mode == 'test':
            test_loader = get_loader(data_list_path=dataset_params['test_list_path'],
                                     batch_size=model_params['batch_size'],
                                     shuffle=False,
                                     num_workers=config.workers,
                                     mode='test',
                                     balance=False,
                                     model_params=model_params,
                                     dataset_params=dataset_params,
                                     parallel_type=config.parallel_type)
        return train_loader, valid_loader, test_loader
    @profile
    def train_and_valid(self, config):
        total_time = 0.0
        total_epoch = 0
        for epoch in range(self.start_epoch, config.model_params[config.model_type]['num_epochs']):
            if config.local_rank == 0:
                print("\nepochs: {}, best_score: {}".format(epoch, self.best_score))
            epoch_start_time = time.time()
            
            # ---lr
            #adjust_learning_rate(self.optimizer, epoch, config.optimizer_params[config.optimizer_type]['lr'])

            # ---train
            if config.parallel_type == 'Distributed' or config.parallel_type == 'Distributed_Apex' or config.parallel_type == 'Horovod':
                # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果
                self.train_loader.sampler.set_epoch(epoch)
            train_record, _, train_total_metric, train_average_metric = train_epoch(
                epoch, self.model, self.train_loader, self.device, self.criterion, self.optimizer, self.scheduler, self.schedule_params, config.parallel_type, self.nprocs, self.scalar, config.use_amp, self.summary_writer)
            if config.local_rank == 0:
                for k, v in train_record.items():
                    self.summary_writer.add_scalar('train/'+k, v, epoch)
                for k, v in train_total_metric.items():
                    self.summary_writer.add_scalar('train_total_metirc/'+k, v, epoch)
                for k, v in train_average_metric.items():
                    self.summary_writer.add_scalar('train_average_metric/'+k, v, epoch)
                
                train_record.update(train_total_metric)
                for k,v in train_average_metric.items():
                    train_record['avg_'+k] = v
                record_epoch(mode='train', epoch=epoch, total_epoch=config.model_params[config.model_type]['num_epochs'],
                         record=train_record, record_path=config.save['train_checkpoint_file'])    
              

            # ---valid
            valid_record, _, valid_total_metric, valid_average_metric = test_epoch(
                epoch, self.model, self.valid_loader, self.device, self.criterion, config.parallel_type, self.nprocs)
            if config.local_rank == 0:
                for k, v in valid_record.items():
                    self.summary_writer.add_scalar('valid/'+k, v, epoch)
                for k, v in valid_total_metric.items():
                    self.summary_writer.add_scalar('valid_total_metirc/'+k, v, epoch)
                for k, v in valid_average_metric.items():
                    self.summary_writer.add_scalar('valid_average_metric/'+k, v, epoch)
                
                valid_record.update(valid_total_metric)
                for k,v in valid_average_metric.items():
                    valid_record['avg_'+k] = v
                record_epoch(mode='valid', epoch=epoch, total_epoch=config.model_params[config.model_type]['num_epochs'],
                         record=valid_record, record_path=config.save['valid_checkpoint_file'])
              
            
            # --- cal time
            epoch_end_time = time.time()
            total_time += epoch_end_time - epoch_start_time
            total_epoch += 1
            if config.local_rank == 0:
                print('\ntime cost: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch_start_time)), epoch_end_time - epoch_start_time)

            # ---save
            if config.local_rank == 0:
                state = {
                    'epoch': epoch,
                    'config': config,
                    'best_score': max(self.best_score, valid_record['total_score']),
                    'state_dict': self.model.state_dict() if config.parallel_type == 'Single' or config.parallel_type == 'Horovod' else self.model.module.state_dict()
                }
                if valid_record['total_score'] > self.best_score:
                    print("Find Better Model")
                    save_model(epoch, state, config.model_type,
                           config.save['save_model_path'], isBetter=True)
                    self.best_score = valid_record['total_score']

                if (epoch) % config.save_freq == 0:
                    save_model(epoch, state, config.model_type,
                           config.save['save_model_path'], isBetter=False)
        self.summary_writer.close()
        if config.local_rank == 0:
            print('\ntotal time cost: {}, total_epoch: {}, average time cost: {}, best score: {}'.format(total_time, total_epoch, total_time/total_epoch, self.best_score))
 

    def test(self, config):
        test_record, test_result = test_epoch(
            0, self.model, self.test_loader, self.device, self.criterion, config.parallel_type, self.nprocs)
        if config.local_rank == 0:
            record_epoch(mode='test', epoch=1, total_epoch=1,
                     record=test_record, record_path=config.save['test_checkpoint_file'])
        #label: [batchsize x n]x(total/batchsize) 
        #pred: [batchsize x n x 3]x(total/batchsize) 
        #info: [(..., len(batchsize x n)]x(total/batchsize)
            datas = []
            for preds, labels, infos in zip(test_result[0], test_result[1], test_result[2]):
                for i in range(len(infos[0])): #shape: 1, 1, 3
                #print(preds[i].data)
                    data = [infos[0][i], labels[i].data.item(), preds[i].data.max(0)[1].item()]
                    datas.append(data)
            write_result_csv(config.save['test_ans_file'], datas)

    def infer(self, infer_path, config, mode='single'):
        dataset_params = config.datasets[config.data_type]
        model_params = config.model_params[config.model_type]
        if mode == 'single':
            self.model.train(False)
            self.model.eval()
            with torch.no_grad():
                image = get_infer_image(infer_path)
                image = image.to(self.device)
                out = self.model(image)
                #out: 1x3
                out = F.softmax(out, dim=1) #归一化
                #最高概率对应的label, 每个label对应概率
                print(out[0].data.max(0)[1].item()) 
                print(out.data)
                return out[0].data.max(0)[1].item(), out[:, 0].data.item(), out[:, 1].data.item(), out[:, 2].data.item()
        else:
            if mode == 'list':
                dataset_params['infer_list_path'] = infer_path
            elif mode == 'fold':
                from preprocessing.process_dataset import write_dataset_list_without_label
                infer_list_path = './tmp.txt'
                write_dataset_list_without_label(infer_path,  infer_list_path)
                dataset_params['infer_list_path'] = infer_list_path
            infer_loader = get_loader(data_list_path=dataset_params['infer_list_path'],
                                      batch_size=model_params['batch_size'],
                                      shuffle=False,
                                      num_workers=config.workers,
                                      mode='infer',
                                      balance=False,
                                      model_params=model_params,
                                      dataset_params=dataset_params)
            infer_result = test_without_label(
                0, self.model, infer_loader, self.device, self.criterion)
            #pred: [(batchsize x n) x 3]x(total/batchsize) 
            #info: [(..., len(batchsize x n)]x(total/batchsize)
            datas = []
            for preds, infos in zip(infer_result[0], infer_result[1]):
                for i in range(len(infos[0])): #shape: 1, 1, 3
                    out = F.softmax(preds[i])
                    data = out.data.tolist() #out = out[:,0:3].data.cpu().numpy().tolist()
                    datas.append(data)
            write_result_csv(config.save['infer_ans_file'], datas)
            print('infer done')
            return datas
