import argparse
import horovod.torch as hvd
import torch.multiprocessing as mp
import yaml

from pipelines.solver import Solver

#@profile
def main(local_rank, config):
    print("local_rank:", local_rank)
    config.local_rank = local_rank
    #print(config)
    solver = Solver(config)
    if config.mode == 'train':
        solver.train_and_valid(config)
    elif config.mode == 'test':
        solver.test(config)
    elif config.mode == 'infer':
        #infer_path_single = '/data_activate/kezhiying/testpj/ori/fake/malefamouspeoplephoto_96.png'
        #infer_path_fold = config.datasets[config.data_type]['infer_path']
        infer_path_list = config.datasets[config.data_type]['infer_list_path']
        #solver.infer(infer_path_single, config, mode='single')
        solver.infer(infer_path_list, config, mode='list')
        #solver.infer(infer_path_fold, config, mode='fold')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,
                        default='./configs/config.yaml')
    #Distributed_1: local_rank代表当前进程，分布式启动会会自动更新该参数,不需要在命令行显式调用
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = yaml.load(f)
    
    #single, DataParallel, Distributed, Distributed_Apex
    main(args.local_rank, argparse.Namespace(**config))
    
    #Distributed, Distributed_Apex
    #mp.spawn(main, nprocs=config['world_size'], args=(argparse.Namespace(**config),))
    
    #Horovod 
    #hvd.init()
    #main(hvd.local_rank(), argparse.Namespace(**config))
    
