# 框架
* config
    * config.yaml 训练测试推理配置文件
    * create_dataset_config.yaml 数据预处理配置文件
* dataset
    * 加载数据相关
* networks
    * networks.py 加载模型和优化器
    * opt.py 优化器相关
* pipelines
    * solver.py 主要处理文件
    * train.py 训练一代
    * test.py 验证一代
* preprocessing
    * process_dataset.py 数据预处理
* results
    * log 存放log
    * models 存储模型
* tools
    * custom.py 自定义类
    * eval.py 计算score相关
    * utils.py 杂乱的自定义函数工具库
* weights 
    * 加载其他权重或保存最优的权重
* main.py 启动函数

# 使用 
## 步骤1 数据预处理
1. 数据处理配置文件configs/create_dataset_config.yaml
2. 执行：在preprocessing/process_dataset.py调用对应的函数
* process_file（划分数据集）
* process_list（只划分数据集list）

## 步骤2 训练和测试
1. 配置文件configs/config.yaml
2. 执行：main.py
    ```
    mode: 'train' or 'test'
    ```

### 并行训练 
#### 1.单卡
* 设置
    ```
    os.environ["CUDA_VISIBLE_DEVICES"]   = "2,3"
    gpu: [0,1]
    local_rank: 0 #必须
    parallel_type: 'Single'
    main(args.local_rank, argparse.Namespace(**config))
    ```
* 执行
    ```
    python main.py
    ```
 #### 2.DataParallel
* 设置
    ```
    os.environ["CUDA_VISIBLE_DEVICES"]   = "2,3"
    gpu: [0,1]
    local_rank: 0 #必须
    parallel_type: 'DataParallel'
    main(args.local_rank, argparse.Namespace(**config))
    ```
* 执行
    ```
    python main.py
    ``` 
 #### 3.Distributed 和 Distributed_Apex
* 设置1 torch.distributed.launch启动
    ```
    os.environ["CUDA_VISIBLE_DEVICES"]   = "2,3"
    world_size: 2 #和nproc_per_node保持一致
    parallel_type: 'Distributed' 或 'Distributed_Apex'
    main(args.local_rank, argparse.Namespace(**config))
    ```
* 执行1
    ```
    python -m torch.distributed.launch --nproc_per_node=2  main.py
    ``` 
* 设置2 spawn启动
    ```
    os.environ["CUDA_VISIBLE_DEVICES"]   = "2,3"
    world_size: 2 #和nproc_per_node保持一致
    parallel_type: 'Distributed' 或 'Distributed_Apex'
    mp.spawn(main, nprocs=config['world_size'], args=(argparse.Namespace(**config),))
    ```
* 执行2
    ```
    python main.py
    ```

 #### 4.Horovod 
* 设置1 torch.distributed.launch启动
    ```
    os.environ["CUDA_VISIBLE_DEVICES"]   = "2,3"
    world_size: 2 #和np保持一致
    parallel_type: 'Horovod' 
    hvd.init()
    main(hvd.local_rank(), argparse.Namespace(**config))
    ```
* 执行
    ```
    HOROVOD_WITH_PYTORCH=1 horovodrun -np 2 -H localhost:4 --verbose python main.py
    ``` 

## 步骤3 推理
1. 配置文件configs/config.yaml
2. 执行：main.py
    ```
    mode: 'infer'
    ```
3. 三种读取数据方式
    ```
    solver.infer(infer_path_single, config, mode='single')
    solver.infer(infer_path_list, config, mode='list')
    solver.infer(infer_path_fold, config, mode='fold')
    ```