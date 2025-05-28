#!/usr/bin/env python
# 运行 Flash 算法的示例脚本

import os
import argparse

# 确保使用指定的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    parser = argparse.ArgumentParser(description='运行Flash算法')
    
    # 基本参数
    parser.add_argument('--dataset', type=str, default='Cifar10', choices=['mnist', 'fmnist', 'Cifar10', 'Cifar100'],
                      help='要使用的数据集')
    parser.add_argument('--model', type=str, default='cnn', choices=['mlr', 'cnn', 'dnn', 'resnet', 'alexnet'],
                      help='要使用的模型架构')
    parser.add_argument('--num_clients', type=int, default=20,
                      help='客户端总数量')
    parser.add_argument('--num_classes', type=int, default=10,
                      help='类别数量')
    parser.add_argument('--global_rounds', type=int, default=300,
                      help='全局训练轮数')
    parser.add_argument('--local_epochs', type=int, default=5,
                      help='本地训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='批量大小')
    parser.add_argument('--join_ratio', type=float, default=0.5,
                      help='每轮参与的客户端比例')
    parser.add_argument('--local_learning_rate', type=float, default=0.01,
                      help='本地学习率')
    
    # Flash特有参数
    parser.add_argument('--loss_decrement', type=float, default=0.01,
                      help='早停损失下降阈值')
    parser.add_argument('--beta1', type=float, default=0.9,
                      help='一阶动量系数')
    parser.add_argument('--beta2', type=float, default=0.99,
                      help='二阶动量系数')
    parser.add_argument('--tau', type=float, default=1e-8,
                      help='数值稳定常数')
    parser.add_argument('--server_learning_rate', type=float, default=1.0,
                      help='服务器学习率')
    
    # 概念漂移数据集参数
    parser.add_argument('--use_drift_dataset', action='store_true',
                      help='使用概念漂移数据集')
    parser.add_argument('--drift_data_dir', type=str, default='../dataset/Cifar100_clustered/',
                      help='概念漂移数据集目录')
    parser.add_argument('--max_iterations', type=int, default=200,
                      help='概念漂移数据集的最大迭代数')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # 构建命令
    cmd = f"python main.py -data {args.dataset} -m {args.model} -nc {args.num_clients} "\
          f"-nb {args.num_classes} -gr {args.global_rounds} -ls {args.local_epochs} "\
          f"-lbs {args.batch_size} -jr {args.join_ratio} -lr {args.local_learning_rate} "\
          f"-algo Flash"
    
    # 添加Flash特有参数
    cmd += f" --loss_decrement {args.loss_decrement} --beta1 {args.beta1} "\
           f"--beta2 {args.beta2} --tau {args.tau} --server_learning_rate {args.server_learning_rate}"
    
    # 添加概念漂移数据集参数
    if args.use_drift_dataset:
        cmd += f" --use_drift_dataset --drift_data_dir {args.drift_data_dir} "\
               f"--max_iterations {args.max_iterations}"
    
    # 输出并执行命令
    print(f"执行命令: {cmd}")
    os.system(cmd)
