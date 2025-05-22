#!/usr/bin/env python
# 运行 FedCCFA 算法的示例脚本

import os
import argparse

# 确保使用指定的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    parser = argparse.ArgumentParser(description='运行FedCCFA算法')
    
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
    
    # FedCCFA特有参数
    parser.add_argument('--clf_epochs', type=int, default=5,
                      help='分类器训练轮数')
    parser.add_argument('--rep_epochs', type=int, default=5,
                      help='表示层训练轮数')
    parser.add_argument('--balanced_epochs', type=int, default=3,
                      help='平衡训练轮数')
    parser.add_argument('--lambda_proto', type=float, default=0.1,
                      help='原型损失权重')
    parser.add_argument('--eps', type=float, default=0.5,
                      help='DBSCAN聚类的eps参数')
    parser.add_argument('--weights', type=str, default='label', choices=['uniform', 'label'],
                      help='聚合权重方式')
    parser.add_argument('--penalize', type=str, default='L2', choices=['L2', 'contrastive'],
                      help='原型损失类型')
    parser.add_argument('--temperature', type=float, default=0.5,
                      help='对比学习温度参数')
    parser.add_argument('--gamma', type=float, default=0.1,
                      help='自适应原型权重参数')
    parser.add_argument('--oracle', action='store_true',
                      help='使用Oracle合并策略')
    parser.add_argument('--clustered_protos', action='store_true',
                      help='使用聚类原型')
    
    # 概念漂移数据集参数
    parser.add_argument('--use_drift_dataset', action='store_true',
                      help='使用概念漂移数据集')
    parser.add_argument('--drift_data_dir', type=str, default='../dataset/Cifar100_clustered/',
                      help='概念漂移数据集目录')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # 构建命令
    cmd = f"python main.py -data {args.dataset} -m {args.model} -nc {args.num_clients} "\
          f"-nb {args.num_classes} -gr {args.global_rounds} -ls {args.local_epochs} "\
          f"-lbs {args.batch_size} -jr {args.join_ratio} -lr {args.local_learning_rate} "\
          f"-algo FedCCFA -cle {args.clf_epochs} -rpe {args.rep_epochs} -be {args.balanced_epochs} "\
          f"-lp {args.lambda_proto} -eps {args.eps} -wts {args.weights} -pnz {args.penalize} "\
          f"-tmp {args.temperature} -gm {args.gamma}"
    
    # 添加布尔参数
    if args.oracle:
        cmd += " -orc"
    if args.clustered_protos:
        cmd += " -cp"
    if args.use_drift_dataset:
        cmd += f" --use_drift_dataset --drift_data_dir {args.drift_data_dir}"
    
    # 输出并执行命令
    print(f"执行命令: {cmd}")
    os.system(cmd)
