"""
使用增强型标签条件聚类的联邦学习示例
此脚本展示了如何使用结合预测标签和中间表征向量的增强型聚类方法
"""

import argparse
import os
import sys
from pathlib import Path

# 确保系统模块可被导入
sys.path.append(str(Path(__file__).parent.absolute()))

def run_enhanced_clustering():
    parser = argparse.ArgumentParser()

    # 基础设置
    parser.add_argument('--dataset', type=str, default='Cifar100')
    parser.add_argument('--model', type=str, default='cnn')
    parser.add_argument('--algorithm', type=str, default='FedDCA')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--local_learning_rate', type=float, default=0.01)
    parser.add_argument('--personal_learning_rate', type=float, default=0.01)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--learning_rate_decay', type=float, default=0.995)
    parser.add_argument('--learning_rate_decay_gamma', type=float, default=0.998)
    parser.add_argument('--lamda', type=float, default=1.0)
    parser.add_argument('--num_global_iters', type=int, default=200)
    parser.add_argument('--local_epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--save_path', type=str, default='results')
    parser.add_argument('--comms_round', type=int, default=200)
    parser.add_argument('--times', type=int, default=1)
    parser.add_argument('--auto_break', type=bool, default=False)

    # 聚类相关设置
    parser.add_argument('--clustering_method', type=str, default='enhanced_label',
                        help='聚类方法: vwc, label_conditional, enhanced_label')
    parser.add_argument('--num_clusters', type=int, default=5,
                        help='聚类数量')
    parser.add_argument('--kde_samples', type=int, default=100,
                        help='每个标签的KDE采样数量')

    # 概念漂移相关设置
    parser.add_argument('--use_drift_dataset', type=bool, default=True,
                        help='是否使用概念漂移数据集')
    parser.add_argument('--drift_data_dir', type=str, default='dataset/Cifar100_clustered',
                        help='概念漂移数据集路径')
    parser.add_argument('--max_iterations', type=int, default=200,
                        help='最大迭代次数')
    parser.add_argument('--drift_threshold', type=float, default=0.2,
                        help='漂移检测阈值')
    parser.add_argument('--split_threshold', type=float, default=0.3,
                        help='集群分裂阈值')
    parser.add_argument('--merge_threshold', type=float, default=0.1,
                        help='集群合并阈值')

    # 其他设置
    parser.add_argument('--verbose', type=bool, default=True,
                        help='是否打印详细信息')
    parser.add_argument('--proxy_dim', type=int, default=128,
                        help='代理特征维度')
    parser.add_argument('--sinkhorn_reg', type=float, default=0.01,
                        help='Sinkhorn正则化参数')

    args = parser.parse_args()

    print("启动使用增强型标签条件聚类的联邦学习...")
    print("聚类方法:", args.clustering_method)
    print("概念漂移数据集:", args.use_drift_dataset)
    print("漂移数据集路径:", args.drift_data_dir)

    # 导入主系统模块
    from system.main import main

    # 运行主程序
    main(args)

if __name__ == "__main__":
    run_enhanced_clustering()
