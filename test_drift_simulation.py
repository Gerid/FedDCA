import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 添加系统路径，以便导入自定义模块
sys.path.append('d:\\repos\\PFL-Non-IID')
from system.main import *
from system.flcore.clients.clientdca import clientDCA

def get_args():
    """构建测试用的命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cifar100', 
                        help='数据集名称: Cifar100')
    parser.add_argument('--algorithm', type=str, default='FedDCA', 
                        help='联邦学习算法: FedDCA')
    parser.add_argument('--model', type=str, default='cnn', 
                        help='神经网络模型类型')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='训练时的批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.01, 
                        help='全局学习率')
    parser.add_argument('--local_learning_rate', type=float, default=0.05, 
                        help='本地学习率')
    parser.add_argument('--learning_rate_decay_gamma', type=float, default=0.99, 
                        help='学习率衰减率')
    parser.add_argument('--beta', type=float, default=0.5, 
                        help='动量参数 beta')
    parser.add_argument('--alpha', type=float, default=0.5, 
                        help='FedDCA算法中的alpha参数')
    parser.add_argument('--local_epochs', type=int, default=1, 
                        help='本地训练轮数')
    parser.add_argument('--num_clients', type=int, default=5, 
                        help='客户端数量')
    parser.add_argument('--num_clusters', type=int, default=5, 
                        help='集群数量')
    parser.add_argument('--frac', type=float, default=1.0, 
                        help='每轮参与训练的客户端比例')
    parser.add_argument('--epochs', type=int, default=300, 
                        help='全局训练轮数')
    parser.add_argument('--times', type=int, default=1, 
                        help='运行次数')
    parser.add_argument('--eval_gap', type=int, default=1, 
                        help='评估间隔')
    parser.add_argument('--device', type=str, default='cuda', 
                        help='计算设备: cuda 或 cpu')
    parser.add_argument('--simulate_drift', type=bool, default=True, 
                        help='是否在标准数据集上模拟概念漂移')
    parser.add_argument('--increment_iteration', type=bool, default=True, 
                        help='每轮训练是否递增迭代次数')
    parser.add_argument('--max_iterations', type=int, default=350, 
                        help='最大迭代次数')
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子')
    
    args = parser.parse_args([])  # 使用空列表，因为在脚本中运行而不是命令行
    return args

def test_single_client_drift():
    """测试单个客户端的概念漂移模拟"""
    args = get_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 加载模型
    model = select_model(args)
    
    # 创建一个虚拟客户端
    client = clientDCA(args, 0, 100, 100)
    client.model = model
    client.simulate_drift = True
    client.initialize_drift_patterns()
    
    # 测试不同迭代下的数据变化
    iterations_to_test = [0, 60, 120, 180, 240, 320]
    
    plt.figure(figsize=(20, 10))
    for idx, iteration in enumerate(iterations_to_test):
        client.current_iteration = iteration
        
        # 加载训练数据
        train_loader = client.load_train_data(batch_size=64)
        
        # 获取一个批次的数据
        images, labels = next(iter(train_loader))
        
        # 如果需要，转换图像以便显示
        if images.shape[1] == 3:  # RGB图像
            # 创建网格显示
            pattern_name = "No drift"
            for schedule in client.drift_schedule:
                if schedule['iterations'][0] <= iteration < schedule['iterations'][1]:
                    pattern_name = schedule['pattern'] or "No drift"
                    
            plt.subplot(2, 3, idx + 1)
            if len(images) >= 16:
                grid_size = 4
                for i in range(grid_size * grid_size):
                    plt.subplot(2, 3, idx + 1)
                    # 在当前子图中创建更细分的网格
                    plt.subplot2grid((2*grid_size, 3*grid_size), 
                                   (i // grid_size + grid_size * (idx // 3), 
                                    i % grid_size + grid_size * (idx % 3)), 
                                   colspan=1, rowspan=1)
                    
                    img = images[i].permute(1, 2, 0).cpu().numpy()
                    # 归一化以便显示
                    img = (img - img.min()) / (img.max() - img.min())
                    plt.imshow(img)
                    plt.title(f"Label: {labels[i]}")
                    plt.axis('off')
            
            plt.subplot(2, 3, idx + 1)
            plt.title(f"Iteration {iteration}: {pattern_name}")
    
    plt.tight_layout()
    plt.savefig('drift_simulation_visualization.png')
    plt.close()
    print(f"保存可视化结果到 drift_simulation_visualization.png")
    
    return "漂移模拟测试完成"

def analyze_model_performance():
    """分析在模拟漂移数据上的模型性能"""
    args = get_args()
    
    # 设置随机种子  
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 修改部分参数
    args.epochs = 350  # 增加总轮数以覆盖所有漂移阶段
    args.eval_gap = 1  # 每轮都评估
    args.simulate_drift = True
    
    # 运行模拟
    run_experiment(args)
    
    # 结果将保存在results目录下
    print("FedDCA实验结束，结果保存在results目录中")

if __name__ == "__main__":
    # 测试单个客户端的漂移模拟
    test_single_client_drift()
    
    # 分析模型性能
    # 注意：此操作将运行完整实验，可能需要较长时间
    # analyze_model_performance()
