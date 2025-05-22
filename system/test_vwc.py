import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.improved_vwc_clustering import VariationalWassersteinClustering

def test_vwc_clustering():
    print("测试改进后的VWC聚类算法...")
    
    # 测试参数
    num_clients = 20
    num_samples = 10
    proxy_dim = 32
    num_clusters = 5
    
    # 生成随机测试数据，但使其有一定的聚类结构
    cluster_centers = torch.randn(num_clusters, proxy_dim) * 2  # 定义簇中心
    
    # 将数据点分配到各个簇周围
    assignments = np.random.randint(0, num_clusters, num_clients)
    print(f"真实簇分配: {np.bincount(assignments, minlength=num_clusters).tolist()}")
    
    # 为每个客户端创建数据点
    proxy_points = []
    for client_idx in range(num_clients):
        # 从分配的簇中心生成数据点
        center = cluster_centers[assignments[client_idx]]
        # 添加噪声
        client_data = center.unsqueeze(0) + torch.randn(num_samples, proxy_dim) * 0.5
        proxy_points.append(client_data)
    
    # 将数据堆叠成批处理
    proxy_points = torch.stack(proxy_points)
    
    # 初始化并测试标准VWC
    print("\n==== 测试标准VWC ====")
    standard_vwc = VariationalWassersteinClustering(
        num_clients=num_clients, 
        num_clusters=num_clusters, 
        proxy_dim=proxy_dim,
        sinkhorn_reg=0.01
    )
    
    # 训练标准VWC
    standard_assignments = standard_vwc.fit(proxy_points, max_iter=50)
    standard_counts = torch.bincount(standard_assignments, minlength=num_clusters).tolist()
    print(f"标准VWC簇分配: {standard_counts}")
    
    # 初始化并测试改进的VWC
    print("\n==== 测试改进的VWC ====")
    improved_vwc = VariationalWassersteinClustering(
        num_clients=num_clients, 
        num_clusters=num_clusters, 
        proxy_dim=proxy_dim,
        sinkhorn_reg=0.1,  # 增加熵正则化参数
        temperature=0.5    # 添加温度参数
    )
    
    # 训练改进的VWC
    improved_assignments = improved_vwc.fit(proxy_points, max_iter=100)
    improved_counts = torch.bincount(improved_assignments, minlength=num_clusters).tolist()
    print(f"改进VWC簇分配: {improved_counts}")
    
    # 比较结果可视化
    plt.figure(figsize=(12, 5))
    
    # 绘制标准VWC分配
    plt.subplot(1, 2, 1)
    plt.bar(range(num_clusters), standard_counts)
    plt.title('标准VWC聚类分配')
    plt.xlabel('簇索引')
    plt.ylabel('客户端数量')
    for i, count in enumerate(standard_counts):
        plt.text(i, count, str(count), ha='center')
    
    # 绘制改进的VWC分配
    plt.subplot(1, 2, 2)
    plt.bar(range(num_clusters), improved_counts)
    plt.title('改进VWC聚类分配')
    plt.xlabel('簇索引')
    plt.ylabel('客户端数量')
    for i, count in enumerate(improved_counts):
        plt.text(i, count, str(count), ha='center')
    
    plt.tight_layout()
    plt.savefig('results/vwc_comparison.png')
    print(f"对比结果已保存到 'results/vwc_comparison.png'")
    plt.close()
    
    return improved_vwc, standard_vwc

if __name__ == "__main__":
    test_vwc_clustering()
