import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def evaluate_clustering(cluster_assignments, true_concepts):
    """
    评估聚类结果与真实概念的一致性
    
    参数:
        cluster_assignments: 字典，将客户端ID映射到算法分配的聚类
        true_concepts: 字典，将客户端ID映射到真实概念
        
    返回:
        dict: 包含各种聚类评估指标的字典
    """
    # 确保两个字典的键集合相同
    common_clients = set(cluster_assignments.keys()) & set(true_concepts.keys())
    
    if not common_clients:
        return {
            "error": "没有可比较的共同客户端",
            "num_clusters": 0,
            "num_concepts": 0
        }
    
    # 提取共同客户端的标签
    cluster_labels = [cluster_assignments[client_id] for client_id in common_clients]
    concept_labels = [true_concepts[client_id] for client_id in common_clients]
    
    # 计算调整兰德指数(ARI)
    ari = adjusted_rand_score(concept_labels, cluster_labels)
    
    # 计算标准化互信息(NMI)
    nmi = normalized_mutual_info_score(concept_labels, cluster_labels)
    
    # 计算聚类纯度(Purity)
    contingency_matrix = np.zeros((max(cluster_labels) + 1, max(concept_labels) + 1))
    for i, j in zip(cluster_labels, concept_labels):
        contingency_matrix[i, j] += 1
    
    # 每个聚类中最主要的概念数量
    cluster_purity = np.sum(np.max(contingency_matrix, axis=1)) / len(cluster_labels)
    
    # 计算同质性与完整性
    cluster_counts = np.zeros(max(cluster_labels) + 1)
    concept_counts = np.zeros(max(concept_labels) + 1)
    
    for label in cluster_labels:
        cluster_counts[label] += 1
    
    for label in concept_labels:
        concept_counts[label] += 1
    
    # 计算每个聚类中来自不同概念的客户端分布
    cluster_concept_distribution = {}
    for i in range(len(contingency_matrix)):
        if np.sum(contingency_matrix[i]) > 0:  # 只处理非空聚类
            distribution = {}
            for j in range(len(contingency_matrix[i])):
                if contingency_matrix[i, j] > 0:
                    concept_percentage = contingency_matrix[i, j] / np.sum(contingency_matrix[i]) * 100
                    distribution[j] = concept_percentage
            cluster_concept_distribution[i] = distribution
    
    # 计算每个概念被分配到不同聚类的客户端分布
    concept_cluster_distribution = {}
    for j in range(contingency_matrix.shape[1]):
        if np.sum(contingency_matrix[:, j]) > 0:  # 只处理非空概念
            distribution = {}
            for i in range(contingency_matrix.shape[0]):
                if contingency_matrix[i, j] > 0:
                    cluster_percentage = contingency_matrix[i, j] / np.sum(contingency_matrix[:, j]) * 100
                    distribution[i] = cluster_percentage
            concept_cluster_distribution[j] = distribution
    
    return {
        "ari": ari,  # 调整兰德指数，-1到1，1表示完全匹配
        "nmi": nmi,  # 标准化互信息，0到1，1表示完全匹配
        "purity": cluster_purity,  # 聚类纯度，0到1，1表示每个聚类只包含一个概念
        "num_clusters": len(set(cluster_labels)),  # 聚类数量
        "num_concepts": len(set(concept_labels)),  # 概念数量
        "cluster_sizes": dict(zip(range(len(cluster_counts)), cluster_counts.tolist())),  # 每个聚类的大小
        "concept_sizes": dict(zip(range(len(concept_counts)), concept_counts.tolist())),  # 每个概念的大小
        "cluster_concept_distribution": cluster_concept_distribution,  # 每个聚类中的概念分布
        "concept_cluster_distribution": concept_cluster_distribution,  # 每个概念在聚类中的分布
    }

def get_true_concepts_at_iteration(drift_data_dir, iteration):
    """
    获取指定迭代中所有客户端的真实概念
    
    参数:
        drift_data_dir: 数据集路径
        iteration: 迭代编号
        
    返回:
        dict: 客户端ID到真实概念的映射
    """
    import os
    import json
    
    # 构建配置文件路径
    config_path = os.path.join(drift_data_dir, "drift_info", "concept_config.json")
    
    if not os.path.exists(config_path):
        print(f"错误: 找不到配置文件: {config_path}")
        return {}
    
    try:
        # 加载配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            concept_config = json.load(f)
        
        # 检查是否存在客户端轨迹
        if 'client_concept_trajectories' not in concept_config:
            print("错误: 配置中未找到客户端概念轨迹")
            return {}
        
        # 获取所有客户端在指定迭代的概念
        client_concepts = {}
        for client_id, trajectory in concept_config['client_concept_trajectories'].items():
            if 0 <= iteration < len(trajectory):
                client_concepts[client_id] = trajectory[iteration]
        
        return client_concepts
        
    except Exception as e:
        print(f"读取配置文件时出错: {e}")
        return {}

def print_clustering_evaluation(metrics):
    """
    打印聚类评估结果
    
    参数:
        metrics: 评估指标字典
    """
    print("\n======== 聚类评估结果 ========")
    print(f"调整兰德指数 (ARI): {metrics['ari']:.4f}  (范围: -1 到 1, 越高越好)")
    print(f"标准化互信息 (NMI): {metrics['nmi']:.4f}  (范围: 0 到 1, 越高越好)")
    print(f"聚类纯度 (Purity): {metrics['purity']:.4f}  (范围: 0 到 1, 越高越好)")
    print(f"聚类数量: {metrics['num_clusters']}")
    print(f"概念数量: {metrics['num_concepts']}")
    
    print("\n聚类大小:")
    for cluster_id, size in metrics['cluster_sizes'].items():
        print(f"  聚类 {cluster_id}: {size} 客户端")
    
    print("\n概念大小:")
    for concept_id, size in metrics['concept_sizes'].items():
        print(f"  概念 {concept_id}: {size} 客户端")
    
    print("\n每个聚类中的概念分布:")
    for cluster_id, distribution in metrics['cluster_concept_distribution'].items():
        print(f"  聚类 {cluster_id}:")
        for concept_id, percentage in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
            print(f"    概念 {concept_id}: {percentage:.1f}%")
    
    print("\n每个概念在聚类中的分布:")
    for concept_id, distribution in metrics['concept_cluster_distribution'].items():
        print(f"  概念 {concept_id}:")
        for cluster_id, percentage in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
            print(f"    聚类 {cluster_id}: {percentage:.1f}%")
    
    # 评估聚类质量
    if metrics['ari'] > 0.8:
        quality = "极佳"
    elif metrics['ari'] > 0.6:
        quality = "良好"
    elif metrics['ari'] > 0.4:
        quality = "一般"
    elif metrics['ari'] > 0.2:
        quality = "较差"
    else:
        quality = "很差"
    
    print(f"\n聚类与真实概念的一致性: {quality}")
    
    if metrics['ari'] < 0.4:
        print("\n改进建议:")
        print("  1. 尝试调整聚类数量，使其接近真实概念数量")
        print("  2. 优化特征提取方法，确保不同概念的特征更加可分")
        print("  3. 考虑使用更强大的聚类算法或调整现有算法的参数")
        print("  4. 为聚类算法提供更多的训练数据或迭代次数")
