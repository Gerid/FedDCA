"""
标签条件分布的Wasserstein聚类算法实现

这个模块实现了基于标签条件分布的Wasserstein距离聚类算法，
通过计算客户端在各个标签上的特征分布之间的Wasserstein距离来进行聚类。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import SpectralClustering
import time
import warnings

# 忽略警告
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class LabelConditionalWassersteinClustering:
    """
    标签条件分布的Wasserstein聚类算法
    
    该算法通过计算客户端在相同标签上的特征分布之间的Wasserstein距离来进行聚类，
    能够更准确地捕捉客户端数据分布的差异。
    """
    
    def __init__(self, num_clients, num_clusters, device='cuda', sinkhorn_reg=0.01, 
                 sinkhorn_iterations=100, temperature=0.5, verbose=True):
        """
        初始化聚类算法
        
        参数:
            num_clients: 客户端数量
            num_clusters: 聚类数量
            device: 计算设备
            sinkhorn_reg: Sinkhorn算法的正则化参数
            sinkhorn_iterations: Sinkhorn算法的最大迭代次数
            temperature: 聚类分配的温度参数
            verbose: 是否打印详细信息
        """
        self.num_clients = num_clients
        self.num_clusters = num_clusters
        self.device = device
        self.sinkhorn_reg = sinkhorn_reg
        self.sinkhorn_iterations = sinkhorn_iterations
        self.temperature = temperature
        self.verbose = verbose
        
        # 初始化聚类分配
        self.assignments = None
        
    def compute_wasserstein_distance(self, X, Y):
        """
        使用Sinkhorn算法计算两个分布之间的Wasserstein距离
        
        参数:
            X: 第一个分布的样本，形状为[n_samples_X, feature_dim]
            Y: 第二个分布的样本，形状为[n_samples_Y, feature_dim]
            
        返回:
            wasserstein_distance: 两个分布之间的Wasserstein距离
        """
        # 确保特征在同一设备上
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        # 如果样本数太少，无法计算距离，则返回最大距离
        if X.shape[0] <= 1 or Y.shape[0] <= 1:
            return torch.tensor(1.0, device=self.device)
        
        # 计算成本矩阵 C (欧氏距离的平方)
        X_expand = X.unsqueeze(1)  # [n_X, 1, d]
        Y_expand = Y.unsqueeze(0)  # [1, n_Y, d]
        C = torch.sum((X_expand - Y_expand)**2, dim=2)  # [n_X, n_Y]
        
        # 初始化边际分布
        a = torch.ones(X.shape[0], device=self.device) / X.shape[0]
        b = torch.ones(Y.shape[0], device=self.device) / Y.shape[0]
        
        # Sinkhorn迭代
        K = torch.exp(-C / self.sinkhorn_reg)
        
        u = torch.ones_like(a)
        
        for _ in range(self.sinkhorn_iterations):
            v = b / (K.t() @ u + 1e-10)
            u = a / (K @ v + 1e-10)
        
        # 计算传输矩阵
        P = torch.diag(u) @ K @ torch.diag(v)
        
        # 计算Wasserstein距离
        wasserstein_distance = torch.sum(P * C)
        
        # 返回距离值
        return wasserstein_distance
    
    def compute_distance_matrix(self, client_features):
        """
        计算客户端之间基于标签条件的Wasserstein距离矩阵
        
        参数:
            client_features: 按客户端ID和标签组织的特征字典 
                            {client_id: {label: features}}
        
        返回:
            distance_matrix: 客户端间的距离矩阵
        """
        start_time = time.time()
        client_ids = list(client_features.keys())
        n_clients = len(client_ids)
        
        if n_clients < 2:
            if self.verbose:
                print("警告: 客户端数量少于2，无法计算距离矩阵")
            return torch.zeros((n_clients, n_clients), device=self.device)
        
        # 初始化距离矩阵
        distance_matrix = torch.zeros((n_clients, n_clients), device=self.device)
        
        total_pairs = (n_clients * (n_clients - 1)) // 2
        processed_pairs = 0
        
        # 计算每对客户端间的距离
        for i in range(n_clients):
            for j in range(i+1, n_clients):
                client_i_id = client_ids[i]
                client_j_id = client_ids[j]
                
                # 获取两个客户端的按标签分组的特征
                features_i = client_features[client_i_id]
                features_j = client_features[client_j_id]
                
                # 计算所有共有标签的Wasserstein距离
                common_labels = set(features_i.keys()) & set(features_j.keys())
                
                if not common_labels:
                    # 如果没有共有标签，使用最大距离
                    distance_matrix[i, j] = distance_matrix[j, i] = 1.0
                    processed_pairs += 1
                    continue
                
                # 计算共有标签的Wasserstein距离
                label_distances = []
                label_weights = []
                
                for label in common_labels:
                    if label in features_i and label in features_j:
                        feat_i = features_i[label]
                        feat_j = features_j[label]
                        
                        if isinstance(feat_i, list) and len(feat_i) > 0:
                            feat_i = torch.stack(feat_i)
                        if isinstance(feat_j, list) and len(feat_j) > 0:
                            feat_j = torch.stack(feat_j)
                        
                        if torch.is_tensor(feat_i) and torch.is_tensor(feat_j) and \
                           feat_i.numel() > 0 and feat_j.numel() > 0:
                            # 计算这个标签下的Wasserstein距离
                            try:
                                label_dist = self.compute_wasserstein_distance(feat_i, feat_j)
                                
                                # 使用这个标签的样本数量作为权重
                                weight = (feat_i.shape[0] + feat_j.shape[0]) / 2
                                
                                label_distances.append(label_dist.item())
                                label_weights.append(weight)
                            except Exception as e:
                                if self.verbose:
                                    print(f"计算标签 {label} 的Wasserstein距离时出错: {str(e)}")
                
                if label_distances:
                    # 计算加权平均距离
                    total_weight = sum(label_weights)
                    if total_weight > 0:
                        weighted_distance = sum(d * w for d, w in zip(label_distances, label_weights)) / total_weight
                        
                        # 距离归一化到[0,1]范围
                        normalized_distance = min(1.0, max(0.0, weighted_distance))
                        
                        distance_matrix[i, j] = distance_matrix[j, i] = normalized_distance
                    else:
                        distance_matrix[i, j] = distance_matrix[j, i] = 1.0
                else:
                    distance_matrix[i, j] = distance_matrix[j, i] = 1.0
                
                processed_pairs += 1
                
                # 输出进度
                if self.verbose and processed_pairs % max(1, total_pairs // 10) == 0:
                    elapsed_time = time.time() - start_time
                    estimated_total = elapsed_time / processed_pairs * total_pairs
                    remaining_time = estimated_total - elapsed_time
                    print(f"距离计算进度: {processed_pairs}/{total_pairs} ({processed_pairs/total_pairs*100:.1f}%), "
                          f"耗时: {elapsed_time:.1f}s, 预计剩余: {remaining_time:.1f}s")
        
        if self.verbose:
            print(f"计算距离矩阵完成，耗时: {time.time() - start_time:.2f}s")
            
        return distance_matrix
    
    def fit(self, client_features):
        """
        执行聚类
        
        参数:
            client_features: 按客户端ID和标签组织的特征字典 
                            {client_id: {label: features}}
        
        返回:
            assignments: 聚类分配结果，字典 {client_id: cluster_id}
        """
        client_ids = list(client_features.keys())
        n_clients = len(client_ids)
        
        if n_clients < self.num_clusters:
            if self.verbose:
                print(f"警告: 客户端数量({n_clients})小于聚类数量({self.num_clusters})，调整聚类数量")
            self.num_clusters = max(1, n_clients)
        
        if self.verbose:
            print(f"计算基于标签条件的Wasserstein距离矩阵，客户端数: {n_clients}，聚类数: {self.num_clusters}")
        
        # 计算距离矩阵
        distance_matrix = self.compute_distance_matrix(client_features)
        
        # 将距离矩阵转换为亲和度矩阵
        # 使用高斯核，sigma = 温度参数
        affinity_matrix = torch.exp(-distance_matrix / self.temperature)
        
        # 将亲和度矩阵转换为CPU numpy数组
        affinity_np = affinity_matrix.cpu().numpy()
        
        # 检查亲和度矩阵是否有效
        if np.isnan(affinity_np).any() or np.isinf(affinity_np).any():
            if self.verbose:
                print("警告: 亲和度矩阵包含NaN或Inf值，将使用均匀分配")
            
            # 均匀分配
            cluster_labels = np.array([i % self.num_clusters for i in range(n_clients)])
        else:
            try:
                # 执行谱聚类
                if self.verbose:
                    print("执行谱聚类...")
                
                spectral = SpectralClustering(
                    n_clusters=self.num_clusters,
                    affinity='precomputed',
                    random_state=42,
                    assign_labels='kmeans'
                )
                
                cluster_labels = spectral.fit_predict(affinity_np)
                
                # 检查聚类结果
                unique_clusters = np.unique(cluster_labels)
                if len(unique_clusters) < self.num_clusters:
                    if self.verbose:
                        print(f"警告: 聚类数量({len(unique_clusters)})小于目标数量({self.num_clusters})，"
                              f"某些簇可能为空")
            except Exception as e:
                if self.verbose:
                    print(f"谱聚类失败: {str(e)}，将使用均匀分配")
                
                # 均匀分配
                cluster_labels = np.array([i % self.num_clusters for i in range(n_clients)])
        
        # 创建客户端ID到聚类ID的映射
        self.assignments = {
            client_ids[i]: int(cluster_labels[i]) 
            for i in range(n_clients)
        }
        
        # 打印各簇的客户端数量
        if self.verbose:
            cluster_counts = np.bincount(cluster_labels, minlength=self.num_clusters)
            print(f"聚类完成，簇分布: {cluster_counts}")
        
        return self.assignments
    
    def get_cluster_assignments(self):
        """获取聚类分配结果"""
        return self.assignments if self.assignments is not None else {}


# 简化的接口函数
def perform_label_conditional_clustering(clients, num_clusters, device='cuda', verbose=True):
    """
    执行基于标签条件的Wasserstein聚类
    
    参数:
        clients: 客户端列表
        num_clusters: 聚类数量
        device: 计算设备
        verbose: 是否打印详细信息
        
    返回:
        cluster_assignments: 聚类分配结果，字典 {client_id: cluster_id}
    """
    if verbose:
        print("\n收集按标签分组的中间表征...")
    
    # 收集客户端特征
    client_features = {}
    for client in clients:
        try:
            features_by_label = client.get_intermediate_outputs_with_labels()
            if features_by_label:
                client_features[client.id] = features_by_label
        except Exception as e:
            if verbose:
                print(f"收集客户端 {client.id} 特征时出错: {str(e)}")
    
    if not client_features:
        if verbose:
            print("错误: 没有收集到客户端特征，无法执行聚类")
        return {}
    
    # 执行聚类
    lcwc = LabelConditionalWassersteinClustering(
        num_clients=len(client_features),
        num_clusters=num_clusters,
        device=device,
        verbose=verbose
    )
    
    cluster_assignments = lcwc.fit(client_features)
    return cluster_assignments
