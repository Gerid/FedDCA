import numpy as np
import torch
import traceback
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, DBSCAN
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score
from sklearn.random_projection import GaussianRandomProjection
import ot  # 用于计算Wasserstein距离
import matplotlib.pyplot as plt

def compute_label_conditional_wasserstein_distance(label_conditional_data, device='cpu'):
    """
    计算基于标签条件的Wasserstein距离矩阵
    
    参数:
        label_conditional_data: 按标签组织的特征，格式为 {client_id: {label: features}}
        device: 计算设备
        
    返回:
        距离矩阵
    """
    try:
        client_ids = list(label_conditional_data.keys())
        num_clients = len(client_ids)
        
        # 初始化距离矩阵
        distance_matrix = np.zeros((num_clients, num_clients))
        
        # 计算每对客户端之间的距离
        for i, client_i in enumerate(client_ids):
            for j, client_j in enumerate(client_ids):
                if i == j:
                    distance_matrix[i, j] = 0.0
                    continue
                    
                # 计算两个客户端之间所有共有标签的Wasserstein距离
                common_labels = set(label_conditional_data[client_i].keys()) & set(label_conditional_data[client_j].keys())
                
                if not common_labels:
                    # 没有共有标签，使用最大距离
                    distance_matrix[i, j] = 1.0
                    continue
                
                # 计算每个共有标签的Wasserstein距离
                label_distances = []
                for label in common_labels:
                    features_i = label_conditional_data[client_i][label]
                    features_j = label_conditional_data[client_j][label]
                    
                    # 确保特征是numpy数组
                    if isinstance(features_i, torch.Tensor):
                        features_i = features_i.detach().cpu().numpy()
                    if isinstance(features_j, torch.Tensor):
                        features_j = features_j.detach().cpu().numpy()
                    
                    # 计算标签间的Wasserstein距离
                    try:
                        # 如果样本数不同，通过随机选择使它们相等
                        min_samples = min(len(features_i), len(features_j))
                        if min_samples < 2:
                            # 样本太少，使用欧氏距离的均值
                            mean_i = np.mean(features_i, axis=0)
                            mean_j = np.mean(features_j, axis=0)
                            label_distance = np.linalg.norm(mean_i - mean_j)
                        else:
                            # 随机选择相同数量的样本
                            idx_i = np.random.choice(len(features_i), min_samples, replace=False)
                            idx_j = np.random.choice(len(features_j), min_samples, replace=False)
                            features_i_sampled = features_i[idx_i]
                            features_j_sampled = features_j[idx_j]
                            
                            # 计算代价矩阵
                            M = ot.dist(features_i_sampled, features_j_sampled)
                            
                            # 计算Wasserstein距离
                            a = np.ones(min_samples) / min_samples
                            b = np.ones(min_samples) / min_samples
                            
                            label_distance = ot.emd2(a, b, M)
                        
                        label_distances.append(label_distance)
                    except Exception as e:
                        print(f"计算标签 {label} 的Wasserstein距离失败: {str(e)}")
                        # 失败时使用均值向量的欧氏距离
                        mean_i = np.mean(features_i, axis=0)
                        mean_j = np.mean(features_j, axis=0)
                        label_distance = np.linalg.norm(mean_i - mean_j)
                        label_distances.append(label_distance)
                
                # 取所有标签距离的平均值作为客户端间的距离
                if label_distances:
                    distance_matrix[i, j] = np.mean(label_distances)
                else:
                    distance_matrix[i, j] = 1.0  # 默认最大距离
        
        return distance_matrix
        
    except Exception as e:
        print(f"计算Wasserstein距离矩阵失败: {str(e)}")
        print(traceback.format_exc())
        
        # 失败时返回单位矩阵（除对角线外都是1）
        distance_matrix = np.ones((num_clients, num_clients))
        np.fill_diagonal(distance_matrix, 0)
        return distance_matrix

def robust_clustering(distance_matrix, n_clusters, client_ids):
    """
    更鲁棒的聚类方法，能够处理异常值和不同的数据分布
    
    参数:
        distance_matrix: 客户端间的距离矩阵
        n_clusters: 期望的聚类数量
        client_ids: 客户端ID列表
        
    返回:
        聚类分配结果
    """
    try:
        # 首先尝试谱聚类，它对数据分布的假设较少
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42
        )
        # 转换距离为相似度
        similarity = np.exp(-distance_matrix / distance_matrix.std())
        assignments = clustering.fit_predict(similarity)
        method = "SpectralClustering"
        
    except Exception as spectral_error:
        print(f"谱聚类失败: {str(spectral_error)}")
        
        try:
            # 尝试使用层次聚类，但不使用可能不兼容的affinity参数
            # 先使用MDS将距离矩阵转换为欧几里得空间中的点
            n_components = min(5, len(client_ids) - 1) if len(client_ids) > 1 else 1
            
            # 确保距离矩阵没有NaN或无穷值
            clean_matrix = np.copy(distance_matrix)
            mask = np.isnan(clean_matrix) | np.isinf(clean_matrix)
            if np.any(mask):
                print("警告: 距离矩阵包含NaN或无穷值，将进行替换")
                clean_matrix[mask] = np.nanmean(clean_matrix[~mask]) * 2 if np.any(~mask) else 1.0
            
            mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42)
            features = mds.fit_transform(clean_matrix)
            
            # 使用欧几里得距离进行聚类
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            assignments = clustering.fit_predict(features)
            method = "AgglomerativeClustering+MDS"
            
        except Exception as agg_error:
            print(f"层次聚类失败: {str(agg_error)}")
            
            try:
                # 尝试DBSCAN，它不需要预先指定集群数量
                # 估计邻域参数
                from sklearn.neighbors import NearestNeighbors
                neighbors = NearestNeighbors(n_neighbors=min(5, len(client_ids)//2 + 1))
                neighbors.fit(distance_matrix)
                distances, _ = neighbors.kneighbors(distance_matrix)
                eps = np.percentile(distances[:, 1:].flatten(), 75)  # 使用75%分位数作为eps
                
                clustering = DBSCAN(eps=eps, min_samples=2, metric='precomputed')
                assignments = clustering.fit_predict(distance_matrix)
                
                # 处理可能的噪声点（标记为-1）
                if -1 in assignments:
                    noise_indices = np.where(assignments == -1)[0]
                    # 将噪声点分配给最近的集群
                    for idx in noise_indices:
                        # 找出最近的非噪声点
                        non_noise = np.where(assignments != -1)[0]
                        if len(non_noise) > 0:
                            closest = non_noise[np.argmin(distance_matrix[idx, non_noise])]
                            assignments[idx] = assignments[closest]
                        else:
                            # 所有点都是噪声，分配到集群0
                            assignments[idx] = 0
                
                # 重新映射集群ID，确保它们是连续的
                unique_clusters = np.unique(assignments)
                mapping = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
                assignments = np.array([mapping[a] for a in assignments])
                
                # 可能的集群数量与期望不一致，调整到期望的数量
                if len(np.unique(assignments)) != n_clusters:
                    # 太多集群：合并最相似的
                    while len(np.unique(assignments)) > n_clusters:
                        # 找出最相似的两个集群
                        unique_clusters = np.unique(assignments)
                        min_dist = np.inf
                        clusters_to_merge = (0, 0)
                        
                        for i in range(len(unique_clusters)):
                            for j in range(i+1, len(unique_clusters)):
                                c1, c2 = unique_clusters[i], unique_clusters[j]
                                # 计算集群间平均距离
                                idx1 = np.where(assignments == c1)[0]
                                idx2 = np.where(assignments == c2)[0]
                                if len(idx1) > 0 and len(idx2) > 0:
                                    avg_dist = np.mean(distance_matrix[np.ix_(idx1, idx2)])
                                    if avg_dist < min_dist:
                                        min_dist = avg_dist
                                        clusters_to_merge = (c1, c2)
                        
                        # 合并集群
                        c1, c2 = clusters_to_merge
                        assignments[assignments == c2] = c1
                        
                        # 重新映射ID以保持连续性
                        unique_clusters = np.unique(assignments)
                        mapping = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
                        assignments = np.array([mapping[a] for a in assignments])
                    
                    # 太少集群：分裂最大的
                    while len(np.unique(assignments)) < n_clusters:
                        unique_clusters = np.unique(assignments)
                        # 找出最大的集群
                        cluster_sizes = [np.sum(assignments == c) for c in unique_clusters]
                        largest_cluster = unique_clusters[np.argmax(cluster_sizes)]
                        
                        # 找出属于最大集群的点
                        largest_indices = np.where(assignments == largest_cluster)[0]
                        
                        if len(largest_indices) >= 2:
                            # 在最大集群内使用KMeans分裂
                            from sklearn.cluster import KMeans
                            # 提取最大集群的特征
                            sub_matrix = distance_matrix[np.ix_(largest_indices, largest_indices)]
                            
                            # 使用MDS将距离转换为特征
                            n_components = min(3, len(largest_indices) - 1) if len(largest_indices) > 1 else 1
                            mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42)
                            sub_features = mds.fit_transform(sub_matrix)
                            
                            # 将集群分成两部分
                            kmeans = KMeans(n_clusters=2, random_state=42)
                            sub_assignments = kmeans.fit_predict(sub_features)
                            
                            # 将一部分分配到新的集群
                            new_cluster_id = max(unique_clusters) + 1
                            for i, idx in enumerate(largest_indices):
                                if sub_assignments[i] == 1:  # 分配第二组到新集群
                                    assignments[idx] = new_cluster_id
                        else:
                            # 如果最大集群只有一个点，无法分裂，随机分配一个点到新集群
                            if len(assignments) > len(unique_clusters):
                                # 随机选择一个未使用的点分配到新集群
                                unused_indices = list(set(range(len(assignments))) - set(largest_indices))
                                if unused_indices:
                                    random_idx = unused_indices[0]
                                    assignments[random_idx] = max(unique_clusters) + 1
                            else:
                                # 没有更多点可以分配了，只能复制一个点
                                break
                
                method = "DBSCAN+Adjustment"
                
            except Exception as dbscan_error:
                print(f"DBSCAN聚类失败: {str(dbscan_error)}")
                # 所有方法都失败，使用均匀分配
                print("所有聚类方法都失败，使用均匀分配...")
                assignments = np.array([i % n_clusters for i in range(len(client_ids))])
                method = "Uniform"
    
    print(f"使用 {method} 聚类方法成功完成聚类")
    
    # 评估聚类质量
    try:
        sil_score = silhouette_score(distance_matrix, assignments, metric='precomputed')
        print(f"轮廓系数: {sil_score:.4f} (越接近1越好)")
    except Exception as e:
        print(f"无法计算轮廓系数: {str(e)}")
    
    return assignments

def visualize_clusters(features, cluster_labels, client_ids, round_idx, output_dir='results/clustering', true_labels=None):
    """
    可视化聚类结果
    
    参数:
        features: 客户端特征
        cluster_labels: 聚类标签
        client_ids: 客户端ID列表
        round_idx: 当前轮次
        output_dir: 输出目录
        true_labels: 真实标签（如果有）
    """
    try:
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 将特征降到2维进行可视化
        if features.shape[1] > 2:
            from sklearn.manifold import TSNE
            # 根据样本数动态调整perplexity参数
            n_samples = len(features)
            perplexity = min(30, max(5, n_samples // 5))
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            features_2d = tsne.fit_transform(features)
        else:
            features_2d = features
        
        # 绘制聚类结果
        plt.figure(figsize=(12, 10))
        
        # 绘制聚类结果
        plt.subplot(1, 1 if true_labels is None else 2, 1)
        
        # 获取唯一的聚类标签并分配颜色
        unique_clusters = np.unique(cluster_labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
        
        for i, cluster in enumerate(unique_clusters):
            cluster_points = features_2d[cluster_labels == cluster]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[colors[i]], label=f'Cluster {cluster}')
        
        # 添加客户端ID标签
        for i, (x, y) in enumerate(features_2d):
            plt.annotate(str(client_ids[i]), (x, y), fontsize=8)
        
        plt.title(f'Client Clustering (Round {round_idx})')
        plt.legend()
        
        # 如果有真实标签，绘制真实分布
        if true_labels is not None:
            plt.subplot(1, 2, 2)
            
            # 获取唯一的真实标签并分配颜色
            unique_true_labels = np.unique(true_labels)
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_true_labels)))
            
            for i, label in enumerate(unique_true_labels):
                label_points = features_2d[true_labels == label]
                plt.scatter(label_points[:, 0], label_points[:, 1], c=[colors[i]], label=f'True {label}')
            
            # 添加客户端ID标签
            for i, (x, y) in enumerate(features_2d):
                plt.annotate(str(client_ids[i]), (x, y), fontsize=8)
            
            plt.title(f'True Distribution (Round {round_idx})')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/cluster_visualization_round_{round_idx}.png')
        plt.close()
        
    except Exception as e:
        print(f"可视化聚类失败: {str(e)}")
        print(traceback.format_exc())
