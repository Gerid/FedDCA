import copy
import os
import random
import time
import traceback
import json
from flcore.clients.clientdca import clientDCA
from flcore.servers.serverbase import Server
from flcore.servers.serverdca_concepts import initialize_shared_concepts, distribute_concepts_to_clients, save_concept_progress, analyze_concept_alignment
from utils.concept_drift_simulation import create_shared_concepts, initialize_drift_patterns
from threading import Thread
import torch
import torch.nn as nn
import numpy as np
import scipy.stats as stats
try:
    import ot  # 导入POT库 (Python Optimal Transport)
except ImportError:
    print("Warning: 未找到POT库，将使用替代方法计算Wasserstein距离")
    
from utils.improved_vwc_clustering import VariationalWassersteinClustering
from utils.label_conditional_clustering import perform_label_conditional_clustering
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
import traceback
from functools import wraps

def plot_metrics(train_loss_list=None, test_acc_list=None):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # 创建存储指标的列表
            if not hasattr(self, 'train_loss_history'):
                self.train_loss_history = []
            if not hasattr(self, 'test_acc_history'):
                self.test_acc_history = []
            
            # 执行原函数
            result = func(self, *args, **kwargs)
            
            # 绘制训练过程图
            plt.figure(figsize=(12, 5))
            
            # Plot training loss
            plt.subplot(121)
            plt.plot(self.rs_train_loss, 'r-', label='Training Loss')
            plt.title('Training Loss vs. Rounds')
            plt.xlabel('Round')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # Plot test accuracy
            plt.subplot(122)
            plt.plot(self.rs_test_acc, 'b-', label='Test Accuracy')
            plt.title('Test Accuracy vs. Rounds')
            plt.xlabel('Round')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            
            # 保存图片
            save_path = os.path.join('results', f'training_metrics_{time.strftime("%Y%m%d-%H%M%S")}.png')
            if not os.path.exists('results'):
                os.makedirs('results')
            plt.savefig(save_path)
            plt.close()
            
            return result
        return wrapper
    return decorator

class FedDCA(Server):    
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientDCA)
        
        self.cluster_inited = False
        self.args = args
        self.args.load_pretrain = False

        # 初始化所有必要的字典
        self.cluster_models = {}  # 映射集群到模型
        self.clusters = {}  # 存储客户端到集群的映射
        self.cluster_centroids = {}  # 存储每个集群的中心模型
        self.client_features = {}  # 存储客户端的特征
        self.drift_threshold = args.drift_threshold if hasattr(args, 'drift_threshold') else 0.1
        
        # 添加聚类历史记录追踪
        self.client_cluster_history = {}  # 客户端聚类历史，用于稳定性分析
        
        # 新增：默认设置聚类方法为增强型标签条件聚类
        if not hasattr(args, 'clustering_method'):
            self.args.clustering_method = 'enhanced_label'
            
        # 概念漂移相关参数设置
        self.drift_threshold = args.drift_threshold if hasattr(args, 'drift_threshold') else 0.1
        self.current_iteration = 0
        self.max_iterations = args.max_iterations if hasattr(args, 'max_iterations') else 200
        self.use_drift_dataset = args.use_drift_dataset if hasattr(args, 'use_drift_dataset') else False
        self.drift_data_dir = args.drift_data_dir if hasattr(args, 'drift_data_dir') else "system/Cifar100_clustered/"

        # 添加命令行参数支持
        if hasattr(args, 'cmd_args') and args.cmd_args:
            # 如果命令行中指定了这些参数，则覆盖默认值
            if hasattr(args.cmd_args, 'use_drift_dataset'):
                self.use_drift_dataset = args.cmd_args.use_drift_dataset
            if hasattr(args.cmd_args, 'drift_data_dir'):
                self.drift_data_dir = args.cmd_args.drift_data_dir
            if hasattr(args.cmd_args, 'max_iterations'):
                self.max_iterations = args.cmd_args.max_iterations
        
        # 初始化服务器端共享的概念漂移模拟
        self.initialize_shared_concepts()
                
        # 如果启用了概念漂移数据集，加载漂移配置
        if self.use_drift_dataset and self.drift_data_dir:
            self.load_drift_config()

        # 确保args中包含必要的聚类参数
        if not hasattr(args, 'num_clusters'):
            args.num_clusters = 2  # 默认集群数
        if not hasattr(args, 'split_threshold'):
            args.split_threshold = 0.3  # 默认分裂阈值
        if not hasattr(args, 'merge_threshold'):
            args.merge_threshold = 0.1  # 默认合并阈值
        if not hasattr(args, 'kde_samples'):
            args.kde_samples = 100  # 默认KDE采样数
        if not hasattr(args, 'proxy_dim'):
            args.proxy_dim = 128  # 代理特征维度
        if not hasattr(args, 'sinkhorn_reg'):
            args.sinkhorn_reg = 0.01  # Sinkhorn 正则化参数

        # 初始化 VWC 聚类器
        self.vwc = VariationalWassersteinClustering(
            num_clients=args.num_clients,
            num_clusters=args.num_clusters,
            proxy_dim=args.proxy_dim,
            sinkhorn_reg=args.sinkhorn_reg
        )

        self.Budget = []  # 用于记录每轮训练的时间成本
        self.rs_test_acc = []  # 初始化测试准确率列表
        self.rs_train_loss = []  # 初始化训练损失列表
        self.rs_test_acc = []  # 初始化测试准确率列表
        self.rs_train_loss = []  # 初始化训练损失列表    # 方法已移至serverdca_concepts.py模块

    def load_drift_config(self):
        """加载概念漂移数据集的配置信息"""
        import os
        import json
    
        try:
            # 构建配置文件路径
            config_path = os.path.join(self.drift_data_dir, "drift_info", "concept_config.json")
        
            if not os.path.exists(config_path):
                print(f"Warning: Drift configuration file not found at {config_path}")
                return
            
            # 加载配置
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # 提取关键配置信息
            self.drift_iterations = config.get('drift_iterations', [])
            self.num_concepts = config.get('num_concepts', 5)
            self.client_concepts = config.get('client_concepts', {})
            self.client_drift_types = config.get('client_drift_types', {})
        
            # 更新聚类数量为概念数量
            if hasattr(self.args, 'num_clusters'):
                self.args.num_clusters = self.num_concepts
                print(f"Setting number of clusters to {self.num_concepts} (matching number of concepts)")
        
            # 将配置信息传递给客户端
            for client in self.clients:
                client.drift_args = {
                    'drift_iterations': self.drift_iterations,
                    'concepts': self.client_concepts.get(str(client.id), []),
                    'drift_type': self.client_drift_types.get(str(client.id), 'sudden')
                }
                # 启用客户端的概念漂移数据集模式
                client.use_drift_dataset = True
                client.drift_data_dir = self.drift_data_dir
                client.max_iterations = self.max_iterations
                client.simulate_drift = True
            
            print(f"Loaded drift configuration: {self.num_concepts} concepts, {len(self.drift_iterations)} drift points")
            print(f"Drift will occur at iterations: {self.drift_iterations}")
        
        except Exception as e:
            print(f"Error loading drift configuration: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
    def train_round(self, round_idx):
        """执行单轮训练，并更新客户端的迭代状态"""
        try:
            # 如果使用概念漂移数据集，更新客户端的迭代状态
            if self.use_drift_dataset:
                print(f"\n更新客户端迭代状态到 {self.current_iteration}")
                for client in self.selected_clients:
                    client.update_iteration(self.current_iteration)
            
                # 检查是否是漂移点
                if self.current_iteration in self.drift_iterations:
                    print(f"\n⚠️ 在迭代 {self.current_iteration} 发生概念漂移")
            
                # 更新迭代计数器，为下一轮做准备
                self.current_iteration = (self.current_iteration + 1) % self.max_iterations            # 执行正常的训练过程
            super().train_round(round_idx)
        
        except Exception as e:
            print(f"Error in train_round: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
    def kde_estimate(self, data):
        try:
            bandwidth = 1.06 * data.std() * data.size ** (-1 / 5.0)
            kde = stats.gaussian_kde(data, bw_method=bandwidth)
            return kde
        except Exception as e:
            print(f"KDE估计失败: {str(e)}")
            return None
        except Exception as e:
            print(f"KDE估计失败: {str(e)}")
            return None

    def collect_proxy_data(self):
        for client in self.selected_clients:
            data = client.intermediate_output()  # Get intermediate representations
            kde = self.kde_estimate(data)
            proxy_data_points = [
                kde.resample()[0] for _ in range(len(data))
            ]  # Generate proxy points
            self.client_features[client.id] = proxy_data_points

    def collect_label_conditional_proxy_data(self):
        """
        收集每个客户端按标签条件分组的代理数据
        
        返回:
            dict: 按客户端ID和标签组织的代理数据字典，格式为 {client_id: {label: features}}
        """
        label_conditional_proxy_data = {}
        
        for client in self.selected_clients:
            # 从客户端获取按标签分组的特征
            features_by_label = client.get_intermediate_outputs_with_labels()
            
            if not features_by_label:
                print(f"Warning: No label-conditional features available for client {client.id}")
                continue
            
            # 为当前客户端创建标签条件代理数据
            label_conditional_proxy_data[client.id] = {}
            
            # 对每个标签的特征生成代理数据点
            for label, features in features_by_label.items():
                # 确保特征数据格式正确
                if isinstance(features, torch.Tensor):
                    features = features.detach().cpu().numpy()
                  # 跳过空的特征集
                if features.size == 0 or features.shape[0] == 0:
                    continue
                    
                # 使用KDE生成该标签的代理数据
                try:                    # 处理一维数据
                    if features.ndim == 1:
                        features = features.reshape(-1, 1)
                        
                    # 如果样本数足够，使用KDE估计分布
                    if features.shape[0] >= 5:  # 最少需要几个样本才能可靠估计
                        try:
                            # 检查维度和样本的关系
                            n_samples, n_dim = features.shape
                            
                            if n_samples > n_dim:
                                # 正常情况：样本数大于维度，直接使用KDE
                                pass
                            try:
                                # 转置特征以适应stats.gaussian_kde的输入格式
                                kde = stats.gaussian_kde(features.T)
                                
                                # 采样点的数量，确保有足够的样本来捕捉分布
                                num_samples = min(self.args.kde_samples, max(20, features.shape[0]))
                                
                                # 重采样生成代理数据
                                sampled = kde.resample(num_samples).T
                                label_conditional_proxy_data[client.id][label] = sampled
                            except Exception as e:
                                #print(f"KDE处理失败，对客户端{client.id}标签{label}使用原始特征: {str(e)}")
                                # KDE失败，使用原始特征
                                label_conditional_proxy_data[client.id][label] = features
                        except Exception as kde_error:
                            #print(f"KDE处理失败，对客户端{client.id}标签{label}使用原始特征: {str(kde_error)}")
                            # KDE失败，使用原始特征
                            label_conditional_proxy_data[client.id][label] = features
                    else:
                        # 如果样本太少，直接使用原始数据
                        label_conditional_proxy_data[client.id][label] = features
                except Exception as kde_error:
                    print(f"Error in KDE for client {client.id}, label {label}: {str(kde_error)}")
                    # 出错时使用原始特征
                    label_conditional_proxy_data[client.id][label] = features
        
        return label_conditional_proxy_data

    def perform_PRE(self, proxy_data):
        """
        使用KDE生成代理数据点
        """
        proxy_points = {}
        for client_id, reps in proxy_data.items():
            if reps.shape[0] == 0:
                continue
            kde = stats.gaussian_kde(reps.T)
            sampled = kde.resample(self.args.kde_samples).T
            proxy_points[client_id] = sampled
        return proxy_points

    def vwc_clustering(self):
        """初始化聚类"""
        try:
            # 收集所有客户端的代理数据
            proxy_points = {}
            for client in self.selected_clients:
                if hasattr(client, 'intermediate_output') and client.intermediate_output is not None:
                    proxy_points[client.id] = client.intermediate_output

            if not proxy_points:
                print("Warning: No proxy data available for clustering")
                return

            # 确保有足够的样本进行聚类
            num_samples = len(proxy_points)
            if num_samples < self.args.num_clusters:
                print(f"Warning: Number of proxy points ({num_samples}) is less than the number of clusters ({self.args.num_clusters})")
                self.args.num_clusters = max(1, num_samples)

            # 随机选择初始集群中心
            cluster_centers = {}
            center_client_ids = random.sample(list(proxy_points.keys()), self.args.num_clusters)
            for i, client_id in enumerate(center_client_ids):
                cluster_centers[i] = proxy_points[client_id]
                self.cluster_centroids[i] = copy.deepcopy(self.global_model)

            # 为每个客户端分配最近的集群
            for client_id, client_data in proxy_points.items():
                distances = {
                    cluster_id: torch.norm(client_data - center)
                    for cluster_id, center in cluster_centers.items()
                }
                closest_cluster = min(distances, key=distances.get)
                self.clusters[client_id] = closest_cluster

        except Exception as e:
            print(f"Error in VWC clustering: {str(e)}")
            # 发生错误时，确保每个客户端至少被分配到默认集群
            for client in self.selected_clients:
                if client.id not in self.clusters:
                    self.clusters[client.id] = 0
            if 0 not in self.cluster_centroids:
                self.cluster_centroids[0] = copy.deepcopy(self.global_model)

    def adaptive_clustering(self):
        """自适应调整集群"""
        try:
            # 计算每个集群的权重
            cluster_weights = {}
            for cluster_id in set(self.clusters.values()):
                count = sum(1 for cid in self.clusters.values() if cid == cluster_id)
                cluster_weights[cluster_id] = count / len(self.clusters)

            # 处理每个集群
            for cluster_id, weight in cluster_weights.items():
                if weight > self.args.split_threshold:
                    # 分裂集群
                    new_cluster_id = max(self.clusters.values()) + 1
                    cluster_clients = [
                        (cid, features) 
                        for cid, features in self.client_features.items()
                        if self.clusters.get(cid) == cluster_id
                    ]
                    
                    if cluster_clients:
                        # 取一半客户端形成新集群
                        half_point = len(cluster_clients) // 2
                        for cid, _ in cluster_clients[half_point:]:
                            self.clusters[cid] = new_cluster_id
                        
                        # 为新集群创建中心模型
                        self.cluster_centroids[new_cluster_id] = copy.deepcopy(
                            self.cluster_centroids[cluster_id]
                        )

                elif weight < self.args.merge_threshold and len(cluster_weights) > 1:
                    # 合并集群
                    other_clusters = [c for c in cluster_weights.keys() if c != cluster_id]
                    if other_clusters:
                        # 找到最近的集群
                        closest_cluster = min(
                            other_clusters,
                            key=lambda c: torch.norm(
                                torch.mean(torch.stack([
                                    f for cid, f in self.client_features.items()
                                    if self.clusters.get(cid) == cluster_id
                                ]), dim=0) -
                                torch.mean(torch.stack([
                                    f for cid, f in self.client_features.items()
                                    if self.clusters.get(cid) == c
                                ]), dim=0)
                            )
                        )
                        
                        # 将当前集群的客户端合并到最近的集群
                        for cid in list(self.clusters.keys()):
                            if self.clusters[cid] == cluster_id:
                                self.clusters[cid] = closest_cluster
                        
                        # 删除旧的集群中心
                        if cluster_id in self.cluster_centroids:
                            del self.cluster_centroids[cluster_id]

        except Exception as e:
            print(f"Error in adaptive clustering: {str(e)}")
            # 发生错误时保持当前集群状态

    def update_global_model(self):
        """更新全局模型和集群中心模型"""
        try:
            # 按集群组织客户端模型
            cluster_models = {}
            for client_id, cluster_id in self.clusters.items():
                if cluster_id not in cluster_models:
                    cluster_models[cluster_id] = []
                # 找到对应的客户端
                client = next((c for c in self.selected_clients if c.id == client_id), None)
                if client:
                    cluster_models[cluster_id].append(client.model)

            # 更新每个集群的中心模型
            for cluster_id, models in cluster_models.items():
                if models:  # 确保集群中有模型可以聚合
                    self.cluster_centroids[cluster_id] = self.average_models(models)

            # 更新全局模型（所有集群中心的平均）
            if self.cluster_centroids:
                self.global_model = self.average_models(list(self.cluster_centroids.values()))

        except Exception as e:
            print(f"Error in update_global_model: {str(e)}")
            # 发生错误时保持当前模型状态    
    
    def select_clustering_algorithm(self):
        """
        根据配置选择使用的聚类算法
        
        返回:
            str: 聚类算法名称 ('vwc', 'label_conditional' 或 'enhanced_label')
        """
        # 默认使用原始的VWC
        clustering_method = 'vwc'
        
        # 如果在args中指定了聚类方法，则使用指定的方法
        if hasattr(self.args, 'clustering_method'):
            clustering_method = self.args.clustering_method
            
        # 确保聚类方法是有效的
        valid_methods = ['vwc', 'label_conditional', 'enhanced_label']
        if clustering_method not in valid_methods:
            print(f"警告：未知的聚类方法 '{clustering_method}'，使用默认的 'vwc' 方法")
            clustering_method = 'vwc'
            
        return clustering_method
    
    def perform_label_conditional_clustering(self, verbose=False):
        """
        执行基于标签条件分布的Wasserstein聚类
        
        参数:
            verbose: 是否打印详细信息
            
        返回:
            bool: 聚类是否成功
        """
        print("使用基于标签条件的Wasserstein聚类...")
        try:
            # 使用标签条件聚类函数
            cluster_assignments = perform_label_conditional_clustering(
                clients=self.selected_clients,
                num_clusters=self.args.num_clusters,
                device=self.device,
                verbose=verbose
            )
            
            # 如果聚类成功，更新客户端分配
            if cluster_assignments:
                # 记录历史分配（用于分析聚类稳定性）
                for client_id, cluster_id in cluster_assignments.items():
                    if client_id not in self.client_cluster_history:
                        self.client_cluster_history[client_id] = []
                    self.client_cluster_history[client_id].append(cluster_id)
                
                # 更新当前分配
                self.clusters.update(cluster_assignments)
                print(f"标签条件聚类完成，形成 {len(set(self.clusters.values()))} 个集群")
                return True
            else:
                print("警告: 标签条件聚类未能产生有效结果")
                return False
        except Exception as e:
            print(f"标签条件聚类失败: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False

    def label_conditional_clustering(self):
        """
        使用标签条件代理数据进行聚类
        """
        print("执行基于标签条件的聚类...")
        try:
            # 收集标签条件代理数据
            label_conditional_data = self.collect_label_conditional_proxy_data()
            
            if not label_conditional_data:
                print("Warning: 没有足够的标签条件代理数据进行聚类")
                # 退回到标准聚类
                return self.vwc_clustering()
            
            # 计算基于标签条件的距离矩阵
            distance_matrix = self.compute_label_conditional_wasserstein_distance(label_conditional_data)
            
            # 聚类客户端
            from sklearn.cluster import AgglomerativeClustering
            
            # 获取客户端ID列表以维护顺序
            client_ids = list(label_conditional_data.keys())
              # 如果可用客户端数量不足以进行所需数量的聚类，调整聚类数
            num_clusters = min(self.args.num_clusters, len(client_ids))
            if num_clusters < 2:
                num_clusters = 1              # 使用层次聚类
            try:
                # 尝试使用预计算的距离矩阵
                clustering = AgglomerativeClustering(
                    n_clusters=num_clusters,
                    affinity='precomputed',  # 我们提供了预计算的距离矩阵
                    linkage='average'
                )
                cluster_assignments = clustering.fit_predict(distance_matrix)
            except TypeError as e:
                print(f"层次聚类参数错误: {str(e)}")
                print("使用兼容性更好的聚类参数...")
                
                # 使用不同的参数配置，可能是scikit-learn版本不同导致兼容性问题
                if 'affinity' in str(e):
                    # 如果是affinity参数问题，不使用预计算距离
                    # 将距离矩阵转换为特征矩阵，使用欧几里得距离
                    from sklearn.manifold import MDS
                    
                    try:
                        # 确保距离矩阵没有NaN或无穷值
                        distance_matrix_np = distance_matrix.cpu().numpy() if isinstance(distance_matrix, torch.Tensor) else distance_matrix
                        if np.isnan(distance_matrix_np).any() or np.isinf(distance_matrix_np).any():
                            # 替换NaN和无穷值
                            print("警告: 距离矩阵包含NaN或无穷值，将进行替换")
                            nan_mask = np.isnan(distance_matrix_np)
                            inf_mask = np.isinf(distance_matrix_np)
                            distance_matrix_np[nan_mask | inf_mask] = np.nanmean(distance_matrix_np[~(nan_mask | inf_mask)]) * 2
                        
                        # 将距离矩阵转换为欧几里得空间中的点
                        n_components = min(5, len(client_ids) - 1) if len(client_ids) > 1 else 1
                        mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42)
                        features = mds.fit_transform(distance_matrix_np)
                        
                        # 使用欧几里得距离进行聚类
                        clustering = AgglomerativeClustering(n_clusters=num_clusters)
                        cluster_assignments = clustering.fit_predict(features)
                    except Exception as mds_error:
                        print(f"MDS转换失败: {str(mds_error)}")
                        # 如果MDS失败，使用均匀分配
                        print("使用均匀分配...")
                        cluster_assignments = np.array([i % num_clusters for i in range(len(client_ids))])
                else:
                    # 其他错误，尝试最简单的参数
                    try:
                        clustering = AgglomerativeClustering(n_clusters=num_clusters)
                        
                        # 如果无法直接使用距离矩阵，尝试随机投影降维
                        from sklearn.random_projection import GaussianRandomProjection
                        
                        # 生成随机特征进行聚类
                        n_components = min(20, len(client_ids))
                        random_projection = GaussianRandomProjection(n_components=n_components, random_state=42)
                        random_features = random_projection.fit_transform(
                            np.eye(len(client_ids))  # 单位矩阵作为输入
                        )
                        
                        cluster_assignments = clustering.fit_predict(random_features)                    
                        
                    except Exception as cluster_error:
                        print(f"聚类完全失败: {str(cluster_error)}")
                        # 如果所有方法都失败，使用均匀分配
                        print("使用均匀分配...")
                        cluster_assignments = np.array([i % num_clusters for i in range(len(client_ids))])
            
            # 更新客户端到集群的映射
            for i, client_id in enumerate(client_ids):
                cluster_id = cluster_assignments[i]
                self.clusters[client_id] = int(cluster_id)
                
                # 记录聚类历史，用于稳定性分析
                if client_id not in self.client_cluster_history:
                    self.client_cluster_history[client_id] = []
                self.client_cluster_history[client_id].append(int(cluster_id))
            
            # 初始化集群中心模型
            self.update_cluster_models()
            
            print(f"基于标签条件的聚类完成，共 {num_clusters} 个集群")
            
            # 计算评估指标
            if len(client_ids) >= 2:
                try:
                    from sklearn.metrics import silhouette_score
                    silhouette_avg = silhouette_score(distance_matrix, cluster_assignments, metric='precomputed')
                    print(f"轮廓系数: {silhouette_avg:.4f} (越接近1越好)")
                except Exception as metric_error:
                    print(f"计算聚类评估指标时出错: {str(metric_error)}")
            
            return True
        
        except Exception as cluster_error:
            print(f"标签条件聚类失败: {str(cluster_error)}")
            print(traceback.format_exc())
            
            # 发生错误时，确保每个客户端至少被分配到默认集群
            for client in self.selected_clients:
                if client.id not in self.clusters:
                    self.clusters[client.id] = 0
                    
                    # 也更新历史记录
                    if client.id not in self.client_cluster_history:
                        self.client_cluster_history[client.id] = []
                    self.client_cluster_history[client.id].append(0)
            
            return False

    def average_models(self, models):
        """对多个模型进行加权平均"""
        if not models:
            return None
        
        try:
            # 使用第一个模型的状态字典作为基础
            avg_state_dict = models[0].state_dict()
            
            # 对所有参数求和
            for key in avg_state_dict.keys():
                for i in range(1, len(models)):
                    avg_state_dict[key] += models[i].state_dict()[key]
                avg_state_dict[key] = torch.div(avg_state_dict[key], len(models))
            
            # 创建一个新的模型并加载平均后的参数
            avg_model = copy.deepcopy(models[0])
            avg_model.load_state_dict(avg_state_dict)
            
            return avg_model
            
        except Exception as e:
            print(f"Error in average_models: {str(e)}")
            # 发生错误时返回第一个模型的副本
            return copy.deepcopy(models[0])

    def update_cluster_models(self):
        """更新每个集群的中心模型"""
        try:
            # 按集群组织客户端模型
            cluster_models = {}
            for client in self.selected_clients:
                cluster_id = self.clusters.get(client.id, 0)
                if cluster_id not in cluster_models:
                    cluster_models[cluster_id] = []
                cluster_models[cluster_id].append(client.model)

            # 更新每个集群的中心模型
            for cluster_id, models in cluster_models.items():
                if models:  # 确保集群中有模型可以聚合
                    # 计算加权平均
                    weights = torch.ones(len(models)) / len(models)  # 简单平均
                    avg_model = self.aggregate_models(models, weights)
                    self.cluster_centroids[cluster_id] = avg_model

        except Exception as e:
            print(f"Error in update_cluster_models: {str(e)}")
            # 发生错误时保持当前模型状态

    def aggregate_models(self, models, weights):
        """聚合多个模型参数
        Args:
            models: 模型列表
            weights: 每个模型的权重
        Returns:
            聚合后的模型
        """
        if not models:
            return None
        try:
            # 确定目标设备
            target_device = self.device  # 使用服务器的设备
            
            # 确保权重张量也在正确的设备上
            weights = weights.to(target_device)
            
            aggregated_model = copy.deepcopy(models[0])
            state_dicts = [model.state_dict() for model in models]
            
            with torch.no_grad():
                for key in state_dicts[0].keys():
                    # 检查所有模型该参数shape是否一致
                    shapes = [sd[key].shape for sd in state_dicts]
                    if all(s == shapes[0] for s in shapes):
                        # 确保所有参数都在同一个设备上，并转为float类型
                        stacked_params = torch.stack([
                            sd[key].float().to(target_device) for sd in state_dicts
                        ])
                        # 支持任意shape参数
                        avg_param = torch.sum(weights.view(-1, *([1]* (stacked_params.dim()-1))) * stacked_params, dim=0)
                        # 确保参数类型匹配原始模型
                        if aggregated_model.state_dict()[key].dtype != avg_param.dtype:
                            avg_param = avg_param.to(dtype=aggregated_model.state_dict()[key].dtype)
                        aggregated_model.state_dict()[key].data.copy_(avg_param)
                    else:
                        # shape不一致，跳过该参数
                        print(f"Warning: Skip param {key} due to shape mismatch: {shapes}")
            
            return aggregated_model
        except Exception as e:
            print(f"Error in aggregate_models: {str(e)}")
            import traceback
            print(traceback.format_exc())  # 打印详细错误信息
            return copy.deepcopy(models[0])

    # def evaluate(self):
    #     """评估当前模型性能"""
    #     stats = {
    #         'acc_per_client': [],
    #         'loss_per_client': []
    #     }

    #     for client in self.selected_clients:
    #         try:
    #             # 使用客户端的测试方法获取性能指标
    #             test_acc, test_num, _ = client.test_metrics()
    #             train_loss, train_num = client.train_metrics()
                
    #             if test_num > 0:
    #                 stats['acc_per_client'].append(test_acc / test_num)
    #             if train_num > 0:
    #                 stats['loss_per_client'].append(train_loss / train_num)
    #         except Exception as e:
    #             print(f"Error evaluating client {client.id}: {str(e)}")
    #             continue

    #     # 计算并记录平均性能
    #     if stats['acc_per_client']:
    #         avg_acc = np.mean(stats['acc_per_client'])
    #         self.rs_test_acc.append(avg_acc)
    #         print(f"Average Test Accuracy: {100*avg_acc:.2f}%")
        
    #     if stats['loss_per_client']:
    #         avg_loss = np.mean(stats['loss_per_client'])
    #         print(f"Average Train Loss: {avg_loss:.4f}")

    #     # 记录集群信息
    #     cluster_stats = {}
    #     for client_id, cluster_id in self.clusters.items():
    #         if cluster_id not in cluster_stats:
    #             cluster_stats[cluster_id] = {'count': 0}
    #         cluster_stats[cluster_id]['count'] += 1

    #     print("\nCluster Distribution:")
    #     for cluster_id, stats in cluster_stats.items():
    #         print(f"Cluster {cluster_id}: {stats['count']} clients")    @plot_metrics()
    def train(self):
        """训练过程的主控制流"""
        # 首先初始化共享概念，确保所有客户端都使用相同的概念集
        if not hasattr(self, 'drift_concepts_initialized'):
            print("\n初始化共享概念漂移配置...")
            config = initialize_shared_concepts(self)
            self.drift_concepts_initialized = True
            
            # 设置聚类数量为概念数量
            if hasattr(config, 'num_concepts') and config['num_concepts'] > 0:
                self.args.num_clusters = config['num_concepts']
                print(f"将聚类数量设置为与概念数量匹配: {config['num_concepts']}")
        
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            
            try:
                # 选择本轮参与训练的客户端
                self.selected_clients = self.select_clients()
                
                if not self.cluster_inited:
                    # 首轮训练时初始化集群
                    print("\nInitializing clusters...")
                    for client in self.selected_clients:
                        self.clusters[client.id] = 0
                    self.cluster_centroids[0] = copy.deepcopy(self.global_model)
                    self.cluster_inited = True

                # 客户端本地训练
                print("\nClients training locally...")
                for client in self.selected_clients:
                    client.train()
                
                # 收集中间特征用于聚类
                print("\nCollecting features for clustering...")
                proxy_points = []
                for client in self.selected_clients:
                    if hasattr(client, 'intermediate_output') and client.intermediate_output is not None:
                        # 确保数据在正确的设备上
                        if isinstance(client.intermediate_output, torch.Tensor):
                            intermediate_output = client.intermediate_output.to(self.device)
                        else:
                            intermediate_output = torch.tensor(client.intermediate_output).to(self.device)
                        proxy_points.append(intermediate_output)
                
                if len(proxy_points) > 0:
                    # 转换为张量并调整维度
                    proxy_points = torch.stack(proxy_points)
                    if len(proxy_points.shape) < 3:
                        proxy_points = proxy_points.unsqueeze(1)
                    
                    # 确保数据在正确的设备上
                    proxy_points = proxy_points.to(self.device)

                    # # 在执行 VWC 聚类前，检查是否刚刚发生概念漂移
                    # is_drift_point = False
                    # if self.use_drift_dataset and (self.current_iteration - 1) in self.drift_iterations:
                    #     is_drift_point = True
                    #     print("\n检测到概念漂移点，强制执行重新聚类...")                    # # 如果是漂移点或者特征更新，则执行聚类
                    # if is_drift_point or len(proxy_points) > 0:
                    #     # 执行 VWC 聚类
                    #     print("\nPerforming VWC clustering...")
                    #     # ...原有的聚类代码...
                          # 执行聚类
                    print("\n执行客户端聚类...")
                      # 确定使用哪种聚类算法
                    clustering_method = 'vwc'
                    if hasattr(self.args, 'clustering_method'):
                        clustering_method = self.args.clustering_method
                        
                    if clustering_method == 'label_conditional':
                        # 执行基于标签条件分布的Wasserstein聚类
                        print("使用基于标签条件分布的Wasserstein聚类...")
                        success = self.label_conditional_clustering()
                        if not success:
                            print("标签条件聚类未能产生有效结果，回退到原始VWC聚类")
                            clustering_method = 'vwc'  # 回退到原始方法
                    
                    if clustering_method == 'enhanced_label':
                        # 使用我们新增的增强型标签条件聚类方法
                        print("使用增强型标签条件聚类（结合预测标签和中间表征）...")
                        # 执行增强型标签条件聚类
                        self.label_conditional_clustering()
                    elif clustering_method == 'vwc':
                        # 使用原始的VWC聚类
                        print("使用原始变分Wasserstein聚类...")
                        self.vwc_clustering()
                      # 如果使用原始VWC聚类或标签条件聚类失败
                    if clustering_method == 'vwc':
                        print("使用原始变分Wasserstein聚类...")
                        try:
                            # 初始化VWC模型
                            vwc_model = VariationalWassersteinClustering(
                                num_clients=len(self.selected_clients),
                                num_clusters=self.args.num_clusters,
                                proxy_dim=proxy_points.shape[-1]
                            ).to(self.device)
                            
                            # 训练VWC模型
                            optimizer = torch.optim.Adam(vwc_model.parameters(), lr=0.01)
                            
                            for epoch in range(50):  # VWC训练轮数
                                optimizer.zero_grad()
                                assignment_probs, loss = vwc_model(proxy_points)
                                loss.backward()
                                optimizer.step()
                                
                                if (epoch + 1) % 10 == 0:
                                    print(f"VWC Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")
                            
                            # 获取聚类分配
                            cluster_assignments = vwc_model.get_cluster_assignments()
                            
                            # 更新客户端的集群分配
                            for idx, client in enumerate(self.selected_clients):
                                new_cluster = cluster_assignments[idx].item()
                                # 记录历史分配
                                if client.id not in self.client_cluster_history:
                                    self.client_cluster_history[client.id] = []
                                self.client_cluster_history[client.id].append(new_cluster)
                                # 更新当前分配
                                self.clusters[client.id] = new_cluster
                                
                        except Exception as e:
                            print(f"VWC聚类失败: {str(e)}")
                            print("Details:", e.__class__.__name__)
                            import traceback
                            print(traceback.format_exc())
                            # 出错时保持原有集群分配
                
                # 更新每个集群的中心模型
                print("\nUpdating cluster models...")
                self.update_cluster_models()
                
                # 向客户端分发更新后的模型
                print("\nDistributing models to clients...")
                for client in self.selected_clients:
                    cluster_id = self.clusters.get(client.id, 0)
                    cluster_model = self.cluster_centroids.get(cluster_id, self.global_model)
                    client.receive_cluster_model(cluster_model)                # 评估当前模型
                if i % self.eval_gap == 0:
                    print(f"\n-------------Round {i}-------------")
                    print("\nEvaluating models...")
                    self.evaluate()                # 可视化聚类结果
                if i % self.eval_gap == 0:
                    print("\n可视化聚类结果...")
                    self.visualize_clustering(i)
                
                # 如果启用了概念漂移，更新迭代并保存进度
                if hasattr(self, 'drift_concepts_initialized') and self.drift_concepts_initialized:
                    # 更新当前迭代
                    if not hasattr(self, 'current_iteration'):
                        self.current_iteration = 0
                    else:
                        self.current_iteration += 1
                    
                    # 保存概念漂移进度
                    save_concept_progress(self, i)
                    
                    # 通知客户端迭代更新
                    print(f"\n更新客户端迭代状态到 {self.current_iteration}")
                    for client in self.selected_clients:
                        if hasattr(client, 'current_iteration'):
                            client.current_iteration = self.current_iteration

                # 记录本轮训练时间
                round_time = time.time() - s_t
                self.Budget.append(round_time)
                print("-" * 25, f"Round {i} time cost: {round_time:.2f}s", "-" * 25)

            except Exception as e:
                print(f"\nError in training round {i}: {str(e)}")
                continue            # 检查是否达到停止条件
            if self.auto_break and self.check_done(
                acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt            ):
                print("\nReached stopping criterion. Training complete.")
                break

        print("\nTraining completed!")
        if self.rs_test_acc:
            print("Best accuracy achieved:", max(self.rs_test_acc))
        else:
            print("Warning: No test accuracy records found")
        if len(self.Budget) > 1:
            print("Average time cost per round:", sum(self.Budget[1:]) / len(self.Budget[1:]))
        else:
            print("Warning: No time cost records found")
        # 评估最终聚类性能
        print("\n执行最终聚类性能评估...")
        self.evaluate_clustering_metrics()
        
        # 绘制聚类演变图
        print("\n绘制聚类演变图...")
        self.plot_clustering_evolution()
        
        # 保存聚类结果
        print("\n保存聚类结果...")
        self.save_clustering_results()
        
        # 保存结果
        self.save_results()
        self.save_models()
        self.save_clustering_results()

    def save_pretrain_models(self):
        """保存预训练模型"""
        if not os.path.exists("saved_models"):
            os.makedirs("saved_models")
            
        # 保存全局模型
        torch.save(self.global_model.state_dict(), 
                  os.path.join("saved_models", "FedDCA_server.pt"))
        
        # 保存自动编码器模型(如果存在)
        if hasattr(self, "autoencoder"):
            torch.save(self.autoencoder.state_dict(), 
                      os.path.join("saved_models", "FedDCA_server_autoencoder.pt"))

    def save_models(self):
        """保存全局模型和各个聚类中心模型"""
        # 创建保存目录
        if not os.path.exists("saved_models"):
            os.makedirs("saved_models")
            
        # 保存全局模型
        torch.save(self.global_model.state_dict(), 
                  os.path.join("saved_models", f"FedDCA_global_{self.dataset}.pt"))
        
        # 保存每个聚类中心模型
        for cluster_id, model in self.cluster_centroids.items():
            torch.save(model.state_dict(), 
                      os.path.join("saved_models", f"FedDCA_cluster_{cluster_id}_{self.dataset}.pt"))
        
        print(f"模型保存完成：1个全局模型和{len(self.cluster_centroids)}个聚类中心模型")
        
        # 如果有自动编码器，也保存它
        if hasattr(self, "autoencoder"):
            torch.save(self.autoencoder.state_dict(), 
                      os.path.join("saved_models", f"FedDCA_autoencoder_{self.dataset}.pt"))
            print("自动编码器模型已保存")

    def visualize_clustering(self, i):
        """可视化聚类结果
        
        Args:
            i: 当前轮次
        """        
        if len(self.selected_clients) == 0 or not hasattr(self, 'clusters'):
            print("没有足够的数据进行聚类可视化")
            return
            
        try:            # 收集所有客户端特征和聚类标签
            features = []
            cluster_labels = []
            client_ids = []
            
            # 如果使用真实概念，也收集真实标签
            true_labels = []
            has_true_concepts = self.use_drift_dataset and hasattr(self, 'client_concepts')
            
            for client in self.selected_clients:
                if hasattr(client, 'intermediate_output') and client.intermediate_output is not None:
                    feat = client.intermediate_output
                    
                    # 打印原始特征的类型和形状，辅助调试
                    # print(f"客户端 {client.id} 原始特征: 类型={type(feat)}, 形状={feat.shape if hasattr(feat, 'shape') else '标量'}")
                    
                    if not isinstance(feat, torch.Tensor):
                        feat = torch.tensor(feat)
                    
                    # 将特征转换为CPU和Numpy格式
                    feat = feat.detach().cpu().numpy()
                    
                    # 检查特征的维度
                    if len(feat.shape) > 1:
                        # 如果是2D以上的张量，取平均值降为1D向量
                        feat = np.mean(feat, axis=0)
                        #print(f"客户端 {client.id} 降维后特征形状: {feat.shape}")
                    
                    # 确保特征是一维数组，便于后续处理
                    if len(feat.shape) == 0:
                        # 标量转为一元数组
                        feat = np.array([float(feat)])
                        #print(f"客户端 {client.id} 标量特征转换为数组: {feat}")
                    
                    # 确保特征不包含无效值(NaN或Inf)
                    if np.isnan(feat).any() or np.isinf(feat).any():
                        #print(f"警告: 客户端 {client.id} 特征包含无效值，将被替换为0")
                        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    features.append(feat)
                    #print(f"客户端 {client.id} 最终特征形状: {feat.shape}")
                    cluster_labels.append(self.clusters.get(client.id, 0))
                    client_ids.append(client.id)
                    
                    # 如果有真实概念，记录客户端的真实概念
                    if has_true_concepts:
                        # 在client_concepts中查找真实概念（可能有多个）
                        if str(client.id) in self.client_concepts:
                            # 使用第一个概念作为标签
                            concept = self.client_concepts[str(client.id)][0]
                            true_labels.append(concept)
                        else:
                            true_labels.append(-1)  # 未知概念
            
            if not features:
                print("没有特征数据可以可视化")
                return
                
            # 将特征转换为numpy数组
            features = np.array(features)
            cluster_labels = np.array(cluster_labels)
            
            # 创建结果目录
            if not os.path.exists('results/clustering'):
                os.makedirs('results/clustering')
                  # 设置绘图
            plt.figure(figsize=(12, 10))
            
            # 使用t-SNE将特征降至2维
            print("执行t-SNE降维...")
            # 根据样本数动态调整perplexity参数，避免"perplexity must be less than n_samples"错误
            n_samples = len(features)
            perplexity = min(30, n_samples - 1)  # 确保perplexity小于样本数
            
            if n_samples <= 2:
                print("样本数太少，无法执行t-SNE，使用PCA代替")
                pca = PCA(n_components=2, random_state=42)
                features_2d = pca.fit_transform(features)
            else:
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                features_2d = tsne.fit_transform(features)
            
            # 绘制聚类结果
            plt.subplot(2, 2, 1)
            scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7, s=50)
            plt.title(f'Clustering Results (Round {i}, {len(set(cluster_labels))} clusters)')
            plt.colorbar(scatter, label='Cluster ID')
            plt.xlabel('t-SNE Feature 1')
            plt.ylabel('t-SNE Feature 2')
            # 为每个点添加客户端ID标签
            for j, client_id in enumerate(client_ids):
                plt.annotate(str(client_id), (features_2d[j, 0], features_2d[j, 1]), fontsize=8)
                
            # 如果有真实概念，绘制真实标签
            if has_true_concepts and len(true_labels) > 0:
                plt.subplot(2, 2, 2)
                true_labels = np.array(true_labels)
                scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=true_labels, cmap='tab10', alpha=0.7, s=50)
                plt.title(f'True Concept Distribution ({len(set(true_labels))} concepts)')
                plt.colorbar(scatter, label='Concept ID')
                plt.xlabel('t-SNE Feature 1')
                plt.ylabel('t-SNE Feature 2')
                
                # 计算聚类准确性指标
                # 删除未知标签
                valid_indices = true_labels != -1
                if np.sum(valid_indices) > 1:
                    valid_true_labels = true_labels[valid_indices]
                    valid_cluster_labels = cluster_labels[valid_indices]
                    
                    # 计算调整兰德指数 (ARI)
                    ari = adjusted_rand_score(valid_true_labels, valid_cluster_labels)
                    # 添加ARI分数文本标签
                    plt.subplot(2, 2, 3)
                    plt.annotate(f'Adjusted Rand Index (ARI): {ari:.4f}', 
                                xy=(0.05, 0.95), 
                                xycoords='axes fraction',
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
            
            # 计算并绘制轮廓系数
            if len(set(cluster_labels)) > 1 and len(features) > len(set(cluster_labels)):
                try:
                    # 计算额外的评估指标
                    print("Calculating clustering evaluation metrics...")
                    metrics = self.calculate_additional_metrics(features, cluster_labels)
                    
                    # 绘制评估指标
                    plt.subplot(2, 2, 4)
                    metrics_names = []
                    metrics_values = []
                    
                    if 'silhouette_score' in metrics:
                        metrics_names.append('Silhouette\nCoefficient')
                        metrics_values.append(metrics['silhouette_score'])
                    
                    if 'davies_bouldin_score' in metrics:
                        # 为了统一显示方向（越大越好），取Davies-Bouldin的负值
                        metrics_names.append('Davies-Bouldin\n(negative)')
                        metrics_values.append(-metrics['davies_bouldin_score'])
                    
                    if 'calinski_harabasz_score' in metrics:
                        # 归一化Calinski-Harabasz分数，因为它通常很大
                        normalized_ch = min(1.0, metrics['calinski_harabasz_score'] / 10000)
                        metrics_names.append('Calinski-Harabasz\n(normalized)')
                        metrics_values.append(normalized_ch)
                      # 绘制指标柱状图
                    if metrics_values:
                        bar_colors = ['skyblue', 'salmon', 'lightgreen'][:len(metrics_values)]
                        plt.bar(range(len(metrics_values)), metrics_values, color=bar_colors)
                        plt.title(f'Clustering Evaluation Metrics')
                        plt.xticks(range(len(metrics_values)), metrics_names)
                        plt.ylabel('Score Value')
                        plt.ylim(-1, 1)  # 统一刻度
                        
                        # 添加数值标签
                        for i, v in enumerate(metrics_values):
                            plt.annotate(f"{v:.2f}", 
                                        xy=(i, v), 
                                        xytext=(0, 3 if v >= 0 else -10),
                                        textcoords="offset points",
                                        ha='center')
                    
                except Exception as e:
                    print(f"Error plotting evaluation metrics: {str(e)}")
                    # 如果绘制失败，回退到只显示轮廓系数
                    try:
                        silhouette_avg = silhouette_score(features, cluster_labels)
                        plt.subplot(2, 2, 4)
                        plt.bar(range(1), [silhouette_avg], color='skyblue')
                        plt.title(f'Silhouette Coefficient: {silhouette_avg:.4f}')
                        plt.xticks([])
                        plt.ylabel('Silhouette Score')
                        plt.ylim(-1, 1)
                        # 添加评估文本
                        quality_text = 'Poor' if silhouette_avg < 0.2 else 'Fair' if silhouette_avg < 0.5 else 'Good' if silhouette_avg < 0.7 else 'Excellent'
                        plt.annotate(f'Clustering Quality: {quality_text}', 
                                    xy=(0.1, 0.5), 
                                    xycoords='axes fraction',
                                    fontsize=12)
                    except Exception as e:
                        print(f"Error calculating silhouette coefficient: {str(e)}")            # 绘制聚类分布柱状图
            plt.subplot(2, 2, 4)
            cluster_counts = {}
            for cluster in cluster_labels:
                if cluster not in cluster_counts:
                    cluster_counts[cluster] = 0
                cluster_counts[cluster] += 1
            
            clusters = sorted(cluster_counts.keys())
            counts = [cluster_counts[c] for c in clusters]
            
            plt.bar(clusters, counts, color='salmon')
            plt.title(f'Cluster Distribution (Round {i})')
            plt.xlabel('Cluster ID')
            plt.ylabel('Client Count')
            
            # 添加数量标签
            for j, count in enumerate(counts):
                plt.annotate(str(count),
                             xy=(clusters[j], count),
                             xytext=(0, 3),
                             textcoords="offset points",
                             ha='center')
            
            # 保存图表
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            plt.tight_layout()
            plt.savefig(f'results/clustering/clustering_round_{i}_{timestamp}.png', dpi=300)
            print(f"Clustering visualization saved to 'results/clustering/clustering_round_{i}_{timestamp}.png'")
            plt.close()
            
            # 如果存在历史聚类分配，绘制聚类稳定性图
            if hasattr(self, 'client_cluster_history') and len(self.client_cluster_history) > 0:
                plt.figure(figsize=(10, 6))
                
                # 选择几个有完整历史的客户端来绘制
                complete_clients = [cid for cid, history in self.client_cluster_history.items() 
                                    if len(history) >= min(i+1, 10)]
                
                if len(complete_clients) > 0:
                    # 如果客户端太多，只选择一部分
                    selected_clients = complete_clients[:min(len(complete_clients), 10)]
                    
                    for cid in selected_clients:
                        history = self.client_cluster_history[cid][-min(i+1, 10):]  # 最近10轮
                        plt.plot(range(len(history)), history, 'o-', label=f'Client {cid}')
                    
                    plt.title('Client Clustering Stability Analysis')
                    plt.xlabel('Round (Last 10 rounds)')
                    plt.ylabel('Cluster Assignment')
                    plt.legend(loc='best')
                    plt.grid(True)
                    plt.savefig(f'results/clustering/cluster_stability_{i}_{timestamp}.png', dpi=300)
                    print(f"Clustering stability graph saved to 'results/clustering/cluster_stability_{i}_{timestamp}.png'")
                    plt.close()
        except Exception as e:
            print(f"Error during clustering visualization: {str(e)}")
            print(traceback.format_exc())

    def evaluate_clustering_metrics(self):
        """Evaluate clustering performance metrics and generate summary report"""
        if not hasattr(self, 'clusters') or len(self.clusters) == 0:
            print("No clustering data available for evaluation")
            return

        try:
            print("\n======= Clustering Performance Evaluation =======")
              # 1. 簇分布情况
            cluster_distribution = {}
            for client_id, cluster_id in self.clusters.items():
                if cluster_id not in cluster_distribution:
                    cluster_distribution[cluster_id] = 0
                cluster_distribution[cluster_id] += 1
            
            print("\nCluster Distribution:")
            for cluster_id, count in sorted(cluster_distribution.items()):
                print(f"  Cluster {cluster_id}: {count} clients ({count/len(self.clusters)*100:.1f}%)")
                
            # 2. 聚类稳定性 - 计算最近几轮的平均变化率
            if hasattr(self, 'client_cluster_history') and len(self.client_cluster_history) > 0:
                changes = []
                for client_id, history in self.client_cluster_history.items():
                    if len(history) >= 2:
                        # 计算最近几轮中发生变化的比例
                        recent_history = history[-min(len(history), 5):]  # 最近5轮
                        change_count = sum(1 for i in range(1, len(recent_history)) 
                                          if recent_history[i] != recent_history[i-1])
                        change_rate = change_count / (len(recent_history) - 1) if len(recent_history) > 1 else 0
                        changes.append(change_rate)
                
                if changes:
                    avg_change_rate = sum(changes) / len(changes)
                    print(f"\nClustering Stability:")
                    print(f"  Average Cluster Change Rate: {avg_change_rate:.2f} (0=completely stable, 1=changes every round)")
                    stability_level = 'Very High' if avg_change_rate < 0.1 else 'High' if avg_change_rate < 0.3 else 'Medium' if avg_change_rate < 0.5 else 'Low'
                    print(f"  Stability Level: {stability_level}")
              # 3. 概念对齐分析 (如果有真实概念标签)
            if self.use_drift_dataset and hasattr(self, 'client_concepts'):
                print("\nConcept-Cluster Alignment Analysis:")
                
                concept_cluster_map = {}  # 映射概念到集群分布
                cluster_concept_map = {}  # 映射集群到概念分布
                
                alignments = 0
                total = 0
                
                for client_id, cluster_id in self.clusters.items():
                    # 获取客户端的真实概念(如果存在)
                    if str(client_id) in self.client_concepts:
                        # 使用第一个概念作为主要概念
                        concept = self.client_concepts[str(client_id)][0]
                        
                        # 更新映射
                        if concept not in concept_cluster_map:
                            concept_cluster_map[concept] = {}
                        if cluster_id not in concept_cluster_map[concept]:
                            concept_cluster_map[concept][cluster_id] = 0
                        concept_cluster_map[concept][cluster_id] += 1
                        
                        if cluster_id not in cluster_concept_map:
                            cluster_concept_map[cluster_id] = {}
                        if concept not in cluster_concept_map[cluster_id]:
                            cluster_concept_map[cluster_id][concept] = 0
                        cluster_concept_map[cluster_id][concept] += 1
                        
                        total += 1
                
                # 计算每个概念对应的主要集群
                concept_primary_clusters = {}
                for concept, clusters in concept_cluster_map.items():
                    primary_cluster = max(clusters, key=clusters.get)
                    concept_primary_clusters[concept] = primary_cluster
                    print(f"  Concept {concept} primarily maps to Cluster {primary_cluster} ({clusters[primary_cluster]/sum(clusters.values())*100:.1f}%)")
                    
                    # 计算该概念的客户端被正确分类的比例
                    correct_assignments = clusters[primary_cluster]
                    total_concept_clients = sum(clusters.values())
                    alignments += correct_assignments
                    print(f"    Alignment Rate: {correct_assignments/total_concept_clients*100:.1f}%")
                
                # 计算总体对齐率
                overall_alignment = alignments / total if total > 0 else 0
                print(f"\nOverall Concept-Cluster Alignment Rate: {overall_alignment*100:.1f}%")
                alignment_quality = 'Excellent' if overall_alignment > 0.8 else 'Good' if overall_alignment > 0.6 else 'Fair' if overall_alignment > 0.4 else 'Poor'
                print(f"Alignment Quality: {alignment_quality}")
                  # 绘制概念-聚类对齐热力图
                if concept_cluster_map and cluster_concept_map:
                    plt.figure(figsize=(10, 8))
                    
                    # 创建热力图数据
                    concepts = sorted(concept_cluster_map.keys())
                    clusters = sorted(cluster_concept_map.keys())
                    
                    alignment_matrix = np.zeros((len(concepts), len(clusters)))
                    
                    for i, concept in enumerate(concepts):
                        for j, cluster in enumerate(clusters):
                            # 如果这个概念-集群对存在
                            if concept in concept_cluster_map and cluster in concept_cluster_map[concept]:
                                # 计算概念中有多少比例分到这个集群
                                alignment_matrix[i, j] = concept_cluster_map[concept].get(cluster, 0) / sum(concept_cluster_map[concept].values())
                    
                    # 绘制热力图
                    plt.imshow(alignment_matrix, cmap='YlOrRd')
                    plt.colorbar(label='Concept to Cluster Assignment Ratio')
                    plt.xticks(range(len(clusters)), [f'Cluster {c}' for c in clusters], rotation=45)
                    plt.yticks(range(len(concepts)), [f'Concept {c}' for c in concepts])
                    plt.title('Concept-Cluster Alignment Heatmap')
                    
                    for i in range(len(concepts)):
                        for j in range(len(clusters)):
                            text = f"{alignment_matrix[i, j]*100:.1f}%" if alignment_matrix[i, j] > 0 else ""
                            color = "white" if alignment_matrix[i, j] > 0.5 else "black"
                            plt.text(j, i, text, ha="center", va="center", color=color)
                    
                    # 保存热力图
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    plt.tight_layout()
                    if not os.path.exists('results/clustering'):
                        os.makedirs('results/clustering')
                    plt.savefig(f'results/clustering/concept_cluster_alignment_{timestamp}.png', dpi=300)
                    print(f"\nConcept-Cluster alignment heatmap saved to 'results/clustering/concept_cluster_alignment_{timestamp}.png'")
                    plt.close()
              # 4. 跨集群性能比较
            print("\nCross-Cluster Performance Comparison:")
            cluster_performance = {}
            
            for client in self.selected_clients:
                cluster_id = self.clusters.get(client.id, 0)
                if cluster_id not in cluster_performance:
                    cluster_performance[cluster_id] = {'acc': [], 'loss': []}
                
                # 收集性能指标
                try:
                    test_acc, test_num, _ = client.test_metrics()
                    train_loss, train_num = client.train_metrics()
                    
                    if test_num > 0:
                        cluster_performance[cluster_id]['acc'].append(test_acc / test_num)
                    if train_num > 0:
                        cluster_performance[cluster_id]['loss'].append(train_loss / train_num)
                except Exception as e:
                    print(f"Failed to get performance metrics from client {client.id}: {str(e)}")
            
            # 计算并显示每个集群的平均性能
            for cluster_id, metrics in sorted(cluster_performance.items()):
                avg_acc = np.mean(metrics['acc']) if metrics['acc'] else 0
                avg_loss = np.mean(metrics['loss']) if metrics['loss'] else 0
                client_count = len(metrics['acc'])
                
                print(f"  Cluster {cluster_id} ({client_count} clients):")
                print(f"    Average Accuracy: {avg_acc*100:.2f}%")
                print(f"    Average Loss: {avg_loss:.4f}")
            
            # 绘制集群性能对比图
            plt.figure(figsize=(12, 5))
            
            # 准确率对比
            plt.subplot(1, 2, 1)
            clusters = []
            accuracies = []
            std_devs = []
            counts = []
            
            for cluster_id, metrics in sorted(cluster_performance.items()):
                if metrics['acc']:
                    clusters.append(cluster_id)
                    avg_acc = np.mean(metrics['acc'])
                    std_dev = np.std(metrics['acc']) if len(metrics['acc']) > 1 else 0
                    accuracies.append(avg_acc)
                    std_devs.append(std_dev)
                    counts.append(len(metrics['acc']))
            
            if clusters:
                plt.bar(range(len(clusters)), accuracies, yerr=std_devs, capsize=5, alpha=0.7, color='skyblue')
                plt.xticks(range(len(clusters)), [f'Cluster {c}\n({counts[i]})' for i, c in enumerate(clusters)])
                plt.title('Cluster Accuracy Comparison')
                plt.ylabel('Accuracy')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # 添加具体数值
                for i, acc in enumerate(accuracies):
                    plt.text(i, acc + 0.01, f"{acc*100:.1f}%", ha='center')
                
                # 损失对比
                plt.subplot(1, 2, 2)
                clusters = []
                losses = []
                std_devs = []
                counts = []
                
                for cluster_id, metrics in sorted(cluster_performance.items()):
                    if metrics['loss']:
                        clusters.append(cluster_id)
                        avg_loss = np.mean(metrics['loss'])
                        std_dev = np.std(metrics['loss']) if len(metrics['loss']) > 1 else 0
                        losses.append(avg_loss)
                        std_devs.append(std_dev)
                        counts.append(len(metrics['loss']))
                
                plt.bar(range(len(clusters)), losses, yerr=std_devs, capsize=5, alpha=0.7, color='salmon')
                plt.xticks(range(len(clusters)), [f'Cluster {c}\n({counts[i]})' for i, c in enumerate(clusters)])
                plt.title('Cluster Loss Comparison')
                plt.ylabel('Loss')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # 添加具体数值
                for i, loss in enumerate(losses):
                    plt.text(i, loss + 0.05, f"{loss:.2f}", ha='center')
                
                plt.tight_layout()
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                if not os.path.exists('results/clustering'):
                    os.makedirs('results/clustering')
                plt.savefig(f'results/clustering/cluster_performance_comparison_{timestamp}.png', dpi=300)
                print(f"\nCluster performance comparison chart saved to 'results/clustering/cluster_performance_comparison_{timestamp}.png'")
                plt.close()
                
            print("\n======= Clustering Evaluation Completed =======")
        except Exception as e:
            print(f"Error evaluating clustering metrics: {str(e)}")
            print(traceback.format_exc())
            
    def save_clustering_results(self):
        """Save clustering results to files"""
        try:
            if not os.path.exists('results/clustering'):
                os.makedirs('results/clustering')
                
            # 保存客户端聚类分配
            cluster_assignments = {}
            for client_id, cluster_id in self.clusters.items():
                cluster_assignments[str(client_id)] = int(cluster_id)
                
            # 保存到JSON文件
            import json
            with open('results/clustering/final_cluster_assignments.json', 'w') as f:
                json.dump(cluster_assignments, f, indent=4)
            
            # 保存聚类历史
            history_data = {}
            for client_id, history in self.client_cluster_history.items():
                history_data[str(client_id)] = [int(x) for x in history]
            
            with open('results/clustering/cluster_history.json', 'w') as f:
                json.dump(history_data, f, indent=4)                  # 如果有真实概念标签，进行概念-聚类映射分析
            if self.use_drift_dataset and hasattr(self, 'client_concepts'):
                # 计算概念和聚类的对应关系
                concept_cluster_map = {}
                for client_id, cluster_id in self.clusters.items():
                    if str(client_id) in self.client_concepts:
                        concept = self.client_concepts[str(client_id)][0]  # 使用第一个概念
                        if concept not in concept_cluster_map:
                            concept_cluster_map[concept] = {}
                        if cluster_id not in concept_cluster_map[concept]:
                            concept_cluster_map[concept][cluster_id] = 0
                        concept_cluster_map[concept][cluster_id] += 1
                
                # 保存概念-聚类映射
                with open('results/clustering/concept_cluster_mapping.json', 'w') as f:
                    json.dump(concept_cluster_map, f, indent=4)
                    
                # 计算调整兰德指数(ARI)
                try:
                    from sklearn.metrics import adjusted_rand_score
                    true_labels = []
                    cluster_labels = []
                    
                    for client_id, cluster_id in self.clusters.items():
                        if str(client_id) in self.client_concepts:
                            true_labels.append(self.client_concepts[str(client_id)][0])
                            cluster_labels.append(cluster_id)
                    
                    if len(true_labels) > 1 and len(set(true_labels)) > 1 and len(set(cluster_labels)) > 1:
                        ari = adjusted_rand_score(true_labels, cluster_labels)
                        
                        # 收集客户端特征，用于计算聚类评估指标
                        features = []
                        for client in self.selected_clients:
                            if hasattr(client, 'intermediate_output') and client.intermediate_output is not None:
                                feat = client.intermediate_output
                                if not isinstance(feat, torch.Tensor):
                                    feat = torch.tensor(feat)
                                
                                # 将特征转换为CPU和Numpy格式
                                feat = feat.detach().cpu().numpy()
                                
                                # 如果特征是多维的，先平均降到2D
                                if len(feat.shape) > 1:
                                    feat = np.mean(feat, axis=0)
                                
                                features.append(feat)
                        
                        # 如果有足够的特征，计算额外的聚类指标
                        extra_metrics = {}
                        if len(features) > 1:
                            features = np.array(features)
                            selected_clients_ids = [client.id for client in self.selected_clients]
                            selected_clusters = [self.clusters.get(cid, 0) for cid in selected_clients_ids]
                            extra_metrics = self.calculate_additional_metrics(features, selected_clusters)
                        
                        # 保存所有评估指标
                        with open('results/clustering/clustering_metrics.txt', 'w') as f:
                            f.write(f"Adjusted Rand Index (ARI): {ari:.4f}\n")
                            for metric_name, metric_value in extra_metrics.items():
                                f.write(f"{metric_name}: {metric_value:.4f}\n")
                            
                        print(f"Final clustering Adjusted Rand Index (ARI): {ari:.4f}")
                except Exception as e:
                    print(f"Error calculating clustering metrics: {str(e)}")
            print("Clustering results saved to 'results/clustering/' directory")
        except Exception as e:
            print(f"Error saving clustering results: {str(e)}")
            print(traceback.format_exc())
            
    def plot_clustering_evolution(self, save_path='results/clustering/clustering_evolution.png'):
        """Plot the evolution of clustering over time
        
        Args:
            save_path: Path to save the chart
        """
        if not hasattr(self, 'client_cluster_history') or len(self.client_cluster_history) == 0:
            print("Not enough clustering history data for visualization")
            return
            
        try:
            # 计算每轮的聚类统计信息
            round_stats = {}
            max_rounds = 0
            
            # 获取历史中的最大轮次
            for client_id, history in self.client_cluster_history.items():
                max_rounds = max(max_rounds, len(history))
            
            # 初始化统计信息
            for i in range(max_rounds):
                round_stats[i] = {'cluster_counts': {}, 'changes': 0, 'clients': 0}
            
            # 计算每轮的聚类分配和变化率
            for client_id, history in self.client_cluster_history.items():
                for i, cluster in enumerate(history):
                    # 更新该轮的聚类计数
                    if cluster not in round_stats[i]['cluster_counts']:
                        round_stats[i]['cluster_counts'][cluster] = 0
                    round_stats[i]['cluster_counts'][cluster] += 1
                    round_stats[i]['clients'] += 1
                    
                    # 计算聚类变化（从第二轮开始）
                    if i > 0 and history[i] != history[i-1]:
                        round_stats[i]['changes'] += 1
            
            # 计算每轮的变化率
            change_rates = []
            cluster_counts = []
            rounds = []
            
            for i in sorted(round_stats.keys()):
                if i > 0:  # 从第二轮开始计算变化率
                    change_rate = round_stats[i]['changes'] / round_stats[i]['clients'] if round_stats[i]['clients'] > 0 else 0
                    change_rates.append(change_rate)
                    cluster_counts.append(len(round_stats[i]['cluster_counts']))
                    rounds.append(i)
              # 创建图表
            plt.figure(figsize=(12, 8))
            
            # 绘制变化率
            plt.subplot(2, 1, 1)
            plt.plot(rounds, change_rates, 'o-', color='blue', label='Change Rate')
            plt.title('Evolution of Clustering Change Rate')
            plt.xlabel('Round')
            plt.ylabel('Change Rate')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # 为突出的变化点添加标注
            threshold = 0.3  # 高变化率阈值
            for i, rate in enumerate(change_rates):
                if rate > threshold:
                    plt.annotate(f"{rate:.2f}", 
                                xy=(rounds[i], rate),
                                xytext=(rounds[i], rate + 0.05),
                                arrowprops=dict(arrowstyle="->", color='red'),
                                ha='center')
              # 绘制聚类数量
            plt.subplot(2, 1, 2)
            plt.plot(rounds, cluster_counts, 'o-', color='green', label='Cluster Count')
            plt.title('Evolution of Active Cluster Count')
            plt.xlabel('Round')
            plt.ylabel('Active Cluster Count')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # 突出显示簇数量变化的点
            for i in range(1, len(cluster_counts)):
                if cluster_counts[i] != cluster_counts[i-1]:
                    plt.annotate(f"{cluster_counts[i]}", 
                                xy=(rounds[i], cluster_counts[i]),
                                xytext=(rounds[i], cluster_counts[i] + 0.3),
                                arrowprops=dict(arrowstyle="->", color='red'),
                                ha='center')
            
            # 保存图表
            plt.tight_layout()
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300)
            print(f"Clustering evolution chart saved to '{save_path}'")
            plt.close()
              # 绘制客户端聚类轨迹图
            plt.figure(figsize=(14, 10))
            
            # 选择一部分有代表性的客户端进行可视化
            selected_clients = list(self.client_cluster_history.keys())
            if len(selected_clients) > 20:  # 如果客户端太多，只选择前20个
                selected_clients = selected_clients[:20]
                
            # 为每个选择的客户端绘制聚类轨迹
            for client_id in selected_clients:
                history = self.client_cluster_history[client_id]
                plt.plot(range(len(history)), history, '-', label=f'Client {client_id}')
            
            plt.title('Client Clustering Trajectory Evolution')
            plt.xlabel('Round')
            plt.ylabel('Cluster Assignment')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 设置y轴刻度为整数
            plt.yticks(np.arange(0, max(max(history) for history in self.client_cluster_history.values()) + 1))
            
            # 如果客户端数量适中，添加图例
            if len(selected_clients) <= 10:
                plt.legend(loc='best')
            
            # 保存轨迹图
            trajectory_path = os.path.join(os.path.dirname(save_path), 'client_trajectories.png')            
            plt.savefig(trajectory_path, dpi=300)
            print(f"Client clustering trajectory chart saved to '{trajectory_path}'")
            plt.close()
        except Exception as e:
            print(f"Error plotting clustering evolution: {str(e)}")
            print(traceback.format_exc())
            
    def calculate_additional_metrics(self, features, cluster_labels):
        """Calculate additional clustering evaluation metrics
        
        Args:
            features: Feature data with shape (n_samples, n_features)
            cluster_labels: Cluster labels with shape (n_samples,)
            
        Returns:
            dict: Dictionary containing various clustering evaluation metrics
        """
        # 初始化结果字典
        metrics = {}
        
        try:
            # 检查是否满足计算指标的条件
            if len(set(cluster_labels)) < 2:
                print("Less than 2 clusters, cannot calculate evaluation metrics")
                return metrics
                
            if len(features) <= len(set(cluster_labels)):
                print("Not enough samples to reliably calculate evaluation metrics")
                return metrics                
            # 计算轮廓系数 (Silhouette Coefficient)
            # 衡量样本与自己所在簇的相似度与其他簇的不相似度
            # 范围：[-1, 1]，越接近1越好
            silhouette_avg = silhouette_score(features, cluster_labels)
            metrics['silhouette_score'] = silhouette_avg
            
            # 计算Davies-Bouldin指数
            # 衡量簇内距离与簇间距离的比率
            # 范围：[0, ∞)，越小越好
            db_score = davies_bouldin_score(features, cluster_labels)
            metrics['davies_bouldin_score'] = db_score
            
            # 计算Calinski-Harabasz指数 (Variance Ratio Criterion)
            # 簇间方差与簇内方差的比率
            # 范围：[0, ∞)，越大越好
            ch_score = calinski_harabasz_score(features, cluster_labels)
            metrics['calinski_harabasz_score'] = ch_score
            
            # 如果有真实标签，计算聚类与真实标签的一致性
            if hasattr(self, 'client_concepts') and self.use_drift_dataset:
                try:
                    # 收集客户端的真实概念标签
                    true_labels = []
                    client_ids_with_features = [client.id for client in self.selected_clients 
                                                if hasattr(client, 'intermediate_output') and 
                                                client.intermediate_output is not None]
                    
                    for client_id in client_ids_with_features:
                        if str(client_id) in self.client_concepts:
                            true_labels.append(self.client_concepts[str(client_id)][0])
                        else:
                            true_labels.append(-1)
                      # 过滤掉未知标签
                    valid_indices = [i for i, label in enumerate(true_labels) if label != -1]
                    
                    if len(valid_indices) >= 2 and len(set([true_labels[i] for i in valid_indices])) >= 2:
                        valid_true_labels = [true_labels[i] for i in valid_indices]
                        valid_cluster_labels = [cluster_labels[i] for i in valid_indices]
                        
                        # 计算调整兰德指数 (ARI)
                        # 衡量两个聚类结果的相似度
                        # 范围：[-1, 1]，越接近1越好，0表示随机分配
                        ari = adjusted_rand_score(valid_true_labels, valid_cluster_labels)
                        metrics['adjusted_rand_score'] = ari
                
                except Exception as label_error:
                    print(f"Error calculating label consistency metrics: {str(label_error)}")
            
                return metrics
            
        except Exception as metric_error:
            print(f"Error calculating additional metrics: {str(metric_error)}")
            return metrics
            
    def compute_label_conditional_wasserstein_distance(self, client_features):
        """
        计算客户端之间基于标签条件的距离
        
        参数:
            client_features: 按客户端ID和标签组织的特征字典 {client_id: {label: features}}
            
        返回:
            numpy.ndarray: 客户端间的距离矩阵
        """
        client_ids = list(client_features.keys())
        n_clients = len(client_ids)
        
        if n_clients <= 1:
            print("Warning: 不足以计算距离，只有一个或零个客户端")
            return np.zeros((n_clients, n_clients))
        
        # 初始化距离矩阵
        distance_matrix = np.zeros((n_clients, n_clients))
        
        # 计算所有标签列表
        all_labels = set()
        for client_id in client_ids:
            all_labels.update(client_features[client_id].keys())
        
        # 为每对客户端计算距离
        for i in range(n_clients):
            for j in range(i+1, n_clients):
                client_i_id = client_ids[i]
                client_j_id = client_ids[j]
                
                # 计算每个标签的距离并加权平均
                label_distances = []
                label_weights = []
                
                for label in all_labels:
                    # 检查两个客户端都有此标签的数据
                    if (label in client_features[client_i_id] and 
                        label in client_features[client_j_id]):
                        
                        features_i = client_features[client_i_id][label]
                        features_j = client_features[client_j_id][label]
                        
                        # 确保特征是numpy数组
                        if isinstance(features_i, torch.Tensor):
                            features_i = features_i.detach().cpu().numpy()
                        if isinstance(features_j, torch.Tensor):
                            features_j = features_j.detach().cpu().numpy()
                        
                        # 检查并处理形状问题
                        if features_i.ndim == 1:
                            features_i = features_i.reshape(1, -1)
                        if features_j.ndim == 1:
                            features_j = features_j.reshape(1, -1)
                        
                        # 确保两个特征矩阵有相同的维度
                        if features_i.shape[1] != features_j.shape[1]:
                            min_dim = min(features_i.shape[1], features_j.shape[1])
                            features_i = features_i[:, :min_dim]
                            features_j = features_j[:, :min_dim]
                        
                        # 计算样本数，用作权重
                        weight_i = features_i.shape[0]
                        weight_j = features_j.shape[0]
                        label_weight = (weight_i + weight_j) / 2
                        
                        try:
                            # 标准化特征以确保可比性
                            features_i_mean = np.mean(features_i, axis=0, keepdims=True)
                            features_i_std = np.std(features_i, axis=0, keepdims=True) + 1e-8
                            features_i_normalized = (features_i - features_i_mean) / features_i_std
                            
                            features_j_mean = np.mean(features_j, axis=0, keepdims=True)
                            features_j_std = np.std(features_j, axis=0, keepdims=True) + 1e-8
                            features_j_normalized = (features_j - features_j_mean) / features_j_std
                            
                            # 处理NaN值
                            features_i_normalized = np.nan_to_num(features_i_normalized)
                            features_j_normalized = np.nan_to_num(features_j_normalized)
                            
                            # 使用欧氏距离计算距离矩阵
                            dist = 0
                            try:
                                # 如果POT库可用，使用Wasserstein距离
                                if 'ot' in globals():
                                    # 限制样本数量
                                    n_samples_i = min(features_i_normalized.shape[0], 100)
                                    n_samples_j = min(features_j_normalized.shape[0], 100)
                                    
                                    a = np.ones(n_samples_i) / n_samples_i
                                    b = np.ones(n_samples_j) / n_samples_j
                                    
                                    # 计算成本矩阵 (欧氏距离)
                                    M = ot.dist(features_i_normalized[:n_samples_i], 
                                              features_j_normalized[:n_samples_j])
                                    
                                    # 规范化成本矩阵
                                    if np.max(M) > 0:
                                        M = M / np.max(M)
                                    
                                    # 使用Sinkhorn算法计算Wasserstein距离
                                    reg = 0.1  # 正则化参数
                                    dist = ot.sinkhorn2(a, b, M, reg)[0]
                                else:
                                    # 使用均值和协方差的距离作为替代
                                    # 计算均值距离
                                    mean_dist = np.linalg.norm(features_i_mean - features_j_mean)
                                    
                                    # 特征维度可能很大，使用更稳健的方法
                                    try:
                                        # 计算协方差矩阵
                                        cov_i = np.cov(features_i_normalized.T)
                                        cov_j = np.cov(features_j_normalized.T)
                                        
                                        # 计算协方差矩阵之间的距离
                                        # 如果维度太高，仅使用对角线元素
                                        if cov_i.shape[0] > 100:
                                            cov_dist = np.linalg.norm(np.diag(cov_i) - np.diag(cov_j))
                                        else:
                                            cov_dist = np.linalg.norm(cov_i - cov_j, 'fro')
                                        
                                        dist = mean_dist + cov_dist
                                    except Exception as cov_error:
                                        # 计算协方差失败，只使用均值距离
                                        print(f"协方差计算失败: {str(cov_error)}，仅使用均值距离")
                                        dist = mean_dist
                            except Exception as dist_error:
                                print(f"使用Wasserstein距离失败，回退到均值距离: {str(dist_error)}")
                                # 如果出错，使用均值距离作为替代
                                dist = np.linalg.norm(np.mean(features_i_normalized, axis=0) - 
                                                     np.mean(features_j_normalized, axis=0))
                            
                            label_distances.append(dist)
                            label_weights.append(label_weight)
                        
                        except Exception as norm_error:
                            print(f"计算标签{label}的距离时出错: {str(norm_error)}")
                            # 尝试使用原始特征的简单距离
                            try:
                                simple_dist = np.linalg.norm(np.mean(features_i, axis=0) - np.mean(features_j, axis=0))
                                label_distances.append(simple_dist)
                                label_weights.append(label_weight)
                            except Exception as simple_error:
                                print(f"计算简单距离也失败: {str(simple_error)}")
                
                # 计算加权平均距离
                if label_distances:
                    total_weight = sum(label_weights)
                    if total_weight > 0:
                        weighted_distance = sum(d * w for d, w in zip(label_distances, label_weights)) / total_weight
                    else:
                        weighted_distance = np.mean(label_distances)
                    
                    distance_matrix[i, j] = weighted_distance
                    distance_matrix[j, i] = weighted_distance
                else:
                    # 没有共同标签，使用一个大距离
                    distance_matrix[i, j] = 1.0
                    distance_matrix[j, i] = 1.0
        
        return distance_matrix
