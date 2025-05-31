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
import wandb # Added wandb import

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

        # # 添加命令行参数支持
        # if hasattr(args, 'cmd_args') and args.cmd_args:
        #     # 如果命令行中指定了这些参数，则覆盖默认值
        #     if hasattr(args.cmd_args, 'use_drift_dataset'):
        #         self.use_drift_dataset = args.cmd_args.use_drift_dataset
        #     if hasattr(args.cmd_args, 'drift_data_dir'):
        #         self.drift_data_dir = args.cmd_args.drift_data_dir
        #     if hasattr(args.cmd_args, 'max_iterations'):
        #         self.max_iterations = args.cmd_args.max_iterations
        
        # 初始化服务器端共享的概念漂移模拟
        # self.initialize_shared_concepts()
                
        # 如果启用了概念漂移数据集，加载漂移配置
        # if self.use_drift_dataset and self.drift_data_dir:
        #     self.load_drift_config()

        # 初始化 VWC 聚类器
        self.vwc = VariationalWassersteinClustering(
            num_clients=args.num_clients,
            num_clusters=args.num_clusters,
            proxy_dim=32,
            sinkhorn_reg= 0.01
        )

        self.Budget = []  # 用于记录每轮训练的时间成本
        # self.rs_test_acc = []  # 初始化测试准确率列表 # Already in serverbase
        # self.rs_train_loss = [] # 初始化训练损失列表 # Already in serverbase

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
                for client in self.clients: # Update for all clients, or selected_clients if selection happens before this
                    client.update_iteration(self.current_iteration)
            
                # 检查是否是漂移点
                if self.current_iteration in self.drift_iterations:
                    print(f"\n⚠️ 在迭代 {self.current_iteration} 发生概念漂移")
                    # Potentially log drift event to wandb
                    if wandb.run is not None:
                        wandb.log({"Concept Drift Event": 1, "Drift Iteration": self.current_iteration}, step=round_idx)
            
            # 执行正常的训练过程
            # super().train_round(round_idx) # This method does not exist in serverbase. Assuming it's a typo and FedDCA has its own train loop structure.
            # The following is a simplified representation of a training loop structure for FedDCA
            # This will need to be adapted to the actual structure of FedDCA's training logic.

            # Placeholder for the actual training loop structure within FedDCA
            # This is a conceptual adjustment based on common patterns in other server files.
            # The actual implementation details of FedDCA's training loop need to be reviewed
            # to correctly integrate the current_round passing.

            # Example structure (needs to be verified and adapted):
            # self.selected_clients = self.select_clients()
            # self.send_models() 
            
            # if round_idx % self.eval_gap == 0:
            #     print(f"\n-------------Round number: {round_idx}-------------")
            #     print("\nEvaluate global model")
            #     self.evaluate(current_round=round_idx)

            # for client in self.selected_clients:
            # client.train()
            
            # self.receive_models()
            # self.aggregate_parameters() # Or specific aggregation for FedDCA
            
            # self.Budget.append(time.time() - s_t) # s_t needs to be defined at the start of the round
            # print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            # if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
            #     pass # break

            # At the end of all rounds:
            # self.save_results()
            # self.save_global_model(current_round=self.global_rounds) # Pass current_round (final round)

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
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.current_round = i # Keep track of current round

            # 概念漂移处理 (如果启用)
            # if self.use_drift_dataset:
            #     print(f"\nUpdating client iteration to {self.current_iteration} for round {i}")
            #     for client in self.clients: # Update for all clients, or selected_clients if selection happens before this
            #         client.update_iteration(self.current_iteration)
            #     if self.current_iteration in self.drift_iterations:
            #         print(f"\n⚠️ Concept drift occurring at iteration {self.current_iteration} (Round {i})")
            #         # Potentially log drift event to wandb
            #         if wandb.run is not None:
            #             wandb.log({"Concept Drift Event": 1, "Drift Iteration": self.current_iteration}, step=i)

            self.selected_clients = self.select_clients()
            
            # 客户端聚类和模型分发
            if i > 0 or not self.cluster_inited: # Perform clustering from the first round or if not initialized
                print(f"\nPerforming clustering for round {i}...")
                clustering_method = self.select_clustering_algorithm()
                if clustering_method == 'vwc':
                    self.vwc_clustering() 
                elif clustering_method == 'label_conditional':
                    self.perform_label_conditional_clustering(verbose=True)
                elif clustering_method == 'enhanced_label':
                    self.perform_enhanced_label_conditional_clustering(verbose=True)
                else:
                    self.vwc_clustering() # Default to VWC if method is unknown
                self.cluster_inited = True
                self.log_cluster_assignments(i) # Log cluster assignments to wandb

            self.send_cluster_models() # Send appropriate cluster model to each client

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate models (global and per-cluster if applicable)")
                self.evaluate(current_round=i) # Evaluate global model (serverbase version)
                # FedDCA might have its own cluster-specific evaluation, which also needs current_round
                self.evaluate_cluster_performance(current_round=i) 

            # 客户端训练
            for client in self.selected_clients:
                # Apply concept drift at specific round (e.g., round 100)
                if i == 0: # Condition for drift
                    if hasattr(client, 'use_drift_dataset') and client.use_drift_dataset:
                        if hasattr(client, 'apply_drift_transformation'):
                            print(f"Server: Applying drift for client {client.id} at round {i}")
                            # Apply drift to both training and testing datasets on the client
                            client.apply_drift_transformation()
                        else:
                            print(f"Warning: Client {client.id} is configured to use drift but does not have apply_drift_transformation method.")
                    # else:
                        # print(f"Client {client.id} not configured for drift or use_drift_dataset is False at round {i}")

                client.current_iteration = i # Pass current round for client-side logging if needed

                client.train()

            self.receive_models()
            self.update_global_model() # This likely involves cluster-specific aggregation first, then global model update

            # Concept alignment analysis (if applicable)
            if hasattr(self.args, 'analyze_concept_alignment') and self.args.analyze_concept_alignment and (i % self.args.concept_analysis_interval == 0):
                self.analyze_concept_alignment(current_round=i)

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
            
        print("\nBest accuracy.")
        if self.rs_test_acc:
            print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        if len(self.Budget) > 1:
            print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model(current_round=self.global_rounds) # Save final model
        if hasattr(self, 'save_concept_progress') and callable(self.save_concept_progress):
            self.save_concept_progress(current_round=self.global_rounds) # Save final concept data



    def send_cluster_models(self):
        assert (len(self.selected_clients) > 0)
        for client in self.selected_clients:
            cluster_id = self.clusters.get(client.id, 0) # Default to cluster 0 if not found
            model_to_send = self.cluster_centroids.get(cluster_id, self.global_model) # Send cluster model or global if not found
            client.set_parameters(model_to_send)

    def evaluate_cluster_performance(self, current_round):
        # This is a placeholder for FedDCA's specific cluster evaluation logic
        # It should log metrics per cluster to wandb if possible
        print(f"Evaluating cluster performance for round {current_round}...")
        for cluster_id, model in self.cluster_centroids.items():
            # Evaluate this cluster_model, similar to how global_model is evaluated in serverbase
            # but on clients belonging to this cluster_id
            cluster_clients = [c for c in self.clients if self.clusters.get(c.id) == cluster_id]
            if not cluster_clients:
                continue

            num_samples = []
            tot_correct = []
            # Store original models to restore later
            original_models = {c.id: copy.deepcopy(c.model) for c in cluster_clients}

            for c in cluster_clients:
                c.set_parameters(model) # Set cluster model for evaluation
                ct, ns, auc = c.test_metrics() # Assuming test_metrics is available and returns (correct_preds, num_samples, auc_val)
                tot_correct.append(ct*1.0)
                num_samples.append(ns)
            
            # Restore original models
            for c in cluster_clients:
                c.set_parameters(original_models[c.id])

            if sum(num_samples) > 0:
                cluster_acc = sum(tot_correct) / sum(num_samples)
                print(f"  Cluster {cluster_id} Test Accuracy: {cluster_acc:.4f}")
                if wandb.run is not None:
                    wandb.log({f"Cluster {cluster_id}/Test Accuracy": cluster_acc}, step=current_round)
            else:
                print(f"  Cluster {cluster_id}: No samples for evaluation.")

    def log_cluster_assignments(self, current_round):
        if wandb.run is not None and self.clusters:
            # Log number of clients per cluster
            cluster_counts = {f"Cluster {cid}/Client Count": 0 for cid in self.cluster_centroids.keys()}
            for client_id, cluster_id in self.clusters.items():
                if f"Cluster {cluster_id}/Client Count" in cluster_counts:
                    cluster_counts[f"Cluster {cluster_id}/Client Count"] += 1
                else: # Should not happen if cluster_centroids keys are source of truth
                    cluster_counts[f"Cluster {cluster_id}/Client Count"] = 1
            wandb.log(cluster_counts, step=current_round)

            # Log individual client assignments if not too many clients
            if len(self.clients) <= 50: # Arbitrary limit to prevent too much data
                client_assignments_log = {}
                for client in self.clients:
                    client_assignments_log[f"Client {client.id}/Cluster Assignment"] = self.clusters.get(client.id, -1) # -1 if not assigned
                wandb.log(client_assignments_log, step=current_round)

    def analyze_concept_alignment(self, current_round):
        # Placeholder for concept alignment analysis and logging to wandb
        if hasattr(super(), 'analyze_concept_alignment') and callable(super().analyze_concept_alignment):
            # This assumes analyze_concept_alignment is defined in a base or utility class
            # and can log its findings to wandb internally or return them for logging here.
            alignment_metrics = super().analyze_concept_alignment(self.clients, self.cluster_centroids, self.clusters, current_round)
            if wandb.run is not None and alignment_metrics:
                wandb.log(alignment_metrics, step=current_round)
        else:
            print("Concept alignment analysis method not found or not callable.")

    # Ensure other methods like save_concept_progress also accept current_round if they save round-specific artifacts
    def save_concept_progress(self, current_round):
        # Modified to save with round number and potentially log to wandb
        if not hasattr(self.args, 'save_concept_path') or not self.args.save_concept_path:
            return
        
        concept_data = {
            "round": current_round,
            "client_concepts": {c.id: c.active_concept_id for c in self.clients if hasattr(c, 'active_concept_id')},
            "cluster_centroids_info": {cid: "model_summary_placeholder" for cid in self.cluster_centroids.keys()},
            "client_cluster_assignments": self.clusters
        }
        
        filepath = os.path.join(self.args.save_concept_path, f"concept_progress_round_{current_round}.json")
        try:
            with open(filepath, 'w') as f:
                json.dump(concept_data, f, indent=4)
            print(f"Concept progress saved to {filepath}")

            if self.args.wandb_save_artifacts and wandb.run is not None:
                artifact_name = f'{self.args.wandb_run_name_prefix}_concept_progress'
                artifact = wandb.Artifact(artifact_name, type='concept-data')
                artifact.add_file(filepath, name=f"concept_progress_round_{current_round}.json")
                wandb.log_artifact(artifact, aliases=['latest_concept_data', f'concept_round_{current_round}'])
                print(f"Concept progress for round {current_round} saved to wandb.")
        except Exception as e:
            print(f"Error saving concept progress: {e}")
