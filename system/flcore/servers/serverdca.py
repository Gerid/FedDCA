import copy
import os
import random
import time
import traceback
import json
from flcore.clients.clientdca import clientDCA
from flcore.servers.serverbase import Server
import torch
import numpy as np
import scipy.stats as stats
try:
    import ot  # 导入POT库 (Python Optimal Transport)
    POT_AVAILABLE = True
except ImportError:
    print("Warning: 未找到POT库，将使用替代方法计算Wasserstein距离")
    POT_AVAILABLE = False
    
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from functools import wraps
import wandb # Added wandb import
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class VariationalWassersteinClustering:
    """简化的变分Wasserstein聚类实现"""
    
    def __init__(self, num_clients, num_clusters, proxy_dim=32, sinkhorn_reg=0.01):
        self.num_clients = num_clients
        self.num_clusters = num_clusters
        self.proxy_dim = proxy_dim
        self.sinkhorn_reg = sinkhorn_reg
    
    def enhanced_clustering(self, label_conditional_data, device, verbose=False):
        """执行增强的聚类算法"""
        try:
            if not label_conditional_data:
                return None
                
            client_ids = list(label_conditional_data.keys())
            num_clients = len(client_ids)
            
            if num_clients < 2:
                return {client_ids[0]: 0} if client_ids else {}
            
            # 计算客户端之间的距离矩阵
            distance_matrix = torch.zeros((num_clients, num_clients))
            
            for i, client_i in enumerate(client_ids):
                for j, client_j in enumerate(client_ids):
                    if i != j:
                        distance = self._compute_client_distance(
                            label_conditional_data[client_i],
                            label_conditional_data[client_j]
                        )
                        distance_matrix[i, j] = distance
            
            # 使用K-means进行聚类
            from sklearn.cluster import KMeans
            
            n_clusters = min(self.num_clusters, num_clients)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            
            # 将距离矩阵转换为特征矩阵
            features = distance_matrix.numpy()
            cluster_labels = kmeans.fit_predict(features)
            
            # 返回聚类结果
            cluster_assignments = {}
            for i, client_id in enumerate(client_ids):
                cluster_assignments[client_id] = int(cluster_labels[i])
            
            if verbose:
                print(f"VWC聚类完成，{len(set(cluster_labels))}个集群")
                
            return cluster_assignments
            
        except Exception as e:
            print(f"VWC聚类失败: {e}")
            return None
    
    def _compute_client_distance(self, client_i_data, client_j_data):
        """计算两个客户端之间的距离"""
        try:
            # 获取共同标签
            common_labels = set(client_i_data.keys()) & set(client_j_data.keys())
            
            if not common_labels:
                return 1.0
            
            total_distance = 0.0
            for label in common_labels:
                features_i = client_i_data[label]
                features_j = client_j_data[label]
                
                if isinstance(features_i, torch.Tensor):
                    features_i = features_i.detach().cpu().numpy()
                if isinstance(features_j, torch.Tensor):
                    features_j = features_j.detach().cpu().numpy()
                
                # 计算简单的欧几里得距离
                if features_i.size > 0 and features_j.size > 0:
                    mean_i = np.mean(features_i.reshape(-1))
                    mean_j = np.mean(features_j.reshape(-1))
                    total_distance += abs(mean_i - mean_j)
            
            return total_distance / len(common_labels) if common_labels else 1.0
            
        except Exception:
            return 0.5

def perform_label_conditional_clustering(clients, num_clusters, device, verbose=False):
    """执行基于标签条件的聚类"""
    try:
        if not clients:
            return {}
            
        # 简化实现：基于客户端ID进行均匀分配
        cluster_assignments = {}
        for i, client in enumerate(clients):
            cluster_id = i % num_clusters
            cluster_assignments[client.id] = cluster_id
        
        if verbose:
            print(f"标签条件聚类完成，{num_clusters}个集群")
            
        return cluster_assignments
        
    except Exception as e:
        print(f"标签条件聚类失败: {e}")
        return {}



class FedDCA(Server):    
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientDCA)
        
        self.cluster_inited = False
        self.args = args
        self.args.load_pretrain = False

        # 初始化所有必要的字典
        self.cluster_models = {}  # 映射集群到模型 (将存储分类器头 nn.Module)
        self.clusters = {}  # 存储客户端到集群的映射
        self.cluster_centroids = {}  # 存储每个集群的中心模型 (可能用途改变或移除)
        self.client_features = {}  # 存储客户端的特征

        
        self.clf_keys = [] # 新增：存储分类器层的参数键名
        
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

        # 初始化 VWC 聚类器
        self.vwc = VariationalWassersteinClustering(
            num_clients=args.num_clients,
            num_clusters=args.num_clusters,
            proxy_dim=32,
            sinkhorn_reg= 0.01
        )

        self.Budget = []  # 用于记录每轮训练的时间成本

    def aggregate_rep_params(self, selected_clients):
        """聚合表示层（body）参数从选定的客户端"""
        if not selected_clients:
            print("aggregate_rep_params: No clients available for aggregation")
            return
            
        try:
            # 收集所有客户端的表示层参数向量
            client_rep_vectors = []
            client_sample_nums = []
            
            for client in selected_clients:
                # 获取客户端表示层参数向量
                rep_vector = client.get_rep_parameters_vector()
                if rep_vector is not None and rep_vector.numel() > 0:
                    client_rep_vectors.append(rep_vector.to(self.device))
                    # 使用客户端的训练样本数量作为权重
                    sample_num = getattr(client, 'train_samples', 1)
                    client_sample_nums.append(sample_num)
                else:
                    print(f"Warning: Client {client.id} has empty representation parameters")
            
            if not client_rep_vectors:
                print("aggregate_rep_params: No valid representation parameters to aggregate")
                return
                
            # 计算聚合权重（基于样本数量）
            total_samples = sum(client_sample_nums)
            weights = torch.tensor([num / total_samples for num in client_sample_nums], 
                                 device=self.device, dtype=torch.float32)
            
            # 执行加权聚合
            stacked_vectors = torch.stack(client_rep_vectors, dim=0)  # [num_clients, param_size]
            aggregated_vector = torch.sum(weights.unsqueeze(1) * stacked_vectors, dim=0)
            
            # 将聚合后的参数向量设置回全局模型的body部分
            if hasattr(self.global_model, 'body') and self.global_model.body is not None:
                body_params = list(self.global_model.body.parameters())
                if body_params:
                    vector_to_parameters(aggregated_vector, body_params)
                    print(f"aggregate_rep_params: Successfully aggregated representation parameters from {len(selected_clients)} clients")
                else:
                    print("Warning: Global model body has no parameters to update")
            else:
                print("Warning: Global model does not have a body attribute")
                
        except Exception as e:
            print(f"Error in aggregate_rep_params: {str(e)}")

    def aggregate_cluster_classifiers(self, selected_clients):
        """聚合每个集群的分类器（head）参数"""
        if not selected_clients:
            print("aggregate_cluster_classifiers: No clients available for aggregation")
            return
            
        try:
            # 按集群组织客户端
            clusters_clients = {}
            for client in selected_clients:
                cluster_id = self.clusters.get(client.id)
                if cluster_id is not None:
                    if cluster_id not in clusters_clients:
                        clusters_clients[cluster_id] = []
                    clusters_clients[cluster_id].append(client)
                else:
                    print(f"Warning: Client {client.id} not assigned to any cluster")
            
            # 为每个集群聚合分类器参数
            for cluster_id, cluster_clients in clusters_clients.items():
                if not cluster_clients:
                    continue
                    
                # 收集该集群中所有客户端的分类器参数
                clf_state_dicts = []
                client_sample_nums = []
                
                for client in cluster_clients:
                    clf_params = client.get_clf_parameters()
                    if clf_params:
                        clf_state_dicts.append(clf_params)
                        sample_num = getattr(client, 'train_samples', 1)
                        client_sample_nums.append(sample_num)
                
                if not clf_state_dicts:
                    print(f"Warning: No valid classifier parameters for cluster {cluster_id}")
                    continue
                
                # 计算聚合权重
                total_samples = sum(client_sample_nums)
                weights = [num / total_samples for num in client_sample_nums]
                
                # 聚合分类器参数
                aggregated_clf_state_dict = {}
                for key in clf_state_dicts[0].keys():
                    # 确保所有客户端都有这个参数
                    if all(key in state_dict for state_dict in clf_state_dicts):
                        # 执行加权平均
                        weighted_sum = sum(w * state_dict[key].to(self.device) 
                                         for w, state_dict in zip(weights, clf_state_dicts))
                        aggregated_clf_state_dict[key] = weighted_sum
                    else:
                        print(f"Warning: Parameter {key} not found in all clients for cluster {cluster_id}")
                
                # 创建或更新集群分类器模型
                if cluster_id not in self.cluster_models:
                    # 创建新的分类器模型
                    if hasattr(self.global_model, 'head') and self.global_model.head is not None:
                        self.cluster_models[cluster_id] = copy.deepcopy(self.global_model.head)
                    else:
                        print(f"Warning: Cannot create cluster model for cluster {cluster_id} - no head template")
                        continue
                
                # 加载聚合后的参数
                self.cluster_models[cluster_id].load_state_dict(aggregated_clf_state_dict)
                print(f"aggregate_cluster_classifiers: Updated cluster {cluster_id} with {len(cluster_clients)} clients")
                
        except Exception as e:
            print(f"Error in aggregate_cluster_classifiers: {str(e)}")

    def vwc_clustering(self):
        """初始化聚类"""
        try:
            # 收集所有客户端的代理数据
            proxy_points = {}
            for client in self.selected_clients:
                if hasattr(client, 'intermediate_output') and callable(client.intermediate_output):
                    try:
                        proxy_points[client.id] = client.intermediate_output()
                    except:
                        print(f"Warning: Failed to get intermediate output from client {client.id}")

            if not proxy_points:
                print("Warning: No proxy data available for clustering")
                # 使用均匀分配
                for i, client in enumerate(self.selected_clients):
                    self.clusters[client.id] = i % self.args.num_clusters
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
                if isinstance(client_data, torch.Tensor):
                    distances = {}
                    for cluster_id, center in cluster_centers.items():
                        if isinstance(center, torch.Tensor):
                            distances[cluster_id] = torch.norm(client_data - center).item()
                        else:
                            distances[cluster_id] = 1.0
                    closest_cluster = min(distances, key=distances.get)
                    self.clusters[client_id] = closest_cluster
                else:
                    # 如果数据格式不对，随机分配
                    self.clusters[client_id] = random.randint(0, self.args.num_clusters - 1)

        except Exception as e:
            print(f"Error in VWC clustering: {str(e)}")
            # 发生错误时，确保每个客户端至少被分配到默认集群
            for client in self.selected_clients:
                if client.id not in self.clusters:
                    self.clusters[client.id] = 0
            if 0 not in self.cluster_centroids:
                self.cluster_centroids[0] = copy.deepcopy(self.global_model)

    def select_clustering_algorithm(self):
        """根据配置选择使用的聚类算法"""
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
        """执行基于标签条件分布的Wasserstein聚类"""
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
            return False

    def perform_enhanced_label_conditional_clustering(self, verbose=False):
        """执行增强的基于标签条件分布的聚类算法"""
        print("使用增强的标签条件聚类算法...")
        try:
            # 收集标签条件代理数据
            label_conditional_data = self.collect_label_conditional_proxy_data()
            
            if not label_conditional_data:
                print("Warning: 没有足够的标签条件代理数据进行聚类，退回到VWC聚类")
                self.vwc_clustering()
                return True
            
            # 使用改进的VWC聚类器
            try:
                cluster_assignments = self.vwc.enhanced_clustering(
                    label_conditional_data=label_conditional_data,
                    device=self.device,
                    verbose=verbose
                )
                
                if cluster_assignments:
                    # 更新集群分配
                    self.clusters.update(cluster_assignments)
                    
                    # 记录聚类历史
                    for client_id, cluster_id in cluster_assignments.items():
                        if client_id not in self.client_cluster_history:
                            self.client_cluster_history[client_id] = []
                        self.client_cluster_history[client_id].append(cluster_id)
                    
                    print(f"增强标签条件聚类完成，形成 {len(set(self.clusters.values()))} 个集群")
                    return True
                else:
                    print("警告: 增强标签条件聚类失败，退回到标准聚类")
                    self.vwc_clustering()
                    return True
                    
            except Exception as vwc_error:
                print(f"VWC增强聚类失败: {str(vwc_error)}")
                # 退回到标签条件聚类
                return self.perform_label_conditional_clustering(verbose=verbose)
                
        except Exception as e:
            print(f"增强标签条件聚类失败: {str(e)}")
            
            # 确保每个客户端至少被分配到默认集群
            for client in self.selected_clients:
                if client.id not in self.clusters:
                    self.clusters[client.id] = 0
                    
                # 更新历史记录
                if client.id not in self.client_cluster_history:
                    self.client_cluster_history[client.id] = []
                self.client_cluster_history[client.id].append(0)
            
            return False

    def collect_label_conditional_proxy_data(self):
        """收集每个客户端按标签条件分组的代理数据"""
        label_conditional_proxy_data = {}
        
        for client in self.selected_clients:
            try:
                # 从客户端获取按标签分组的特征
                if hasattr(client, 'get_intermediate_outputs_with_labels'):
                    features_by_label = client.get_intermediate_outputs_with_labels()
                else:
                    # 如果客户端没有这个方法，使用简化版本
                    features_by_label = {}
                
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
                    
                    # 处理一维数据
                    if features.ndim == 1:
                        features = features.reshape(-1, 1)
                        
                    # 如果样本数足够，使用KDE估计分布
                    if features.shape[0] >= 5:  
                        try:
                            # 转置特征以适应stats.gaussian_kde的输入格式
                            kde = stats.gaussian_kde(features.T)
                            
                            # 采样点的数量
                            num_samples = min(getattr(self.args, 'kde_samples', 20), max(20, features.shape[0]))
                            
                            # 重采样生成代理数据
                            sampled = kde.resample(num_samples).T
                            label_conditional_proxy_data[client.id][label] = sampled
                        except Exception:
                            # KDE失败，使用原始特征
                            label_conditional_proxy_data[client.id][label] = features
                    else:
                        # 如果样本太少，直接使用原始数据
                        label_conditional_proxy_data[client.id][label] = features
                        
            except Exception as e:
                print(f"Error collecting data from client {client.id}: {str(e)}")
                continue
        
        return label_conditional_proxy_data

    def evaluate_cluster_performance(self, current_round=None):
        """评估每个集群的性能"""
        try:
            cluster_stats = {}
            
            for client in self.selected_clients:
                cluster_id = self.clusters.get(client.id)
                if cluster_id is not None:
                    if cluster_id not in cluster_stats:
                        cluster_stats[cluster_id] = {
                            'clients': [],
                            'test_acc': [],
                            'train_loss': [],
                            'sample_nums': []
                        }
                    
                    # 获取客户端性能指标
                    try:
                        test_acc, test_num, _ = client.test_metrics()
                        train_loss, train_num = client.train_metrics()
                        
                        cluster_stats[cluster_id]['clients'].append(client.id)
                        if test_num > 0:
                            cluster_stats[cluster_id]['test_acc'].append(test_acc / test_num)
                        if train_num > 0:
                            cluster_stats[cluster_id]['train_loss'].append(train_loss / train_num)
                        cluster_stats[cluster_id]['sample_nums'].append(test_num)
                        
                    except Exception as e:
                        print(f"Error getting metrics for client {client.id}: {str(e)}")
            
            # 计算并显示每个集群的平均性能
            print(f"\n=== Cluster Performance (Round {current_round}) ===")
            total_weighted_acc = 0
            total_samples = 0
            
            for cluster_id, stats in cluster_stats.items():
                if stats['test_acc']:
                    # 计算加权平均准确率
                    weighted_acc = sum(acc * num for acc, num in zip(stats['test_acc'], stats['sample_nums']))
                    cluster_samples = sum(stats['sample_nums'])
                    avg_acc = weighted_acc / cluster_samples if cluster_samples > 0 else 0
                    
                    total_weighted_acc += weighted_acc
                    total_samples += cluster_samples
                    
                    avg_loss = np.mean(stats['train_loss']) if stats['train_loss'] else 0
                    
                    print(f"Cluster {cluster_id}: {len(stats['clients'])} clients, "
                          f"Avg Acc: {100*avg_acc:.2f}%, Avg Loss: {avg_loss:.4f}")
                else:
                    print(f"Cluster {cluster_id}: {len(stats['clients'])} clients, No performance data")
            
            # 计算总体加权平均准确率
            if total_samples > 0:
                overall_avg_acc = total_weighted_acc / total_samples
                print(f"Overall Weighted Avg Accuracy: {100*overall_avg_acc:.2f}%")
                
                # 记录到wandb
                if wandb.run is not None:
                    wandb.log({
                        "cluster_performance/overall_accuracy": overall_avg_acc,
                        "cluster_performance/num_clusters": len(cluster_stats),

                    }, step=self.current_round)
                    
        except Exception as e:
            print(f"Error in evaluate_cluster_performance: {str(e)}")

    def log_cluster_assignments(self, current_round):
        """记录当前轮次的集群分配情况"""
        try:
            cluster_distribution = {}
            for client_id, cluster_id in self.clusters.items():
                if cluster_id not in cluster_distribution:
                    cluster_distribution[cluster_id] = []
                cluster_distribution[cluster_id].append(client_id)
            
            print(f"\n=== Cluster Assignments (Round {current_round}) ===")
            for cluster_id, client_ids in cluster_distribution.items():
                print(f"Cluster {cluster_id}: {len(client_ids)} clients {client_ids}")
            
            # 记录到wandb
            if wandb.run is not None:
                cluster_sizes = {f"cluster_{cid}_size": len(clients) 
                               for cid, clients in cluster_distribution.items()}
                wandb.log({
                    "cluster_assignments/num_clusters": len(cluster_distribution),
                    **cluster_sizes,
                }, step=self.current_round)
                
        except Exception as e:
            print(f"Error in log_cluster_assignments: {str(e)}")

    #@plot_metrics()
    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.current_round = i
            if self.args.verbose:
                print(f"\n FedDCA Round {i} Starting...")

            self.selected_clients = self.select_clients()
            if not self.selected_clients:
                print(f"Round {i}: No clients selected. Skipping round.")
                self.Budget.append(time.time() - s_t)
                continue            # 1. Clients perform local training
            if self.args.verbose:
                print(f"Round {i}: Starting local training for {len(self.selected_clients)} clients.")
            
            # Apply concept drift transformation if needed
            self.apply_drift_transformation()
            
            for client in self.selected_clients:
                client.train()
            if self.args.verbose:
                print(f"Round {i}: Local training completed.")

            # 2. Aggregate Representation Layers (body)
            if self.args.verbose:
                print(f"Round {i}: Aggregating representation layers (body)...")
            self.aggregate_rep_params(self.selected_clients)

            # 3. Send the newly aggregated global body to clients
            if self.args.verbose:
                print(f"Round {i}: Sending updated global body to clients.")
            if hasattr(self.global_model, 'body') and self.global_model.body is not None:
                global_rep_vector = parameters_to_vector([p.clone().detach() for p in self.global_model.body.parameters()])
                if global_rep_vector.numel() > 0:
                    for client in self.selected_clients:
                        client.receive_global_model_body(global_rep_vector.clone())
                else:
                    print(f"Round {i}: Warning - Global representation vector is empty.")
            else:
                print(f"Round {i}: Warning - Global model body not found.")

            # 4. Perform Clustering
            if self.args.verbose:
                print(f"Round {i}: Performing clustering...")
            clustering_method = self.select_clustering_algorithm()
            if clustering_method == 'vwc':
                self.vwc_clustering() 
            elif clustering_method == 'label_conditional':
                self.perform_label_conditional_clustering(verbose=self.args.verbose)
            elif clustering_method == 'enhanced_label':
                self.perform_enhanced_label_conditional_clustering(verbose=self.args.verbose)
            else:
                self.vwc_clustering()  # Default fallback
            
            self.log_cluster_assignments(i)
            if self.args.verbose:
                print(f"Round {i}: Clustering complete. Clusters: {self.clusters}")

            # 5. Aggregate Classifiers (head) for each cluster
            if self.args.verbose:
                print(f"Round {i}: Aggregating classifiers (heads) for each cluster...")
            self.aggregate_cluster_classifiers(self.selected_clients)

            # 6. Send updated models to clients
            if self.args.verbose:
                print(f"Round {i}: Sending cluster-specific heads to clients.")
            self.send_models_to_clients(self.selected_clients)

            # Evaluation and Logging
            if i % self.eval_gap == 0:
                if self.args.verbose:
                    print(f"\n-------------Round {i} Evaluation-------------")
                self.evaluate(current_round=i)
                self.evaluate_cluster_performance(current_round=i) 
            
            self.Budget.append(time.time() - s_t)
            if self.args.verbose:
                print(f"Round {i} completed. Time cost: {self.Budget[-1]:.2f}s")

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                print(f"Auto-break condition met at round {i}.")
                break
        
        print("\n======== FedDCA Training Complete =========")
        if self.rs_test_acc:
            print(f"Best Test Accuracy: {max(self.rs_test_acc):.4f}")
        if len(self.Budget) > 1:
            print(f"Average Time Cost per Round: {np.mean(self.Budget[1:]):.2f}s")

        self.save_results()

    def send_models_to_clients(self, clients_to_send_to):
        """Sends the cluster-specific classifier (head) to clients."""
        if not self.cluster_models:
            if self.args.verbose:
                print("send_models_to_clients: No cluster models (heads) available to send.")
            return

        for client in clients_to_send_to:
            cluster_id_for_client = None
            # 修复：正确查找客户端的集群ID
            cluster_id_for_client = self.clusters.get(client.id)
            
            if cluster_id_for_client is not None and cluster_id_for_client in self.cluster_models:
                cluster_head_module = self.cluster_models[cluster_id_for_client]
                if cluster_head_module is not None:
                    client.receive_cluster_model(copy.deepcopy(cluster_head_module.state_dict()))
                    if self.args.verbose:
                        print(f"Client {client.id} (Cluster {cluster_id_for_client}): Received cluster head.")
                else:
                    if self.args.verbose:
                        print(f"Client {client.id} (Cluster {cluster_id_for_client}): Cluster head model is None.")
            elif self.args.verbose:
                print(f"Client {client.id}: Not found in cluster assignments or cluster model missing.")

    def set_clf_keys(self, clf_keys):
        """Sets the classifier keys for the server."""
        self.clf_keys = clf_keys
        print(f"Server clf_keys set to: {self.clf_keys}")
