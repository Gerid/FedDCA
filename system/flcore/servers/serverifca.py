import copy
import torch
import time
import numpy as np
import os
import json
from flcore.servers.serverbase import Server
from flcore.clients.clientifca import clientIFCA
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from threading import Thread
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class FedIFCA(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        
        # 设置客户端
        self.set_slow_clients()
        self.set_clients(clientIFCA)
        
        # 集群相关参数
        self.cluster_num = args.cluster_num if hasattr(args, 'cluster_num') else 3
        self.global_models = [copy.deepcopy(args.model) for _ in range(self.cluster_num)]
        
        # IFCA特有参数
        self.client_cluster_identity = {}  # 记录客户端分配的集群身份
        self.cluster_clients = {}  # 记录每个集群分配的客户端
        
        # 追踪每轮集群分配和性能
        self.cluster_history = {}  # 记录每个客户端的集群历史
        self.cluster_performance = {}  # 记录每个集群的性能
        self.Budget = []
        
        print(f"\nFedIFCA设置完成! 集群数量: {self.cluster_num}")
        print(f"参与率 / 总客户端数: {self.join_ratio} / {self.num_clients}")

    def train(self):
        """训练过程的主控制流"""
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            
            self.selected_clients = self.select_clients()
            
            if i == 0:  # 第一轮初始化
                print(f"\n初始随机分配客户端到集群...")
                for client in self.selected_clients:
                    # 随机分配初始集群
                    cluster_id = np.random.randint(0, self.cluster_num)
                    client.cluster_identity = cluster_id
                    self.client_cluster_identity[client.id] = cluster_id
                    
                    # 跟踪集群历史
                    if client.id not in self.cluster_history:
                        self.cluster_history[client.id] = []
                    self.cluster_history[client.id].append(cluster_id)
            
            # 根据当前集群分配发送模型
            print("\n向客户端发送模型...")
            self.send_models()
            
            # 客户端计算集群身份
            if i > 0:  # 第一轮后开始集群分配
                print("\n客户端确定集群身份...")
                for client in self.selected_clients:
                    client.clustering(self.global_models)
                    self.client_cluster_identity[client.id] = client.cluster_identity
                    
                    # 跟踪集群历史
                    if client.id not in self.cluster_history:
                        self.cluster_history[client.id] = []
                    self.cluster_history[client.id].append(client.cluster_identity)
            
            # 按照确定的集群身份再次发送模型
            if i > 0:
                self.send_models()
            
            # 评估当前模型
            if i % self.eval_gap == 0:
                print(f"\n-------------轮次 {i}-------------")
                print("\n评估集群模型...")
                self.evaluate_clusters()
            
            # 客户端本地训练
            print("\n客户端本地训练...")
            for client in self.selected_clients:
                client.train()
            
            # 服务器端收集模型
            self.receive_models_with_clustering()
            
            # 按集群聚合模型
            self.aggregate_with_clustering()
            
            # 统计集群分布
            if i % self.eval_gap == 0:
                self.print_cluster_distribution()
                
            # 如果设置，可视化集群
            if i % self.eval_gap == 0 and hasattr(self.args, 'visualize_clusters') and self.args.visualize_clusters:
                self.visualize_clustering(i)
            
            # 记录训练时间
            self.Budget.append(time.time() - s_t)
            print(f"轮次 {i} 时间消耗: {self.Budget[-1]:.2f}s")
            
            # 自动停止条件
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                print("\n达到停止条件，训练完成。")
                break
        
        # 保存结果和可视化
        self.save_results()
        self.save_cluster_history()



    def send_models(self):
        """向客户端发送模型参数"""
        for client in self.selected_clients:
            start_time = time.time()
            
            # 向客户端发送其对应集群的全局模型
            cluster_id = client.cluster_identity
            client.set_parameters(self.global_models[cluster_id].parameters())
            
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += time.time() - start_time

    def receive_models_with_clustering(self):
        """收集客户端更新并按集群组织"""
        assert len(self.selected_clients) > 0
        
        # 清空旧的集群客户端映射
        self.cluster_clients = {}
        for i in range(self.cluster_num):
            self.cluster_clients[i] = []
        
        # 按照客户端的集群身份进行组织
        for client in self.selected_clients:
            cluster_id = client.cluster_identity
            # 添加到对应集群
            if cluster_id not in self.cluster_clients:
                self.cluster_clients[cluster_id] = []
            self.cluster_clients[cluster_id].append(client)

    def aggregate_with_clustering(self):
        """按集群聚合模型参数"""
        # 对每个集群进行聚合
        for cluster_id in range(self.cluster_num):
            clients_in_cluster = self.cluster_clients.get(cluster_id, [])
            
            if not clients_in_cluster:
                # 如果该集群没有客户端，保持不变
                print(f"集群 {cluster_id} 没有分配客户端，保持模型不变")
                continue
            
            # 按照训练数据量加权平均
            total_samples = sum(client.train_samples for client in clients_in_cluster)
            
            # 创建新的模型
            new_model = copy.deepcopy(self.global_models[cluster_id])
            
            # 将参数初始化为0
            for param in new_model.parameters():
                param.data = torch.zeros_like(param.data)
            
            # 加权聚合
            for client in clients_in_cluster:
                client_weight = client.train_samples / total_samples
                
                for server_param, client_param in zip(new_model.parameters(), client.model.parameters()):
                    server_param.data += client_param.data.clone() * client_weight
            
            # 更新集群模型
            self.global_models[cluster_id] = new_model
            print(f"集群 {cluster_id} 更新完成，包含 {len(clients_in_cluster)} 个客户端")

    def evaluate_clusters(self):
        """评估每个集群模型的性能"""
        for cluster_id in range(self.cluster_num):
            model = self.global_models[cluster_id]
            
            clients_in_cluster = [c for c in self.selected_clients if c.cluster_identity == cluster_id]
            
            if not clients_in_cluster:
                print(f"集群 {cluster_id}: 无客户端，跳过评估")
                continue
            
            # 汇总该集群所有客户端的测试性能
            test_acc = 0
            test_samples = 0
            
            for client in clients_in_cluster:
                # 临时保存原始模型
                original_model = copy.deepcopy(client.model)
                
                # 设置为集群模型进行评估
                client.model = copy.deepcopy(model)
                acc, num = client.test_metrics()
                
                test_acc += acc
                test_samples += num
                
                # 恢复原始模型
                client.model = original_model
            
            # 计算并记录平均准确率
            if test_samples > 0:
                avg_acc = test_acc / test_samples
                
                if cluster_id not in self.cluster_performance:
                    self.cluster_performance[cluster_id] = []
                
                self.cluster_performance[cluster_id].append(avg_acc)
                
                print(f"集群 {cluster_id} 测试准确率: {avg_acc:.4f} ({len(clients_in_cluster)} 客户端)")
                
                # 记录到全局评估
                if len(self.rs_test_acc) <= self.global_rounds // self.eval_gap:
                    self.rs_test_acc.append(avg_acc)
                else:
                    # 更新为所有集群的最大值
                    self.rs_test_acc[-1] = max(self.rs_test_acc[-1], avg_acc)

    def print_cluster_distribution(self):
        """打印每个集群的客户端分布情况"""
        print("\n集群分布情况:")
        
        cluster_stats = {}
        for client_id, cluster_id in self.client_cluster_identity.items():
            if cluster_id not in cluster_stats:
                cluster_stats[cluster_id] = {'count': 0, 'clients': []}
            cluster_stats[cluster_id]['count'] += 1
            cluster_stats[cluster_id]['clients'].append(client_id)
        
        for cluster_id, stats in cluster_stats.items():
            client_list = stats['clients']
            if len(client_list) > 10:
                client_list = client_list[:5] + ["..."] + client_list[-5:]
            print(f"集群 {cluster_id}: {stats['count']} 个客户端 {client_list}")

    def visualize_clustering(self, round_idx):
        """可视化当前的集群分配"""
        try:
            # 创建输出目录
            vis_dir = os.path.join('results', 'clustering')
            os.makedirs(vis_dir, exist_ok=True)
            
            # 收集每个客户端的特征表示
            client_features = {}
            for client in self.selected_clients:
                # 提取倒数第二层特征
                if hasattr(client, 'get_features'):
                    features = client.get_features()
                else:
                    # 如果没有特征提取方法，简单使用随机特征（仅用于演示）
                    features = np.random.randn(32)
                
                client_features[client.id] = features
            
            # 使用TSNE降维
            if len(client_features) > 1:
                features = np.array(list(client_features.values()))
                client_ids = list(client_features.keys())
                
                # 降维到2D便于可视化
                tsne = TSNE(n_components=2, random_state=42)
                features_2d = tsne.fit_transform(features)
                
                # 按照集群给数据点着色
                colors = plt.cm.rainbow(np.linspace(0, 1, self.cluster_num))
                
                plt.figure(figsize=(10, 8))
                
                # 绘制每个集群的点
                for cluster_id in range(self.cluster_num):
                    indices = [i for i, cid in enumerate(client_ids) 
                               if self.client_cluster_identity.get(cid, 0) == cluster_id]
                    
                    if indices:
                        cluster_features = features_2d[indices]
                        plt.scatter(
                            cluster_features[:, 0], 
                            cluster_features[:, 1], 
                            color=colors[cluster_id],
                            label=f'Cluster {cluster_id}',
                            alpha=0.7,
                            s=100
                        )
                
                # 添加标签
                for i, cid in enumerate(client_ids):
                    plt.annotate(
                        f'{cid}', 
                        (features_2d[i, 0], features_2d[i, 1]),
                        fontsize=8
                    )
                
                plt.title(f'Round {round_idx} - Client Clustering')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # 保存图像
                plt.savefig(os.path.join(vis_dir, f'clustering_round_{round_idx}.png'))
                plt.close()
                
                print(f"集群可视化已保存到 {vis_dir}/clustering_round_{round_idx}.png")
                
        except Exception as e:
            print(f"集群可视化失败: {str(e)}")

    def save_cluster_history(self):
        """保存集群历史记录，用于分析"""
        try:
            # 创建输出目录
            output_dir = os.path.join('results', 'clustering')
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存集群历史
            history_data = {
                'cluster_history': self.cluster_history,
                'cluster_performance': self.cluster_performance,
                'num_clusters': self.cluster_num
            }
            
            with open(os.path.join(output_dir, 'ifca_cluster_history.json'), 'w') as f:
                json.dump(history_data, f, indent=4)
                
            print(f"集群历史记录已保存到 {output_dir}/ifca_cluster_history.json")
            
        except Exception as e:
            print(f"保存集群历史记录失败: {str(e)}")
