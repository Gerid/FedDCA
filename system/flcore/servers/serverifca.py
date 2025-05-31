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
import wandb


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
            self.current_round = i 
            s_t = time.time()
            
            self.selected_clients = self.select_clients()
            
            print("\\n客户端计算集群身份...") # Applies to all rounds now
            for client in self.selected_clients:
                if i == 100:  # Condition for drift
                    if hasattr(client, 'use_drift_dataset') and client.use_drift_dataset:
                        if hasattr(client, 'apply_drift_transformation'):
                            print(f"Server: Applying drift for client {client.id} at round {i}")
                            # Apply drift to both training and testing datasets on the client
                            client.apply_drift_transformation()
                        else:
                            print(f"Warning: Client {client.id} is configured to use drift but does not have apply_drift_transformation method.")

                try:
                    # Store old cluster ID for logging changes, if client was seen before
                    old_cluster_id = self.client_cluster_identity.get(client.id, -1) # Default to -1 if new client

                    client.clustering(self.global_models) 
                    new_cluster_id = client.cluster_identity 
                    
                    if old_cluster_id != -1 and new_cluster_id != old_cluster_id: # Log if known client changed cluster
                        print(f"客户端 {client.id} 从集群 {old_cluster_id} 迁移到集群 {new_cluster_id}")
                    elif old_cluster_id == -1: # New client or first time clustering for this client
                         print(f"客户端 {client.id} 被分配到集群 {new_cluster_id}")
                    # If old_cluster_id != -1 and new_cluster_id == old_cluster_id, no message is printed, which is fine.

                    self.client_cluster_identity[client.id] = new_cluster_id
                    
                    # 跟踪集群历史
                    if client.id not in self.cluster_history:
                        self.cluster_history[client.id] = []
                    self.cluster_history[client.id].append(new_cluster_id)

                except Exception as e:
                    print(f"客户端 {client.id} 执行聚类失败: {str(e)}")
                    # Fallback: assign to a default cluster (e.g., 0) or keep previous if known
                    fallback_cluster_id = self.client_cluster_identity.get(client.id, 0) # Default to 0 if never seen or error during first clustering
                    client.cluster_identity = fallback_cluster_id
                    self.client_cluster_identity[client.id] = fallback_cluster_id
                    print(f"客户端 {client.id} 由于聚类失败，被分配到集群 {fallback_cluster_id} (回退逻辑)")

                    # 跟踪集群历史 (fallback)
                    if client.id not in self.cluster_history:
                        self.cluster_history[client.id] = []
                    self.cluster_history[client.id].append(fallback_cluster_id)
            
            # 根据当前集群分配发送模型
            print("\n向客户端发送模型...")
            self.send_models()
            
            # 评估当前模型
            if self.current_round % self.eval_gap == 0:
                print(f"\n-------------轮次 {self.current_round}-------------")
                print("\n评估集群模型...")
                self.evaluate(current_round=self.current_round) # Pass current_round
                # self.evaluate_clusters(current_round=self.current_round) # Pass current_round to cluster specific evaluation
            
            # 客户端本地训练
            print("\n客户端本地训练...")
            for client in self.selected_clients:
                client.train()
            
            # 服务器端收集模型
            self.receive_models_with_clustering()
            
            # 按集群聚合模型
            self.aggregate_with_clustering() # This method will now use self.current_round for logging
            
            # 统计集群分布
            if self.current_round % self.eval_gap == 0:
                self.print_cluster_distribution(current_round=self.current_round)
                
            # 如果设置，可视化集群
            if self.current_round % self.eval_gap == 0 and hasattr(self.args, 'visualize_clusters') and self.args.visualize_clusters:
                self.visualize_clustering(self.current_round)
            
            # 记录训练时间
            self.Budget.append(time.time() - s_t)
            print(f"轮次 {self.current_round} 时间消耗: {self.Budget[-1]:.2f}s")
            
            # 自动停止条件
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                print("\n达到停止条件，训练完成。")
                break
        
        # 保存结果和可视化
        self.save_results()
        self.save_cluster_history(current_round=self.current_round) # Pass final round
        # Save all global models (cluster models) at the end of training
        for cluster_id in range(self.cluster_num):
            self.save_global_model(current_round=self.current_round, cluster_id=cluster_id) # Pass current_round and cluster_id


    def send_models(self):
        """向客户端发送模型参数"""
        for client in self.selected_clients:
            start_time = time.time()
            
            # 向客户端发送其对应集群的全局模型
            cluster_id = client.cluster_identity
            client.set_parameters(self.global_models[cluster_id])
            
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
                print(f"集群 {cluster_id} 没有分配客户端，保持模型不变")
                continue
            
            total_samples = sum(client.train_samples for client in clients_in_cluster)
            
            if total_samples == 0:
                print(f"警告: 集群 {cluster_id} 的客户端总样本数为0，跳过聚合")
                continue
                
            new_model = copy.deepcopy(self.global_models[cluster_id])
            old_params = parameters_to_vector([param for param in new_model.parameters()])
            
            # 将参数和缓冲区初始化为0
            for param in new_model.parameters():
                param.data.zero_()
            for buffer in new_model.buffers(): # Added buffer zeroing
                buffer.data.zero_()
            
            aggregated_clients = 0
            skipped_clients = 0

            # 主聚合过程的 try-except 块
            try: 
                for client in clients_in_cluster:
                    client_weight = client.train_samples / total_samples
                    
                    # 单个客户端聚合的 try-except 块
                    try: 
                        # 聚合参数到 new_model
                        for server_param, client_param in zip(new_model.parameters(), client.model.parameters()):
                            if server_param.data.shape != client_param.data.shape:
                                print(f"警告: 集群 {cluster_id} 客户端 {client.id} 参数形状不匹配 ({server_param.data.shape} vs {client_param.data.shape})")
                                raise ValueError("参数形状不匹配")
                            if torch.isnan(client_param.data).any() or torch.isinf(client_param.data).any():
                                print(f"警告: 集群 {cluster_id} 客户端 {client.id} 参数包含NaN或Inf值")
                                raise ValueError("参数包含无效值")
                            server_param.data.add_(client_param.data.clone() * client_weight)

                        # 聚合缓冲区到 new_model
                        for server_buffer, client_buffer in zip(new_model.buffers(), client.model.buffers()):
                            if server_buffer.data.shape != client_buffer.data.shape:
                                print(f"警告: 集群 {cluster_id} 客户端 {client.id} 缓冲区形状不匹配 ({server_buffer.data.shape} vs {client_buffer.data.shape})")
                                raise ValueError("缓冲区形状不匹配")
                            if torch.isnan(client_buffer.data).any() or torch.isinf(client_buffer.data).any():
                                print(f"警告: 集群 {cluster_id} 客户端 {client.id} 缓冲区包含NaN或Inf值")
                                raise ValueError("缓冲区包含无效值")
                            server_buffer.data.add_(client_buffer.data.clone() * client_weight)
                            
                        aggregated_clients += 1
                    except ValueError as ve:
                        print(f"聚合客户端 {client.id} (参数/缓冲区) 失败: {str(ve)}")
                        skipped_clients += 1
                        continue # 跳过此客户端，继续处理其他客户端
                    except Exception as e:
                        print(f"聚合客户端 {client.id} 时发生意外错误: {str(e)}")
                        skipped_clients += 1
                        continue # 跳过此客户端
                
                # 处理完集群中的所有客户端后:
                new_params_vector = parameters_to_vector([param for param in new_model.parameters()])
                param_diff = torch.norm(new_params_vector - old_params)

                if param_diff < 1e-9 and aggregated_clients > 0 : # 仅当有客户端成功聚合时才警告
                    print(f"警告: 集群 {cluster_id} 参数几乎没有变化 (差异: {param_diff:.2e})。聚合客户端数: {aggregated_clients}")
                
                if torch.isnan(new_params_vector).any() or torch.isinf(new_params_vector).any():
                    print(f"错误: 集群 {cluster_id} 聚合后参数包含NaN或Inf值。将保留旧模型。")
                    if wandb.run is not None and hasattr(self, 'current_round') and self.current_round is not None:
                        wandb.log({
                            f"Cluster {cluster_id}/Aggregated Clients": aggregated_clients, # 记录尝试聚合的客户端
                            f"Cluster {cluster_id}/Skipped Clients": skipped_clients + (len(clients_in_cluster) - aggregated_clients - skipped_clients), # 所有未成功聚合的
                            f"Cluster {cluster_id}/Param Diff": param_diff.item(),
                            f"Cluster {cluster_id}/Status": "AggregationFailed_NaN_Inf"
                        }, step=self.current_round)
                    continue # 跳过更新此集群的模型，进入下一个集群的聚合

                # 如果所有检查通过，则更新此集群的全局模型
                self.global_models[cluster_id] = new_model
                print(f"集群 {cluster_id} 更新完成，成功聚合 {aggregated_clients} 个客户端，跳过 {skipped_clients} 个客户端")
                
                if wandb.run is not None and hasattr(self, 'current_round') and self.current_round is not None: 
                    wandb.log({
                        f"Cluster {cluster_id}/Aggregated Clients": aggregated_clients,
                        f"Cluster {cluster_id}/Skipped Clients": skipped_clients,
                        f"Cluster {cluster_id}/Param Diff": param_diff.item(),
                        f"Cluster {cluster_id}/Status": "AggregationSuccess"
                    }, step=self.current_round)

            except Exception as e: # 捕获整个集群聚合过程中的错误
                print(f"集群 {cluster_id} 整体聚合失败: {str(e)}。将保留旧模型。")
                if wandb.run is not None and hasattr(self, 'current_round') and self.current_round is not None:
                    wandb.log({
                        f"Cluster {cluster_id}/Status": "AggregationFailed_OuterException",
                        f"Cluster {cluster_id}/Aggregated Clients": aggregated_clients, # 记录到目前为止聚合的客户端
                        f"Cluster {cluster_id}/Skipped Clients": skipped_clients + (len(clients_in_cluster) - aggregated_clients - skipped_clients)
                    }, step=self.current_round)
                # 此处的 continue 会跳到下一个 cluster_id 的循环
            
    def evaluate_clusters(self, current_round=None): # Add current_round parameter
        """评估每个集群模型的性能"""
        all_cluster_accs = []
        all_cluster_sizes = []
        
        for cluster_id in range(self.cluster_num):
            model = self.global_models[cluster_id]
            
            clients_in_cluster = [c for c in self.selected_clients if c.cluster_identity == cluster_id]
            
            if not clients_in_cluster:
                print(f"集群 {cluster_id}: 无客户端，跳过评估")
                continue
            
            # 汇总该集群所有客户端的测试性能
            test_acc = 0
            test_samples = 0
            client_accs = []
            
            for client in clients_in_cluster:
                # 临时保存原始模型
                original_model = copy.deepcopy(client.model)
                
                # 设置为集群模型进行评估
                client.model = copy.deepcopy(model)
                
                try:
                    # 尝试获取测试指标
                    acc, num, _ = client.test_metrics() if hasattr(client, 'test_metrics') and callable(getattr(client, 'test_metrics')) else (0, 0, 0)
                    
                    if num > 0:
                        client_acc = acc / num
                        client_accs.append(client_acc)
                        test_acc += acc
                        test_samples += num
                        print(f"  客户端 {client.id} 测试准确率: {client_acc:.4f} ({num} 样本)")
                except Exception as e:
                    print(f"  客户端 {client.id} 评估失败: {str(e)}")
                
                # 恢复原始模型
                client.model = original_model
            # 计算并记录平均准确率
            if test_samples > 0:
                avg_acc = test_acc / test_samples
                
                # 计算并显示准确率的标准差，以检查集群内一致性
                if client_accs:
                    acc_std = np.std(client_accs)
                    acc_min = np.min(client_accs)
                    acc_max = np.max(client_accs)
                    print(f"集群 {cluster_id} 测试准确率: {avg_acc:.4f} ± {acc_std:.4f} [最小: {acc_min:.4f}, 最大: {acc_max:.4f}] ({len(clients_in_cluster)} 客户端)")
                else:
                    print(f"集群 {cluster_id} 测试准确率: {avg_acc:.4f} ({len(clients_in_cluster)} 客户端)")
                
                if wandb.run is not None and current_round is not None:
                    wandb.log({
                        f"Cluster {cluster_id}/Test Accuracy": avg_acc,
                        f"Cluster {cluster_id}/Client Count": len(clients_in_cluster),
                        f"Cluster {cluster_id}/Accuracy Std Dev": acc_std if client_accs else 0
                    }, step=current_round)

                if cluster_id not in self.cluster_performance:
                    self.cluster_performance[cluster_id] = []
                
                self.cluster_performance[cluster_id].append(avg_acc)
                
                # 收集所有集群的准确率和大小
                all_cluster_accs.append(avg_acc)
                all_cluster_sizes.append(len(clients_in_cluster))
                
        # 计算全局平均准确率，优先考虑较大的集群
        if all_cluster_accs:
            # 方法1: 选择性能最好的集群
            best_acc = max(all_cluster_accs)
            best_cluster_idx = all_cluster_accs.index(best_acc)
            
            # 方法2: 加权平均所有集群的准确率
            total_clients = sum(all_cluster_sizes)
            weighted_avg_acc = sum(acc * size / total_clients for acc, size in zip(all_cluster_accs, all_cluster_sizes))
            
            print(f"\n全局评估: 最佳集群 {best_cluster_idx} 准确率: {best_acc:.4f}, 加权平均准确率: {weighted_avg_acc:.4f}")
            
            # 记录全局评估结果
            if len(self.rs_test_acc) <= self.global_rounds // self.eval_gap:
                self.rs_test_acc.append(best_acc)  # 使用最佳集群的准确率
                if wandb.run is not None and current_round is not None:
                    wandb.log({"Overall Best Cluster Accuracy": best_acc, "Overall Weighted Avg Accuracy": weighted_avg_acc}, step=current_round)
            else:
                # 更新为所有集群的最大值
                self.rs_test_acc[-1] = max(self.rs_test_acc[-1], best_acc)
                # Log updated best accuracy if it changed
                if wandb.run is not None and current_round is not None:
                     wandb.log({"Overall Best Cluster Accuracy": self.rs_test_acc[-1], "Overall Weighted Avg Accuracy": weighted_avg_acc}, step=current_round)

    def print_cluster_distribution(self, current_round=None): # Add current_round
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
            if wandb.run is not None and current_round is not None:
                 wandb.log({f"Cluster {cluster_id}/Client Count": stats['count']}, step=current_round)

    def visualize_clustering(self, round_idx):
        """可视化当前的集群分配"""
        try:
            # 创建输出目录
            vis_dir = os.path.join('results', 'ifca_clustering')
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
                if self.args.wandb_save_artifacts and wandb.run is not None:
                    try:
                        vis_artifact = wandb.Artifact(
                            f'{self.args.wandb_run_name_prefix}_cluster_visualization',
                            type='visualization',
                            description=f'IFCA client clustering visualization at round {round_idx}'
                        )
                        vis_artifact.add_file(os.path.join(vis_dir, f'clustering_round_{round_idx}.png'), name=f'clustering_round_{round_idx}.png')
                        wandb.log_artifact(vis_artifact, aliases=[f'vis_round_{round_idx}', 'latest_vis'])
                        print(f"Cluster visualization for round {round_idx} saved to wandb.")
                    except Exception as e:
                        print(f"Error saving cluster visualization to wandb: {e}")
                
        except Exception as e:
            print(f"集群可视化失败: {str(e)}")

    def save_cluster_history(self, current_round=None): # Add current_round for consistency, though might save only at end
        """保存集群历史记录，用于分析"""
        try:
            # 创建输出目录
            output_dir = os.path.join('results', 'clustering')
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存集群历史
            history_data = {
                'cluster_history': self.cluster_history,
                'cluster_performance': self.cluster_performance,
                'num_clusters': self.cluster_num,
                'final_round': current_round if current_round is not None else self.global_rounds # record when it was saved
            }
            
            filepath = os.path.join(output_dir, 'ifca_cluster_history.json')
            with open(filepath, 'w') as f:
                json.dump(history_data, f, indent=4)
                
            print(f"集群历史记录已保存到 {filepath}")
            if self.args.wandb_save_artifacts and wandb.run is not None:
                try:
                    hist_artifact = wandb.Artifact(
                        f'{self.args.wandb_run_name_prefix}_cluster_history',
                        type='cluster-history'
                    )
                    hist_artifact.add_file(filepath, name='ifca_cluster_history.json')
                    wandb.log_artifact(hist_artifact, aliases=['latest_cluster_history'])
                    print(f"Cluster history saved to wandb.")
                except Exception as e:
                    print(f"Error saving cluster history to wandb: {e}")
            
        except Exception as e:
            print(f"保存集群历史记录失败: {str(e)}")

    # Modify save_global_model to handle cluster-specific models for IFCA
    def save_global_model(self, current_round=None, cluster_id=None):
        if cluster_id is None:
            # This case should ideally not be hit if IFCA always saves cluster models.
            # Fallback to serverbase logic if needed, or raise error.
            print("Warning: save_global_model called without cluster_id in FedIFCA. Saving all cluster models.")
            for cid in range(self.cluster_num):
                self._save_single_cluster_model(current_round, cid)
            return

        self._save_single_cluster_model(current_round, cluster_id)

    def _save_single_cluster_model(self, current_round, cluster_id):
        model_to_save = self.global_models[cluster_id]
        model_path = os.path.join("models", self.dataset, self.algorithm)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        model_filename = f"server_cluster_{cluster_id}.pt"
        model_filepath = os.path.join(model_path, model_filename)
        torch.save(model_to_save, model_filepath)
        print(f"Cluster {cluster_id} model saved to {model_filepath}")

        if self.args.wandb_save_model and wandb.run is not None and current_round is not None:
            try:
                artifact_name = f'{self.args.wandb_run_name_prefix}_cluster_{cluster_id}_model'
                model_artifact = wandb.Artifact(
                    artifact_name,
                    type='model',
                    description=f'Cluster {cluster_id} model for {self.algorithm} at round {current_round}',
                    metadata={'dataset': self.dataset, 'algorithm': self.algorithm, 'round': current_round, 'cluster_id': cluster_id}
                )
                model_artifact.add_file(model_filepath, name=f'cluster_{cluster_id}_model_round_{current_round}.pt')
                aliases = [f'cluster_{cluster_id}_latest', f'cluster_{cluster_id}_round_{current_round}']
                if current_round == self.global_rounds:
                    aliases.append(f'cluster_{cluster_id}_final')
                wandb.log_artifact(model_artifact, aliases=aliases)
                print(f"Cluster {cluster_id} model saved to wandb as artifact at round {current_round}")
            except Exception as e:
                print(f"Error saving cluster {cluster_id} model to wandb: {e}")
