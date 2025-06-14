import copy
import torch
import time
import numpy as np
import os
import wandb # Added wandb import
from flcore.servers.serverbase import Server
from flcore.clients.clientfeddrift import clientFedDrift
from threading import Thread
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class FedDrift(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        
        # 设置客户端
        self.set_slow_clients()
        self.set_clients(clientFedDrift)
        
        # 集群相关参数
        self.detection_threshold = args.detection_threshold if hasattr(args, 'detection_threshold') else 0.1
        
        # 初始化多个全局模型，开始时只有一个
        self.global_models = [copy.deepcopy(self.global_model)]
        
        # 性能追踪
        self.Budget = []
        self.client_clusters = {}  # 记录每个客户端所属的集群
        self.cluster_counts = []   # 记录每轮的集群数量
        
        print(f"\nFedDrift设置完成!")
        print(f"概念漂移检测阈值: {self.detection_threshold}")
        print(f"参与率 / 总客户端数: {self.join_ratio} / {self.num_clients}")

    def train(self):
        """训练过程的主控制流"""
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.current_round = i # Store current round
              # 每轮选择客户端
            self.selected_clients = self.select_clients()
            
            # Apply concept drift transformation if needed
            self.apply_drift_transformation()
            
            # 每个客户端进行集群分配
            for client in self.selected_clients:

                client.clustering(self.global_models)
            
            # 如果有多于一个集群，尝试合并相似的集群
            if len(self.global_models) > 1:
                self.merge_clusters(self.selected_clients)
            
            # 处理检测到概念漂移的客户端
            for client in self.selected_clients:
                if client.cluster_identity is None:
                    # 为每个漂移的客户端，初始化一个新模型添加到全局模型集合
                    self.global_models.append(copy.deepcopy(self.global_model))
                    client.cluster_identity = len(self.global_models) - 1
                    print(f"为客户端 {client.id} 创建新集群，当前集群数量: {len(self.global_models)}")
            
            # 记录当前轮次的集群数量
            cluster_identities = [client.cluster_identity for client in self.selected_clients if client.cluster_identity is not None]
            unique_clusters = len(set(cluster_identities)) if cluster_identities else 0
            self.cluster_counts.append(unique_clusters)
            print(f"当前轮次 {i} 的集群数量: {unique_clusters}")
            if  self.args.use_wandb:
                wandb.log({"FedDrift/Unique Clusters": unique_clusters, "round": i}, step=i)
            
            # 发送参数给客户端
            self.send_models(self.selected_clients)

            if self.current_round % self.args.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global models")
                self.evaluate(is_global=True)  # ServerBase evaluate method

            # 客户端本地训练
            for client in self.selected_clients:
                client.train()
            
            # 基于集群聚合模型
            self.aggregate_with_clustering(self.selected_clients)
            
            # 再次发送更新后的参数给客户端
            self.send_models(self.selected_clients)
            
            # 更新客户端的前一轮训练样本
            for client in self.selected_clients:
                client.update_prev_train_samples()
            
            # 定期评估
            if i % self.eval_gap == 0:  # Avoid evaluation at round 0 if not meaningful
                print("\nEvaluate personalized models")
                self.evaluate(is_global=False)
            
            # 计算耗时
            e_t = time.time()
            self.Budget.append(e_t - s_t)
            
            # 是否达到预设的准确率要求提前结束
            if self.auto_break and len(self.rs_test_acc) > 0 and self.rs_test_acc[-1] > self.args.goal_accuracy:
                print(f"达到目标准确率 {self.args.goal_accuracy}，提前停止训练。")
                break
        
        # 最终评估和输出
        print("\n训练完成!")
        # self.evaluate(current_round=self.global_rounds) # Pass final round
        print(f"集群演变: {self.cluster_counts}")
        
        # 输出耗时统计
        avg_time = sum(self.Budget) / len(self.Budget) if len(self.Budget) > 0 else 0
        print(f"平均每轮耗时: {avg_time:.2f}秒")
        # if self.args.wandb:
        #     wandb.log({"FedDrift/Average Round Time": avg_time})
            
        # 保存结果和模型
        self.save_results() # Uses serverbase save_results, which handles wandb artifact for h5
        # self.save_models(current_round=self.global_rounds) # Pass final round
        
        # 绘制集群数量变化图
        self.plot_cluster_evolution()

    def send_models(self, clients):
        """
        向客户端发送模型参数，根据其集群身份
        
        Args:
            clients: 客户端列表
        """
        for client in clients:
            if client.cluster_identity is None: # Should not happen if drift is handled properly
                print(f"警告: 客户端 {client.id} 没有集群身份，跳过发送模型。")
                continue
            if client.cluster_identity >= len(self.global_models):
                print(f"警告: 客户端 {client.id} 的集群身份 {client.cluster_identity} 超出范围 {len(self.global_models)}。可能存在问题。")
                # Fallback or error handling: assign to a default cluster or skip
                # For now, skip to prevent crash, but this indicates an issue in clustering logic
                continue
            cluster_id = client.cluster_identity
            client.set_parameters(self.global_models[cluster_id].parameters())

    def aggregate_with_clustering(self, clients):
        """
        基于集群聚合客户端更新
        
        Args:
            clients: 客户端列表
        """
        # 按集群组织客户端
        client_groups = {}
        for identity in range(len(self.global_models)):
            client_groups[identity] = []
        
        for client in clients:
            if client.cluster_identity is None: # Should not happen
                continue
            if client.cluster_identity >= len(self.global_models): # Should not happen
                continue
            client_groups[client.cluster_identity].append(client)
        
        # 多模型聚合
        for identity, global_model in enumerate(self.global_models):
            if len(client_groups[identity]) == 0:
                # 如果没有客户端更新这个全局模型，跳过
                continue
            
            # 模型加权平均
            total_size = 0
            # Ensure new_params is initialized on the correct device
            # new_params = torch.zeros_like(parameters_to_vector(global_model.parameters()), device=self.device)
            # It seems parameters_to_vector already places it on the same device as model parameters
            
            # Initialize new_params based on the first client's model parameters in the group to ensure correct device
            # This is a common pattern if models might be on different devices, though here global_model should be on self.device
            first_client_params = parameters_to_vector(client_groups[identity][0].model.parameters())
            new_params = torch.zeros_like(first_client_params)


            for client in client_groups[identity]:
                client_size = len(client.train_data) if client.train_data else 0 # Use train_loader.dataset
                if client_size == 0: # Skip clients with no training data for aggregation
                    continue
                total_size += client_size
                client_params = parameters_to_vector(client.model.parameters())
                new_params += client_size * client_params
            
            if total_size > 0:
                new_params /= total_size
                # new_params = new_params.to(self.device) # Ensure it's on the correct device before vector_to_parameters
                vector_to_parameters(new_params, global_model.parameters())


    def merge_clusters(self, clients):
        """
        合并相似的集群。参考 FedDrift.py 的实现。
        
        Args:
            clients: 客户端列表
        """
        global_model_num = len(self.global_models)
        if global_model_num <= 1:
            return

        # 初始化损失矩阵
        # loss_matrix[i][j] 表示集群 j 中的客户端数据在全局模型 i 上的平均损失
        loss_matrix = np.full((global_model_num, global_model_num), -1.0)
        
        # 生成损失矩阵用于计算集群距离
        for i in range(global_model_num):
            for j in range(global_model_num):
                total_data_size_in_cluster_j = 0
                total_loss_model_i_on_cluster_j = 0.0
                
                for client in clients:
                    if client.cluster_identity == j:
                        client_data_size = len(client.train_data) if client.train_data else 0
                        if client_data_size > 0:
                            # 使用 client.get_loss 计算模型 i 在该客户端数据上的损失
                            # self.global_models[i] 是模型，client.train_data 是数据样本列表
                            loss_on_client = client.get_loss(self.global_models[i], client.train_data, self.batch_size)
                            if loss_on_client != float('inf'):
                                total_loss_model_i_on_cluster_j += loss_on_client * client_data_size
                                total_data_size_in_cluster_j += client_data_size
                
                if total_data_size_in_cluster_j > 0:
                    loss_matrix[i][j] = total_loss_model_i_on_cluster_j / total_data_size_in_cluster_j
                else:
                    loss_matrix[i][j] = -1.0 # 标记为无效或无穷大，表示无法计算
        
        # 计算集群间距离
        # cluster_distances[i][j] 表示集群 i 和集群 j 之间的距离
        cluster_distances = np.full((global_model_num, global_model_num), -1.0)
        for i in range(global_model_num):
            for j in range(i + 1, global_model_num): # 仅计算上三角，避免重复和自比较
                if loss_matrix[i][j] == -1.0 or loss_matrix[j][i] == -1.0 or \
                   loss_matrix[i][i] == -1.0 or loss_matrix[j][j] == -1.0:
                    # 如果任何必要的损失值无效，则距离无效
                    dist = -1.0 
                else:
                    # FedDrift.py 中的距离定义: dist = max(loss_matrix[i][j] - loss_matrix[i][i], loss_matrix[j][i] - loss_matrix[j][j], 0)
                    # 这个定义衡量的是一个集群的模型在另一个集群数据上的表现相对于其自身集群数据的表现的恶化程度。
                    dist = max(loss_matrix[i][j] - loss_matrix[i][i], loss_matrix[j][i] - loss_matrix[j][j], 0.0)
                
                cluster_distances[i][j] = dist
                cluster_distances[j][i] = dist  # 对称矩阵
        
        deleted_models_indices = [] # 存储要删除的模型的原始索引

        while True:
            current_num_models = len(self.global_models) - len(deleted_models_indices)
            if current_num_models <= 1:
                break

            # 计算每个活动集群的数据大小
            # cluster_data_size 的索引对应于原始的 global_model_num
            cluster_data_size = np.zeros(global_model_num)
            active_original_indices = [idx for idx in range(global_model_num) if idx not in deleted_models_indices]

            for client in clients:
                if client.cluster_identity is not None and client.cluster_identity in active_original_indices:
                    client_data_len = len(client.train_data) if client.train_data else 0
                    cluster_data_size[client.cluster_identity] += client_data_len
            
            min_dist_val = self.detection_threshold # 使用检测阈值作为合并的最小距离上限
            merge_pair_original_indices = None

            # 在活动的集群中寻找最小距离的集群对
            for idx1_ptr in range(len(active_original_indices)):
                orig_idx1 = active_original_indices[idx1_ptr]
                for idx2_ptr in range(idx1_ptr + 1, len(active_original_indices)):
                    orig_idx2 = active_original_indices[idx2_ptr]
                    
                    current_dist = cluster_distances[orig_idx1][orig_idx2]
                    if current_dist != -1.0 and current_dist < min_dist_val:
                        min_dist_val = current_dist
                        merge_pair_original_indices = (orig_idx1, orig_idx2)
            
            if merge_pair_original_indices is None: # 没有可合并的集群
                break
            
            # cluster_i_orig 和 cluster_j_orig 是原始索引
            cluster_i_orig, cluster_j_orig = merge_pair_original_indices
            
            # 合并集群 (cluster_i_orig 将吸收 cluster_j_orig)
            # FedDrift.py 论文中，模型 k 是合并后的模型，这里我们将 cluster_i_orig 作为模型 k
            size_i = cluster_data_size[cluster_i_orig]
            size_j = cluster_data_size[cluster_j_orig]
            
            print(f"尝试合并集群 {cluster_j_orig} (大小: {size_j}) 到集群 {cluster_i_orig} (大小: {size_i})，距离: {min_dist_val:.4f}")

            if size_i + size_j > 0:
                model_i_params = parameters_to_vector(self.global_models[cluster_i_orig].parameters())
                model_j_params = parameters_to_vector(self.global_models[cluster_j_orig].parameters())
                
                merged_model_params = ((size_i * model_i_params + size_j * model_j_params) / (size_i + size_j))
                vector_to_parameters(merged_model_params, self.global_models[cluster_i_orig].parameters())
            # else: 两个集群都没有数据点，模型参数不合并，但逻辑上 cluster_j_orig 仍然被合并
            
            if cluster_j_orig not in deleted_models_indices:
                deleted_models_indices.append(cluster_j_orig)
            
            print(f"集群 {cluster_j_orig} 已合并到集群 {cluster_i_orig} (原始索引)")

            # 更新受影响客户端的集群身份
            for client in clients:
                if client.cluster_identity == cluster_j_orig:
                    client.cluster_identity = cluster_i_orig
            
            # 更新 cluster_distances 矩阵以反映合并
            # FedDrift.py 的逻辑: 对于其他集群 l，到新合并集群的距离是 max(dist(l,i), dist(l,j))
            # 然后将 cluster_j 的距离标记为无效
            for l_orig_idx_ptr in range(len(active_original_indices)):
                l_orig = active_original_indices[l_orig_idx_ptr]
                if l_orig == cluster_i_orig or l_orig == cluster_j_orig: # 跳过合并中的集群自身
                    continue
                
                dist_l_i = cluster_distances[l_orig][cluster_i_orig]
                dist_l_j = cluster_distances[l_orig][cluster_j_orig]

                if dist_l_i == -1.0 and dist_l_j == -1.0:
                    new_dist_l_merged = -1.0
                elif dist_l_i == -1.0:
                    new_dist_l_merged = dist_l_j
                elif dist_l_j == -1.0:
                    new_dist_l_merged = dist_l_i
                else:
                    new_dist_l_merged = max(dist_l_i, dist_l_j)
                
                cluster_distances[l_orig][cluster_i_orig] = new_dist_l_merged
                cluster_distances[cluster_i_orig][l_orig] = new_dist_l_merged

            # 将 cluster_j_orig 的所有距离标记为无效，因为它已被合并
            cluster_distances[cluster_j_orig, :] = -1.0
            cluster_distances[:, cluster_j_orig] = -1.0
            # loss_matrix 也应该相应更新，但 FedDrift.py 的参考实现似乎主要依赖于更新后的 cluster_distances
            # 为了简化，我们不在这里重新计算整个 loss_matrix，因为下一次 merge_clusters 调用会重新计算它。
            # 但在当前的 while True 循环中，这可能导致后续迭代使用部分过时的 loss_matrix（如果它被其他地方间接使用）。
            # FedDrift.py 的 while 循环似乎直接操作 cluster_distances 来寻找下一个合并对。

        # 循环结束后，根据 deleted_models_indices 清理 self.global_models
        if deleted_models_indices:
            deleted_models_indices.sort(reverse=True) # 按索引降序删除
            
            # 创建旧索引到新索引的映射
            old_to_new_map = {}
            current_new_idx = 0
            for i in range(global_model_num):
                if i not in deleted_models_indices:
                    old_to_new_map[i] = current_new_idx
                    current_new_idx += 1
            
            # 更新客户端的集群ID
            for client in clients: # 更新所有客户端，不仅仅是 selected_clients
                if client.cluster_identity is not None:
                    if client.cluster_identity in old_to_new_map:
                        client.cluster_identity = old_to_new_map[client.cluster_identity]
                    elif client.cluster_identity in deleted_models_indices: # 指向一个被删除的集群
                        # 这种情况理论上不应该发生，因为上面已经将这些客户端重新分配给了 cluster_i_orig
                        # 但作为安全措施，如果发生了，需要处理
                        print(f"警告: 客户端 {client.id} 的集群ID {client.cluster_identity} 指向一个刚被删除的集群。检查合并逻辑。将其重新分配给映射后的吸收集群或默认集群。")
                        # 尝试找到它被合并到的集群的新索引
                        # 这个逻辑比较复杂，因为不知道它具体被合并到了哪个保留的集群。 
                        # 最安全的是在合并时就正确更新所有客户端的指向。
                        # 鉴于上面的 for client in clients: if client.cluster_identity == cluster_j_orig: client.cluster_identity = cluster_i_orig
                        # 这里的 client.cluster_identity 应该是 cluster_i_orig (如果它之前是 cluster_j_orig)
                        # 所以它应该在 old_to_new_map 中。
                        # 如果 client.cluster_identity 本身就是一个被删除的索引，但不是 cluster_j_orig，则存在问题。
                        # 假设上面的客户端身份更新是正确的，这里不需要特别处理指向 deleted_models_indices 的情况。
                        pass 
            
            # 更新全局模型列表
            new_global_models = []
            for i in range(global_model_num):
                if i not in deleted_models_indices:
                    new_global_models.append(self.global_models[i])
            self.global_models = new_global_models
            print(f"集群合并后，全局模型数量: {len(self.global_models)}")

    def plot_cluster_evolution(self):
        """
        绘制集群数量随轮次变化的图表
        """
        if not self.cluster_counts:
            print("没有集群数量数据可供绘制。")
            return
        
        plt.figure()
        plt.plot(range(len(self.cluster_counts)), self.cluster_counts, marker='o')
        plt.title("集群数量演变")
        plt.xlabel("全局轮次")
        plt.ylabel("集群数量")
        plt.grid(True)
        
        # 保存图表
        plot_path = os.path.join(self.save_folder_name, "results", f"{self.dataset}_{self.algorithm}_cluster_evolution.png")
        if not os.path.exists(os.path.dirname(plot_path)):
            os.makedirs(os.path.dirname(plot_path))
        plt.savefig(plot_path)
        print(f"集群演变图已保存到: {plot_path}")
        if self.args.use_wandb and wandb.run is not None:
            wandb.log({"FedDrift/Cluster Evolution Plot": wandb.Image(plot_path)})
        plt.close()

    def visualize_clustering(self, current_round):
        """
        使用t-SNE可视化客户端的集群分配（基于模型参数或特征表示）
        注意: 这可能非常耗时，特别是对于大型模型或大量客户端。
        """
        if not self.selected_clients or not self.global_models:
            print("没有足够的客户端或全局模型进行可视化。")
            return

        print(f"在轮次 {current_round} 可视化集群...")
        client_model_vectors = []
        client_ids_for_plot = []
        client_assigned_clusters = []

        for client in self.selected_clients:
            if client.cluster_identity is not None:
                # 使用客户端本地模型的参数作为表示
                client_model_vectors.append(parameters_to_vector(client.model.parameters()).cpu().detach().numpy())
                client_ids_for_plot.append(client.id)
                client_assigned_clusters.append(client.cluster_identity)
        
        if not client_model_vectors:
            print("没有可用于可视化的客户端模型向量。")
            return

        client_model_vectors = np.array(client_model_vectors)
        
        # 使用t-SNE降维
        tsne = TSNE(n_components=2, random_state=0, perplexity=min(30, len(client_model_vectors)-1) if len(client_model_vectors) > 1 else 1)
        try:
            embedded_vectors = tsne.fit_transform(client_model_vectors)
        except Exception as e:
            print(f"t-SNE降维失败: {e}")
            return

        plt.figure(figsize=(10, 8))
        unique_cluster_ids = sorted(list(set(client_assigned_clusters)))
        
        # 为每个集群选择一种颜色
        # colors = plt.cm.get_cmap('viridis', len(unique_cluster_ids))
        # 使用更明确的颜色列表，以防 unique_cluster_ids 数量少于 cmap 默认值
        cmap = plt.get_cmap('tab10') # tab10 最多支持10个不同的颜色
        colors = [cmap(i) for i in np.linspace(0, 1, len(unique_cluster_ids))]

        for i, cluster_id in enumerate(unique_cluster_ids):
            indices = [idx for idx, c_id in enumerate(client_assigned_clusters) if c_id == cluster_id]
            if indices:
                plt.scatter(embedded_vectors[indices, 0], embedded_vectors[indices, 1], label=f'Cluster {cluster_id}', color=colors[i % len(colors)])
        
        plt.title(f'客户端集群可视化 (t-SNE) - 轮次 {current_round}')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend()
        plt.grid(True)
        
        # 保存图表
        plot_filename = f"{self.dataset}_{self.algorithm}_tsne_clusters_round_{current_round}.png"
        plot_path = os.path.join(self.save_folder_name, "results", plot_filename)
        if not os.path.exists(os.path.dirname(plot_path)):
            os.makedirs(os.path.dirname(plot_path))
        plt.savefig(plot_path)
        print(f"t-SNE集群可视化图已保存到: {plot_path}")
        if self.args.use_wandb and wandb.run is not None:
            wandb.log({f"FedDrift/t-SNE Visualization Round {current_round}": wandb.Image(plot_path)}, step=current_round)
        plt.close()


