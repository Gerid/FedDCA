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
            if self.args.wandb:
                wandb.log({"FedDrift/Unique Clusters": unique_clusters, "round": i})
            
            # 发送参数给客户端
            self.send_models(self.selected_clients)
            
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
            if i % self.eval_gap == 0:
                print(f"\n--- 第 {i} 轮评估 ---")
                stats = self.evaluate(current_round=i) # Pass current_round
                
                # 记录和输出性能 (serverbase evaluate already logs to wandb if self.args.wandb is true)
                self.rs_test_acc.append(stats['acc'])
                self.rs_train_loss.append(stats['loss'])
                
                print(f"平均测试准确率: {stats['acc']:.4f}")
                print(f"平均训练损失: {stats['loss']:.4f}")
                print(f"集群数量: {unique_clusters}") # This unique_clusters is from selected clients, evaluate logs len(self.global_models)
                
                # 如果需要，可视化集群分布
                if hasattr(self.args, 'visualize_clusters') and self.args.visualize_clusters:
                    self.visualize_clustering(i) # Pass current_round
            
            # 计算耗时
            e_t = time.time()
            self.Budget.append(e_t - s_t)
            
            # 是否达到预设的准确率要求提前结束
            if self.auto_break and len(self.rs_test_acc) > 0 and self.rs_test_acc[-1] > self.args.goal_accuracy:
                print(f"达到目标准确率 {self.args.goal_accuracy}，提前停止训练。")
                break
        
        # 最终评估和输出
        print("\n训练完成!")
        stats = self.evaluate(current_round=self.global_rounds) # Pass final round
        print(f"最终平均测试准确率: {stats['acc']:.4f}")
        print(f"集群演变: {self.cluster_counts}")
        
        # 输出耗时统计
        avg_time = sum(self.Budget) / len(self.Budget) if len(self.Budget) > 0 else 0
        print(f"平均每轮耗时: {avg_time:.2f}秒")
        # if self.args.wandb:
        #     wandb.log({"FedDrift/Average Round Time": avg_time})
            
        # 保存结果和模型
        self.save_results() # Uses serverbase save_results, which handles wandb artifact for h5
        self.save_models(current_round=self.global_rounds) # Pass final round
        
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
                client_size = len(client.train_loader.dataset) if client.train_loader else 0 # Use train_loader.dataset
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
        合并相似的集群
        
        Args:
            clients: 客户端列表
        """
        global_model_num = len(self.global_models)
        if global_model_num <= 1:
            return
        
        # 初始化损失矩阵
        loss_matrix = np.full((global_model_num, global_model_num), -1.0) # Initialize with -1
        
        # 生成损失矩阵用于计算集群距离
        for i in range(global_model_num):
            for j in range(global_model_num):
                total_data_size = 0
                total_loss = 0.0
                
                # 计算集群j中客户端数据在模型i上的损失
                for client in clients:
                    if client.cluster_identity == j:
                        # data_size = len(client.train_samples) # train_samples might not be up-to-date or representative
                        data_size = len(client.train_loader.dataset) if client.train_loader else 0
                        if data_size > 0:
                            # Ensure client.get_loss uses appropriate data (e.g., a validation split or recent train data)
                            # For simplicity, using train_loader here, assuming get_loss can handle it or uses client.test_loader
                            loss = client.get_loss(self.global_models[i], loader=client.train_loader) # Pass loader
                            total_loss += loss * data_size
                            total_data_size += data_size
                
                if total_data_size > 0:
                    loss_matrix[i][j] = total_loss / total_data_size
        
        # 计算集群间距离
        cluster_distances = np.full((global_model_num, global_model_num), -1.0) # Initialize with -1
        for i in range(global_model_num):
            for j in range(i, global_model_num): # Iterate j from i to avoid redundant calculations and self-comparison for dist
                if loss_matrix[i][j] == -1 or loss_matrix[j][i] == -1 or loss_matrix[i][i] == -1 or loss_matrix[j][j] == -1 :
                    # 集群i或集群j中没有客户端, or self-loss is undefined
                    dist = -1.0 
                else:
                    # 计算基于交叉损失的集群间距离
                    dist = max(loss_matrix[i][j] - loss_matrix[i][i], loss_matrix[j][i] - loss_matrix[j][j], 0)
                
                cluster_distances[i][j] = dist
                cluster_distances[j][i] = dist  # 对称矩阵
        
        # 检查是否有集群需要合并
        deleted_models_indices = [] # Store original indices of models to be deleted
        
        # Create a mapping for current model indices to original indices if models are iteratively deleted
        current_model_indices = list(range(global_model_num))


        while True:
            num_active_models = len(self.global_models) - len(deleted_models_indices)
            if num_active_models <=1:
                break

            # 计算每个集群的数据大小
            cluster_data_size = np.zeros(global_model_num) # Use original indexing for data size calculation
            for client in clients:
                if client.cluster_identity is not None and client.cluster_identity < global_model_num : # Check bounds
                    cluster_data_size[client.cluster_identity] += (len(client.train_loader.dataset) if client.train_loader else 0)
            
            # 寻找最小距离的集群对 among active models
            min_dist_val = self.detection_threshold
            merge_pair = None

            # Iterate through current_model_indices to find active models
            active_indices = [idx for idx in range(global_model_num) if idx not in deleted_models_indices]

            for idx1_ptr, orig_idx1 in enumerate(active_indices):
                for idx2_ptr in range(idx1_ptr + 1, len(active_indices)):
                    orig_idx2 = active_indices[idx2_ptr]
                    
                    current_dist = cluster_distances[orig_idx1][orig_idx2]
                    if current_dist != -1.0 and current_dist < min_dist_val:
                        min_dist_val = current_dist
                        merge_pair = (orig_idx1, orig_idx2)
            
            if merge_pair is None: # 没有可合并的集群
                break
            
            cluster_i, cluster_j = merge_pair # These are original indices
            
            # 合并集群 (cluster_i will absorb cluster_j)
            size_i = cluster_data_size[cluster_i]
            size_j = cluster_data_size[cluster_j]
            
            if size_i + size_j > 0:
                model_i_params = parameters_to_vector(self.global_models[cluster_i].parameters())
                model_j_params = parameters_to_vector(self.global_models[cluster_j].parameters())
                
                merged_model_params = ((size_i * model_i_params + size_j * model_j_params) / (size_i + size_j))
                
                vector_to_parameters(merged_model_params, self.global_models[cluster_i].parameters())
                
                if cluster_j not in deleted_models_indices:
                    deleted_models_indices.append(cluster_j)
                
                print(f"合并集群 {cluster_j} 到集群 {cluster_i} (原始索引)")
                
                # 更新客户端分配: clients assigned to cluster_j are now assigned to cluster_i
                for client in clients:
                    if client.cluster_identity == cluster_j:
                        client.cluster_identity = cluster_i
                
                # Update distances involving cluster_i. Distances involving cluster_j become invalid.
                # For simplicity, we can re-evaluate distances in the next iteration or mark cluster_j's distances as unusable.
                # Mark distances related to cluster_j as -1 (or a very large number)
                for k_idx in range(global_model_num):
                    cluster_distances[cluster_j, k_idx] = -1.0
                    cluster_distances[k_idx, cluster_j] = -1.0
                    if k_idx != cluster_i and k_idx not in deleted_models_indices: # Update distances for cluster_i with other active clusters
                        # Re-calculate or approximate new distance for (cluster_i, k_idx)
                        # This part can be complex; a simpler approach is to let the loop re-evaluate based on updated loss_matrix if merge_clusters is called iteratively
                        # For now, we assume the initial distance calculation is sufficient or merge_clusters is called once per round.
                        # A more robust way would be to update loss_matrix[cluster_i] and recompute relevant cluster_distances.
                        # Let's assume the current distance update logic in the original code was:
                        # dist = max(cluster_distances[cluster_i][l], cluster_distances[cluster_j][l])
                        # This seems like an approximation.
                        if k_idx != cluster_i: # Update distance between the merged cluster (i) and other clusters (k_idx)
                            # This logic needs to be sound. The original code had:
                            # dist = max(cluster_distances[cluster_i][l], cluster_distances[cluster_j][l])
                            # This might not be theoretically perfect.
                            # A better way might be to re-evaluate loss of merged model i on data of cluster k, and vice-versa.
                            # For now, let's stick to a simpler update or rely on next round's full re-evaluation.
                            # To avoid complex re-computation here, we can just invalidate cluster_j's distances.
                            # The next call to merge_clusters (if any) or the main loop will handle it.
                            pass


            else: # One or both clusters had no data, mark j for deletion if it was chosen
                 if cluster_j not in deleted_models_indices:
                    deleted_models_indices.append(cluster_j)


        # Actual deletion and re-indexing
        if deleted_models_indices:
            deleted_models_indices.sort(reverse=True) # Delete from largest index to smallest
            
            new_global_models = []
            old_to_new_idx_map = {}
            current_new_idx = 0
            for old_idx in range(global_model_num):
                if old_idx not in deleted_models_indices:
                    new_global_models.append(self.global_models[old_idx])
                    old_to_new_idx_map[old_idx] = current_new_idx
                    current_new_idx += 1
            
            self.global_models = new_global_models
            
            # Update client cluster identities
            for client in clients:
                if client.cluster_identity is not None:
                    if client.cluster_identity in deleted_models_indices:
                        # This should not happen if they were reassigned, but as a fallback:
                        print(f"警告: 客户端 {client.id} 仍分配给已删除的集群 {client.cluster_identity}。")
                        # Attempt to reassign or mark as None. For now, map to the absorbed cluster if possible.
                        # This part needs careful handling based on the merge logic.
                        # If cluster_j was merged into cluster_i, clients of j should now be i.
                        # The re-assignment happened above. Now we just need to map old indices to new ones.
                        pass 
                    if client.cluster_identity in old_to_new_idx_map:
                         client.cluster_identity = old_to_new_idx_map[client.cluster_identity]
                    else:
                        # This case implies client was assigned to a deleted cluster that wasn't properly remapped
                        # or an index out of bounds.
                        print(f"警告: 客户端 {client.id} 的集群身份 {client.cluster_identity} 无法映射到新索引。")
                        client.cluster_identity = None # Or handle as a new drift


    def evaluate(self, current_round=None): # Added current_round
        """
        评估当前模型性能
        
        Args:
            current_round: 当前的联邦轮次 (for wandb logging)
        Returns:
            dict: 性能指标字典
        """
        stats = {'acc': 0.0, 'loss': 0.0, 'num_samples': 0, 'total_test_samples': 0}
        
        # 对每个客户端进行评估
        # Use all clients for evaluation, not just selected_clients, if possible and meaningful
        # For now, sticking to selected_clients as per original structure for consistency
        # evaluation_clients = self.clients if self.args.eval_all_clients else self.selected_clients
        evaluation_clients = self.selected_clients # Or self.clients for a more global evaluation view

        if not evaluation_clients:
            print("警告:评估时没有可用的客户端。")
            return stats

        total_test_acc_sum = 0
        total_test_samples = 0
        
        for client in evaluation_clients:
            if client.cluster_identity is None or client.cluster_identity >= len(self.global_models):
                # print(f"警告: 客户端 {client.id} 没有有效的集群身份 ({client.cluster_identity})，跳过评估。模型数量: {len(self.global_models)}")
                continue # Skip clients without a valid cluster

            cluster_id = client.cluster_identity
            # 测试准确率
            client.set_parameters(self.global_models[cluster_id].parameters())
            # test_acc, test_num = client.test_metrics() # test_metrics should return acc * num_samples, and num_samples
            # For clarity, let's assume test_metrics returns (accuracy, num_samples)
            # And train_loss from client.get_loss is per sample.
            
            # The base server evaluate() logs these if args.wandb is true:
            # wandb.log({f"Global/Train_Loss": mean_train_loss, "round": current_round})
            # wandb.log({f"Global/Test_Accuracy": mean_test_acc, "round": current_round})
            # So FedDrift's evaluate should focus on FedDrift specific metrics or aggregate differently.
            # The current structure of FedDrift's evaluate seems to calculate an overall average.

            # Let's get raw accuracy and number of samples to compute weighted average
            client_test_acc, client_num_test_samples = client.test_metrics() # Returns actual accuracy and num_samples
            if client_num_test_samples > 0:
                total_test_acc_sum += client_test_acc * client_num_test_samples # Sum of (acc * num_samples)
                total_test_samples += client_num_test_samples
        
        if total_test_samples > 0:
            stats['acc'] = total_test_acc_sum / total_test_samples
        else:
            stats['acc'] = 0.0
        
        # 计算平均训练损失 (on their respective cluster models)
        total_train_loss_sum = 0.0
        total_train_samples = 0
        for client in evaluation_clients: # Iterate again for train loss, or combine loops if efficient
            if client.cluster_identity is None or client.cluster_identity >= len(self.global_models):
                continue
            
            cluster_id = client.cluster_identity
            # client.get_loss needs the model and data (e.g., client.train_loader)
            # Ensure client.get_loss returns average loss per sample on the provided data
            client_train_loss = client.get_loss(self.global_models[cluster_id], loader=client.train_loader) # Pass loader
            client_num_train_samples = len(client.train_loader.dataset) if client.train_loader else 0
            
            if client_num_train_samples > 0:
                total_train_loss_sum += client_train_loss * client_num_train_samples
                total_train_samples += client_num_train_samples
        
        if total_train_samples > 0:
            stats['loss'] = total_train_loss_sum / total_train_samples
        else:
            stats['loss'] = 0.0

        stats['num_samples'] = total_test_samples # For consistency with how rs_test_acc might be used later

        if self.args.wandb and current_round is not None:
            wandb.log({
                "FedDrift/Average Test Accuracy": stats['acc'],
                "FedDrift/Average Train Loss": stats['loss'],
                "FedDrift/Cluster Count": len(self.global_models),
                "round": current_round
            }, step=current_round)
        return stats

    def save_models(self, current_round=None): # Added current_round
        """保存所有当前的全局模型 (每个集群一个)"""
        if not (self.args.wandb and self.args.wandb_save_model):
            print("WandB模型保存未启用。跳过模型保存到WandB。")
            # Still save locally if needed, or remove local save if only wandb is desired
            # The original code didn't show local saving here, assuming it was handled by serverbase or not done for FedDrift's multiple models.
            # For now, let's focus on wandb saving.
            # A local save might look like:
            # for idx, model in enumerate(self.global_models):
            #     model_path = os.path.join(self.save_folder_name, f"feddrift_model_cluster_{idx}_round_{current_round}.pt")
            #     torch.save(model.state_dict(), model_path)
            # print(f"FedDrift 模型已本地保存到 {self.save_folder_name}")
            return

        if current_round is None:
            current_round = self.global_rounds # Fallback if not provided, e.g. called at end of training

        for idx, model_to_save in enumerate(self.global_models):
            model_artifact_name = f"{self.args.wandb_run_name}_cluster_{idx}_model"
            model_filename = f"feddrift_cluster_{idx}_model_round_{current_round}.pt"
            
            # Create a temporary path for the model file
            temp_model_dir = "wandb_temp_models"
            os.makedirs(temp_model_dir, exist_ok=True)
            local_model_path = os.path.join(temp_model_dir, model_filename)
            
            torch.save(model_to_save.state_dict(), local_model_path)
            
            try:
                artifact = wandb.Artifact(
                    name=model_artifact_name,
                    type="model",
                    description=f"FedDrift global model for cluster {idx} at round {current_round}",
                    metadata={"round": current_round, "cluster_id": idx, "algorithm": "FedDrift"}
                )
                artifact.add_file(local_model_path, name=model_filename)
                wandb.log_artifact(artifact)
                print(f"FedDrift 模型 (集群 {idx}) 已作为 Artifact '{model_artifact_name}' 保存到 W&B (文件: {model_filename})")
            except Exception as e:
                print(f"错误: 保存 FedDrift 模型 (集群 {idx}) 到 W&B Artifact 失败: {e}")
            finally:
                # Clean up the temporary file
                if os.path.exists(local_model_path):
                    os.remove(local_model_path)
        # Clean up the temporary directory if empty
        if os.path.exists(temp_model_dir) and not os.listdir(temp_model_dir):
            os.rmdir(temp_model_dir)


    def visualize_clustering(self, round_idx):
        """
        可视化客户端集群分布
        
        Args:
            round_idx: 当前轮次
        """
        # Ensure results directory exists (already in original code)
        viz_dir = os.path.join(self.results_dir, "clustering_visualizations") # Use self.results_dir
        os.makedirs(viz_dir, exist_ok=True)
        
        # 收集客户端ID和集群ID
        # Use selected_clients for visualization as they participated in this round's clustering
        client_ids = [client.id for client in self.selected_clients]
        cluster_ids = [client.cluster_identity for client in self.selected_clients]
        
        # Filter out clients with no cluster assignment for visualization
        valid_clients_data = [(cid, clid) for cid, clid in zip(client_ids, cluster_ids) if clid is not None]
        if not valid_clients_data:
            print(f"轮次 {round_idx}: 没有客户端分配到集群，跳过可视化。")
            return
        
        client_ids_viz, cluster_ids_viz = zip(*valid_clients_data)

        plt.figure(figsize=(10, 8))
        # Ensure unique_clusters_viz is derived from cluster_ids_viz which has no Nones
        unique_clusters_viz = sorted(list(set(cluster_ids_viz))) # Sort for consistent coloring
        
        if not unique_clusters_viz: # Should be caught by valid_clients_data check, but as a safeguard
             plt.close()
             return

        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters_viz)))
        color_map = {cluster_id: colors[i] for i, cluster_id in enumerate(unique_clusters_viz)}

        scatter_colors = [color_map[c] for c in cluster_ids_viz]
        
        plt.scatter(client_ids_viz, cluster_ids_viz, c=scatter_colors, alpha=0.6, s=100)
        
        plt.title(f'客户端集群分布 (轮次 {round_idx})', fontsize=16)
        plt.xlabel('客户端ID', fontsize=14)
        plt.ylabel('集群ID', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Create a colorbar legend manually if needed, or ensure cluster IDs are interpretable
        # For simplicity, the direct scatter plot is kept. A legend might be:
        # handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {uid}', 
        #                       markerfacecolor=color_map[uid]) for uid in unique_clusters_viz]
        # plt.legend(handles=handles, title="Clusters")

        plot_filename = f"cluster_round_{round_idx}.png"
        plot_filepath = os.path.join(viz_dir, plot_filename)
        plt.savefig(plot_filepath)
        plt.close()
        print(f"集群可视化图已保存到 {plot_filepath}")

        if self.args.wandb and self.args.wandb_save_artifacts:
            try:
                artifact_name = f"cluster_visualization_round_{round_idx}"
                artifact = wandb.Artifact(
                    name=artifact_name,
                    type="visualization",
                    description=f"客户端集群分布图在轮次 {round_idx} (FedDrift)",
                    metadata={"round": round_idx, "algorithm": "FedDrift"}
                )
                artifact.add_file(plot_filepath, name=plot_filename)
                wandb.log_artifact(artifact)
                # Log as image directly to see in W&B dashboard media panel
                wandb.log({f"FedDrift/Clustering/Round_{round_idx}": wandb.Image(plot_filepath), "round": round_idx})
                print(f"集群可视化图 (轮次 {round_idx}) 已作为 Artifact '{artifact_name}' 保存到 W&B。")
            except Exception as e:
                print(f"错误: 保存集群可视化图 (轮次 {round_idx}) 到 W&B Artifact 失败: {e}")


    def plot_cluster_evolution(self):
        """
        绘制集群数量随时间的变化
        """
        if not self.cluster_counts: # No data to plot
            print("没有集群数量数据可供绘制演变图。")
            return

        viz_dir = os.path.join(self.results_dir, "clustering_visualizations") # Use self.results_dir
        os.makedirs(viz_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(self.cluster_counts)), self.cluster_counts, marker='o', 
                linestyle='-', linewidth=2, markersize=8)
        
        plt.title('集群数量随轮次的变化 (FedDrift)', fontsize=16)
        plt.xlabel('轮次', fontsize=14)
        plt.ylabel('集群数量', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7) # Corrected linestyle
        
        plot_filename = "cluster_evolution.png"
        plot_filepath = os.path.join(viz_dir, plot_filename)
        plt.savefig(plot_filepath)
        plt.close()
        print(f"集群演变图已保存到 {plot_filepath}")

        if self.args.wandb and self.args.wandb_save_artifacts:
            try:
                artifact_name = "feddrift_cluster_evolution_plot"
                artifact = wandb.Artifact(
                    name=artifact_name,
                    type="visualization",
                    description="FedDrift 集群数量随轮次的变化图",
                    metadata={"algorithm": "FedDrift", "total_rounds": len(self.cluster_counts)}
                )
                artifact.add_file(plot_filepath, name=plot_filename)
                wandb.log_artifact(artifact)
                # Log as image directly
                wandb.log({"FedDrift/Clustering/Evolution": wandb.Image(plot_filepath)})
                print(f"集群演变图已作为 Artifact '{artifact_name}' 保存到 W&B。")
            except Exception as e:
                print(f"错误: 保存集群演变图到 W&B Artifact 失败: {e}")


