import copy
import torch
import time
import numpy as np
import os
from flcore.servers.serverbase import Server
from flcore.clients.clientflash import clientFlash
from threading import Thread
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import matplotlib.pyplot as plt
import wandb # Added wandb import


class Flash(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        
        # 设置客户端
        self.set_slow_clients()
        self.set_clients(clientFlash)
        
        # Flash特有参数
        self.beta1 = args.beta1 if hasattr(args, 'beta1') else 0.9
        self.beta2 = args.beta2 if hasattr(args, 'beta2') else 0.99
        self.beta3 = 0
        self.ftau = args.ftau if hasattr(args, 'ftau') else 1e-8
        self.server_lr = args.server_learning_rate if hasattr(args, 'server_learning_rate') else 0.01 
        
        # Flash优化器动量
        self.first_momentum = 0
        self.second_momentum = self.ftau ** 2
        self.prev_second_momentum = 0
        self.delta_momentum = 0
        
        # 性能追踪
        self.Budget = []
        self.client_data_size = {}  # 记录每个客户端的数据大小
        
        print(f"\nFlash设置完成!")
        print(f"参与率 / 总客户端数: {self.join_ratio} / {self.num_clients}")
        print(f"Beta1: {self.beta1}, Beta2: {self.beta2}, Tau: {self.ftau}")

    def train(self):
        """训练过程的主控制流"""
        for i in range(self.global_rounds + 1):
            self.current_round = i # Set current_round as a class attribute for potential use in other methods
            s_t = time.time()
            current_round = i # Define current_round for local use in this loop, passed to methods
            
            # 每轮选择客户端
            self.selected_clients = self.select_clients()
            self.apply_drift_transformation()
            
            # 向选定客户端发送模型
            self.send_parameters(self.selected_clients)
            
            # 客户端本地训练
            for client in self.selected_clients:
                client.train()
            
            # 基于客户端更新聚合
            self.aggregate_by_updates(self.selected_clients)
            
            # 再次向选定客户端发送更新后的模型
            self.send_parameters(self.selected_clients)
            
            # 定期评估
            if i % self.eval_gap == 0:
                print(f"\n--- 第 {i} 轮评估 ---")
                self.evaluate(current_round=current_round, is_global=True) # Pass current_round
            
            # 计算耗时
            e_t = time.time()
            self.Budget.append(e_t - s_t)
            
            # 是否达到预设的准确率要求提前结束
            if self.auto_break and len(self.rs_test_acc) > 0 and self.rs_test_acc[-1] > 0.97:
                break
                
        # 最终评估和输出
        print("\n训练完成!")
        self.evaluate(current_round=self.global_rounds) # Pass final round to evaluate
        
        # 输出耗时统计
        avg_time = sum(self.Budget) / len(self.Budget)
        print(f"平均每轮耗时: {avg_time:.2f}秒")
        
        # 保存结果和模型
        self.save_results()
        # self.save_model(current_round=self.global_rounds) # Pass current_round for final model save

    def get_client_data_size(self, clients):
        """
        记录每个客户端的数据大小
        
        Args:
            clients: 客户端列表
        """
        for client in clients:
            self.client_data_size[client.id] = client.train_samples

    def send_parameters(self, clients):
        """向客户端发送模型参数"""
        for client in clients:
            client.set_parameters(self.global_model.state_dict())

    def aggregate_by_updates(self, clients):
        """
        基于客户端更新聚合模型
        
        Args:
            clients: 客户端列表
        """
        # 获取客户端更新
        total_size = 0
        # Initialize update_sum on the correct device
        # Assuming self.device is defined in the base Server class and is the target device
        update_sum = torch.zeros_like(parameters_to_vector(self.global_model.parameters()), device=self.device)
        
        active_clients_count = 0
        for client in clients:
            if client.local_update is None:
                print(f"Warning: Client {client.id} has no local_update. Skipping.")
                continue
            
            # Ensure client.local_update is a tensor and on the correct device
            local_update_tensor = client.local_update
            if not isinstance(local_update_tensor, torch.Tensor):
                try:
                    local_update_tensor = torch.tensor(local_update_tensor, device=self.device, dtype=update_sum.dtype)
                except Exception as e:
                    print(f"Error converting client {client.id}'s local_update to tensor: {e}. Skipping.")
                    continue
            
            client_size = client.train_samples
            total_size += client_size
            update_sum += client_size * local_update_tensor.to(self.device) # Ensure it's on the same device
            active_clients_count += 1
        

        # 计算加权平均更新
        aggregated_update_tensor = update_sum / total_size
        aggregated_update_np = aggregated_update_tensor.detach().cpu().numpy()


        
        # Flash自适应优化器更新
        self.first_momentum = self.beta1 * self.first_momentum + (1 - self.beta1) * aggregated_update_np
        
        squared_aggregated_update = np.square(aggregated_update_np)
        self.prev_second_momentum = self.second_momentum
        self.second_momentum = self.beta2 * self.second_momentum + (1 - self.beta2) * squared_aggregated_update
        
        # 计算自适应Beta3和Delta动量
        # Add epsilon to denominator for beta3 calculation
        beta3_denominator = (np.abs(squared_aggregated_update - self.second_momentum) + 
                               np.abs(self.prev_second_momentum) + 1e-9) # Epsilon for stability
        self.beta3 = np.abs(self.prev_second_momentum) / beta3_denominator
        
        self.delta_momentum = self.beta3 * self.delta_momentum + (1 - self.beta3) * (
                squared_aggregated_update - self.second_momentum)
        


        # 最终更新
        # Use self.second_momentum directly for np.sqrt, as in the reference
        # Note: if self.second_momentum can become negative due to numerical issues, this could error.
        # It should theoretically be non-negative.
        sqrt_term = np.sqrt(self.second_momentum)

        denominator = sqrt_term - self.delta_momentum + self.ftau
        
        # REMOVED: denominator = np.where(denominator < 1e-6, 1e-6, denominator)
        # Now using raw denominator as per reference. This might lead to issues if it's invalid.

        # Check if denominator is too small or NaN/Inf
        # This check is still useful to log potential issues.

        final_update_np = self.server_lr * self.first_momentum / denominator
        


        # 更新全局模型
        cur_global_params = parameters_to_vector(self.global_model.parameters())
        final_update_tensor = torch.tensor(final_update_np, device=cur_global_params.device, dtype=cur_global_params.dtype)
        new_global_params = cur_global_params + final_update_tensor
        vector_to_parameters(new_global_params, self.global_model.parameters())


