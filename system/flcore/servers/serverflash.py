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
        self.server_lr = args.server_learning_rate if hasattr(args, 'server_learning_rate') else 1.0
        
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
                self.evaluate(current_round=current_round) # Pass current_round
            
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
        self.save_model(current_round=self.global_rounds) # Pass current_round for final model save

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
            client.set_parameters(self.global_model)

    def aggregate_by_updates(self, clients):
        """
        基于客户端更新聚合模型
        
        Args:
            clients: 客户端列表
        """
        # 获取客户端更新
        total_size = 0
        update_sum = torch.zeros_like(parameters_to_vector(self.global_model.parameters()))
        
        for client in clients:
            client_size = len(client.train_samples)
            total_size += client_size
            update_sum += client_size * client.local_update
        
        # 计算加权平均更新
        aggregated_update = update_sum / total_size
        aggregated_update = aggregated_update.detach().cpu().numpy()
        
        # Flash自适应优化器更新
        self.first_momentum = self.beta1 * self.first_momentum + (1 - self.beta1) * aggregated_update
        self.prev_second_momentum = self.second_momentum
        self.second_momentum = self.beta2 * self.second_momentum + (1 - self.beta2) * np.square(aggregated_update)
        
        # 计算自适应Beta3和Delta动量
        self.beta3 = np.abs(self.prev_second_momentum) / (
                np.abs(np.square(aggregated_update) - self.second_momentum) + np.abs(self.prev_second_momentum))
        self.delta_momentum = self.beta3 * self.delta_momentum + (1 - self.beta3) * (
                np.square(aggregated_update) - self.second_momentum)
        
        # 最终更新
        aggregated_update = self.server_lr * self.first_momentum / (
                np.sqrt(self.second_momentum) - self.delta_momentum + self.ftau)
        
        # 更新全局模型
        cur_global_params = parameters_to_vector(self.global_model.parameters())
        new_global_params = cur_global_params + torch.tensor(aggregated_update).to(self.device)
        
        vector_to_parameters(new_global_params, self.global_model.parameters())

    def evaluate(self, current_round=None):
        """评估当前模型性能"""
        stats = {'acc': 0.0, 'loss': 0.0, 'num_samples': 0}
        
        for client in self.selected_clients:
            client.set_parameters(self.global_model)
            acc, num = client.test_metrics()
            
            stats['acc'] += acc
            stats['num_samples'] += num
        
        stats['acc'] = stats['acc'] / stats['num_samples'] if stats['num_samples'] > 0 else 0
        
        self.rs_test_acc.append(stats['acc'])
        
        # 计算客户端平均训练损失
        train_loss = 0
        train_samples = 0
        for client in self.selected_clients:
            client.set_parameters(self.global_model)
            losses, num_samples = client.train_metrics()
            
            train_loss += losses
            train_samples += num_samples
            
        stats['loss'] = train_loss / train_samples if train_samples > 0 else 0
        self.rs_train_loss.append(stats['loss'])
        
        print(f"平均测试准确率: {stats['acc']:.4f}")
        print(f"平均训练损失: {stats['loss']:.4f}")

        # Wandb logging is handled by serverbase.evaluate, no need to duplicate here unless Flash has specific metrics.
        # If Flash needs to log additional specific metrics:
        # if wandb.run is not None and hasattr(self, 'current_round'): # Ensure current_round is available
        #     wandb.log({
        #         "Flash Specific Metric": some_value 
        #     }, step=self.current_round) # self.current_round should be the round number for this evaluation
        
        return stats

    def save_model(self, current_round=None): # Add current_round parameter
        """保存全局模型"""
        model_path = os.path.join("saved_models", self.dataset) # Consider using "models" consistent with serverbase
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        model_filename = f"Flash_{self.dataset}_{self.times}.pt"
        model_filepath = os.path.join(model_path, model_filename)
        # torch.save(self.global_model.state_dict(), model_filepath) # serverbase saves the whole model, be consistent
        torch.save(self.global_model, model_filepath)
        print(f"Global model saved to {model_filepath}")

        if self.args.wandb_save_model and wandb.run is not None and current_round is not None:
            try:
                # Use the run name prefix from args for consistency
                artifact_name = f'{self.args.wandb_run_name_prefix}_global_model' 
                model_artifact = wandb.Artifact(
                    artifact_name, 
                    type='model',
                    description=f'Global model for Flash algorithm at round {current_round}',
                    metadata={'dataset': self.dataset, 'algorithm': self.algorithm, 'times': self.times, 'round': current_round}
                )
                # Add file with a round-specific name if desired, or just the standard name
                model_artifact.add_file(model_filepath, name=f'flash_model_round_{current_round}.pt')
                aliases = ['latest', f'round_{current_round}']
                if current_round == self.global_rounds: # Mark as final if it's the last round
                    aliases.append('final')
                wandb.log_artifact(model_artifact, aliases=aliases)
                print(f"Flash global model saved to wandb as artifact at round {current_round}")
            except Exception as e:
                print(f"Error saving Flash model to wandb: {e}")

    def save_results(self):
        """保存训练结果"""
        algo = self.algorithm
        result_path = os.path.join("results", algo)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
            
        result_path = os.path.join(result_path, f"{self.dataset}_{algo}_{self.goal}_{self.times}.h5")
        print(f"结果保存到: {result_path}")
        
        import h5py
        with h5py.File(result_path, 'w') as hf:
            hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
            hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

        # Wandb artifact saving for results is handled by serverbase.save_results
        # No need to duplicate here unless Flash has a different results file structure or additional artifacts.
        # If Flash had its own specific artifact to save, it would be done here:
        # if self.args.wandb_save_artifacts and wandb.run is not None:
        #     try:
        #         # ... create and log Flash-specific artifact ...
        #         print(f"Flash specific results saved to wandb.")
        #     except Exception as e:
        #         print(f"Error saving Flash specific results to wandb: {e}")
