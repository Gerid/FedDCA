import os
import json
import numpy as np
import traceback
import matplotlib.pyplot as plt

class DriftHandler:
    """处理概念漂移相关的功能"""
    
    def __init__(self, args=None):
        self.args = args
        self.current_iteration = 0
        self.max_iterations = args.max_iterations if hasattr(args, 'max_iterations') else 200
        self.use_drift_dataset = args.use_drift_dataset if hasattr(args, 'use_drift_dataset') else False
        self.drift_data_dir = args.drift_data_dir if hasattr(args, 'drift_data_dir') else None
        self.drift_iterations = []
        self.num_concepts = 0
        self.client_concepts = {}
        self.client_drift_types = {}
        
        # 如果启用了概念漂移数据集，加载漂移配置
        if self.use_drift_dataset and self.drift_data_dir:
            self.load_drift_config()
    
    def load_drift_config(self):
        """加载概念漂移数据集的配置信息"""
        try:
            # 构建配置文件路径
            config_path = os.path.join(self.drift_data_dir, "drift_info", "concept_config.json")
            
            if not os.path.exists(config_path):
                print(f"警告: 未找到漂移配置文件 {config_path}")
                return
            
            # 加载配置
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # 提取关键配置信息
            self.drift_iterations = config.get('drift_iterations', [])
            self.num_concepts = config.get('num_concepts', 5)
            self.client_concepts = config.get('client_concepts', {})
            self.client_drift_types = config.get('client_drift_types', {})
            
            print(f"已加载漂移配置: {self.num_concepts} 个概念, {len(self.drift_iterations)} 个漂移点")
            print(f"漂移将在以下迭代发生: {self.drift_iterations}")
            
            return True
            
        except Exception as e:
            print(f"加载漂移配置失败: {str(e)}")
            print(traceback.format_exc())
            return False
    
    def update_clients_iteration(self, clients):
        """更新客户端的迭代状态"""
        if not self.use_drift_dataset:
            return
            
        print(f"\n更新客户端迭代状态到 {self.current_iteration}")
        for client in clients:
            if hasattr(client, 'update_iteration'):
                client.update_iteration(self.current_iteration)
        
        # 检查是否是漂移点
        if self.current_iteration in self.drift_iterations:
            print(f"\n⚠️ 在迭代 {self.current_iteration} 发生概念漂移")
        
        # 更新迭代计数器，为下一轮做准备
        self.current_iteration = (self.current_iteration + 1) % self.max_iterations
    
    def setup_clients_for_drift(self, clients):
        """为客户端设置漂移配置"""
        if not self.use_drift_dataset:
            return
            
        for client in clients:
            client.drift_args = {
                'drift_iterations': self.drift_iterations,
                'concepts': self.client_concepts.get(str(client.id), []),
                'drift_type': self.client_drift_types.get(str(client.id), 'sudden')
            }
            # 启用客户端的概念漂移数据集模式
            client.use_drift_dataset = True
            client.drift_data_dir = self.drift_data_dir
            client.max_iterations = self.max_iterations
    
    def is_drift_point(self):
        """检查当前迭代是否是漂移点"""
        return self.current_iteration in self.drift_iterations
    
    def analyze_concept_drift(self, client_cluster_history, output_dir='results/drift_analysis'):
        """分析概念漂移对客户端聚类的影响"""
        if not self.use_drift_dataset or not client_cluster_history:
            return
            
        try:
            import os
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 转换聚类历史为数组
            client_ids = list(client_cluster_history.keys())
            max_history_len = max(len(history) for history in client_cluster_history.values())
            
            cluster_history_array = np.zeros((len(client_ids), max_history_len))
            for i, client_id in enumerate(client_ids):
                history = client_cluster_history[client_id]
                cluster_history_array[i, :len(history)] = history
                # 填充缺失值
                if len(history) < max_history_len:
                    cluster_history_array[i, len(history):] = history[-1]
            
            # 分析聚类稳定性
            cluster_changes = np.zeros(max_history_len - 1)
            for i in range(1, max_history_len):
                # 计算每轮有多少客户端改变了聚类
                changes = np.sum(cluster_history_array[:, i] != cluster_history_array[:, i-1])
                cluster_changes[i-1] = changes / len(client_ids)
            
            # 绘制聚类变化率
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, max_history_len), cluster_changes, marker='o')
            
            # 标记漂移点
            drift_rounds = []
            for drift_iter in self.drift_iterations:
                # 找到最接近漂移迭代的训练轮次
                closest_round = min(range(1, max_history_len), key=lambda x: abs(x - drift_iter/10))
                drift_rounds.append(closest_round)
                
            for drift_round in drift_rounds:
                if 1 <= drift_round < max_history_len:
                    plt.axvline(x=drift_round, color='r', linestyle='--', alpha=0.5)
                    plt.text(drift_round, 0.05, f'Drift', rotation=90)
            
            plt.xlabel('Training Round')
            plt.ylabel('Proportion of Clients Changing Clusters')
            plt.title('Cluster Stability Analysis with Concept Drift')
            plt.grid(True)
            plt.savefig(f'{output_dir}/cluster_stability_analysis.png')
            
            # 分析每个客户端的聚类轨迹
            plt.figure(figsize=(12, 8))
            for i, client_id in enumerate(client_ids):
                plt.plot(range(1, max_history_len + 1), cluster_history_array[i], 
                         marker='.', label=f'Client {client_id}')
            
            # 标记漂移点
            for drift_round in drift_rounds:
                if 1 <= drift_round <= max_history_len:
                    plt.axvline(x=drift_round, color='r', linestyle='--', alpha=0.5)
                    plt.text(drift_round, 0.5, f'Drift', rotation=90)
            
            plt.xlabel('Training Round')
            plt.ylabel('Cluster Assignment')
            plt.title('Client Cluster Trajectories')
            plt.grid(True)
            # 如果客户端太多，不显示图例
            if len(client_ids) <= 10:
                plt.legend()
            plt.savefig(f'{output_dir}/client_cluster_trajectories.png')
            
            return True
            
        except Exception as e:
            print(f"分析概念漂移失败: {str(e)}")
            print(traceback.format_exc())
            return False
