import copy
import torch
import time
import numpy as np
from torch.utils.data import DataLoader

from ..clients.clientbase import Client
from utils.data_utils import read_client_data


class clientFedDrift(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # 保存所有必要的参数
        self.args = args
        self.learning_rate = args.local_learning_rate
        self.model = copy.deepcopy(args.model)
        
        # FedDrift特有参数
        self.prev_train_samples = copy.deepcopy(train_samples)  # 保存前一轮的训练数据
        self.cluster_identity = 0  # 初始化集群身份
        
        # 优化器配置
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=args.momentum if hasattr(args, 'momentum') else 0.9,
            weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 0
        )
        
        # 损失函数
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
    
    def get_loss(self, model, data_samples, batch_size=32):
        """
        计算模型在给定数据上的损失
        
        Args:
            model: 要评估的模型
            data_samples: 评估数据
            batch_size: 批大小
            
        Returns:
            float: 平均损失值
        """
        # 创建临时数据加载器
        temp_dataloader = self.create_dataloader(data_samples, batch_size)
        
        # 计算损失
        total_loss = 0.0
        total_samples = 0
        
        model.eval()
        with torch.no_grad():
            for x, y in temp_dataloader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                outputs = model(x)
                loss = self.criterion(outputs, y)
                
                total_loss += loss.item() * y.size(0)
                total_samples += y.size(0)
                
        # 返回平均损失
        if total_samples > 0:
            return total_loss / total_samples
        else:
            return float('inf')  # 如果没有样本，返回无穷大
    
    def create_dataloader(self, data_samples, batch_size):
        """
        从数据样本创建DataLoader
        
        Args:
            data_samples: 数据样本
            batch_size: 批大小
            
        Returns:
            DataLoader: 数据加载器
        """
        # 使用read_client_data创建数据集
        if isinstance(data_samples, dict):
            # 如果是训练或测试样本格式
            dataset = read_client_data(self.dataset, self.id, is_train=True, 
                                     data_dir=self.drift_data_dir if hasattr(self, 'drift_data_dir') else None,
                                     iteration=self.current_iteration if hasattr(self, 'use_drift_dataset') and self.use_drift_dataset else None,
                                     manually_load=True, specific_data=data_samples)
        else:
            # 如果是数据集对象
            dataset = data_samples
            
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    def clustering(self, global_models):
        """
        计算集群身份 - 检测概念漂移并选择最佳模型
        
        当检测到概念漂移时，返回None，表示需要创建新的全局模型
        
        Args:
            global_models: 用于计算集群身份的全局模型列表
        """
        prev_loss_list = []
        loss_list = []
        
        # 计算每个全局模型在前一轮训练数据上的损失
        for model in global_models:
            prev_loss = self.get_loss(model, self.prev_train_samples)
            prev_loss_list.append(prev_loss)
        min_prev_loss = min(prev_loss_list)
        
        # 计算每个全局模型在当前训练数据上的损失
        for model in global_models:
            loss = self.get_loss(model, self.train_samples)
            loss_list.append(loss)
        min_loss = min(loss_list)
        
        # 获取检测阈值
        detection_threshold = self.args.detection_threshold if hasattr(self.args, 'detection_threshold') else 0.1
        
        # 如果当前最小损失比前一轮最小损失增加超过阈值，检测到概念漂移
        if min_loss > min_prev_loss + detection_threshold:
            # 检测到概念漂移，为所有漂移的客户端创建新模型
            self.cluster_identity = None
            print(f"Client {self.id} detected concept drift! min_loss={min_loss:.4f}, min_prev_loss={min_prev_loss:.4f}")
        else:
            # 从现有集群中选择最佳模型
            self.cluster_identity = int(np.argmin(loss_list))
            print(f"Client {self.id} selected cluster {self.cluster_identity} (losses: {[round(l, 4) for l in loss_list]})")
    
    def train(self):
        """
        进行本地训练
        """
        trainloader = self.load_train_data()
        self.model.train()
        
        start_time = time.time()
        max_local_epochs = self.local_epochs
        
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        
        for _ in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
        
        # 更新训练时间成本
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
    
    def set_parameters(self, model_params):
        """
        设置模型参数
        
        Args:
            model_params: 要设置的模型参数
        """
        for old_param, new_param in zip(self.model.parameters(), model_params):
            old_param.data = new_param.data.clone()
    
    def update_prev_train_samples(self):
        """
        更新前一轮的训练样本
        """
        self.prev_train_samples = copy.deepcopy(self.train_samples)
