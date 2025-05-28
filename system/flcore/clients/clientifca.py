import copy
import torch
import time
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb # Add wandb import

from ..clients.clientbase import Client
from utils.data_utils import read_client_data


class clientIFCA(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # 保存所有必要的参数
        self.args = args
        self.learning_rate = args.local_learning_rate
        self.learning_rate_decay_gamma = args.learning_rate_decay_gamma
        self.model = copy.deepcopy(args.model)
        
        # IFCA特有参数
        self.cluster_identity = 0
        
        # 优化器配置
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=args.momentum if hasattr(args, 'momentum') else 0.9,
            weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 0
        )
        
        # 学习率调度器
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=self.learning_rate_decay_gamma
        )
        
        # 损失函数
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
    
    def clustering(self, global_models):
        """
        计算集群身份 - 选择最适合客户端数据的模型
        
        Args:
            global_models: 用于计算集群身份的全局模型列表
        """
        loss_list = []
        
        # 创建用于计算损失的数据加载器
        trainloader = self.load_train_data(batch_size=32)  # 使用较小的批量以加快计算速度
        
        # 计算每个模型在客户端数据上的损失
        for model_idx, model in enumerate(global_models): # Added model_idx for logging
            model.eval()  # 设置为评估模式
            total_loss = 0
            sample_count = 0
            
            with torch.no_grad():
                for x, y in trainloader:
                    if isinstance(x, list):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    
                    output = model(x)
                    loss = self.criterion(output, y)
                    total_loss += loss.item() * y.size(0)
                    sample_count += y.size(0)
                    
 
            
            # 计算平均损失
            avg_loss = total_loss / sample_count if sample_count > 0 else float('inf')
            loss_list.append(avg_loss)

            # Log individual cluster model loss for this client
            if wandb.run is not None:
                wandb.log({f"Client_{self.id}/Loss_for_Cluster_Model_{model_idx}": avg_loss})
        
        # 选择损失最小的模型
        self.cluster_identity = int(np.argmin(loss_list))
        
        # Log the chosen cluster and the full list of losses
        if wandb.run is not None:
            log_data = {f"Client_{self.id}/Chosen_Cluster": self.cluster_identity}
            for idx, l in enumerate(loss_list):
                log_data[f"Client_{self.id}/Loss_List_Cluster_{idx}"] = l
            wandb.log(log_data)

        # 通知
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
        
        # 更新学习率
        if self.args.learning_rate_decay:  # 仅当 learning_rate_decay 标志为 True 时才执行 step
            if hasattr(self, 'learning_rate_scheduler'):
                self.learning_rate_scheduler.step()



