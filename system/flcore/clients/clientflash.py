import copy
import torch
import time
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader

from ..clients.clientbase import Client
from utils.data_utils import read_client_data


class clientFlash(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # 保存所有必要的参数
        self.args = args
        self.learning_rate = args.local_learning_rate
        self.model = copy.deepcopy(args.model)
        
        # Flash特有参数
        self.loss_decrement = args.loss_decrement if hasattr(args, 'loss_decrement') else 0.01
        self.prev_val_loss = -1  # 初始化上一轮验证损失
        self.local_update = None  # 存储本地更新
        
        # 优化器配置
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=args.momentum if hasattr(args, 'momentum') else 0.9,
            weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 0.0001
        )
        # Add grad_clip parameter
        self.grad_clip = args.grad_clip if hasattr(args, 'grad_clip') else 10.0

    def train(self):
        """执行早停训练"""
        trainloader = self.load_train_data()
        testloader = self.load_test_data()  # 用于验证
        
        # 记录初始参数
        init_params = parameters_to_vector(self.model.parameters())
        
        # 设置模型为训练模式
        self.model.train()
        self.model.to(self.device)
        
        start_time = time.time()
        
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
            
        # 进行本地训练
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                
                # Add gradient clipping here
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.optimizer.step()
            
            # 在验证集上计算损失
            val_loss = self.get_validation_loss(testloader)
            
            # 早停条件：如果损失下降不够显著，则停止训练
            if self.prev_val_loss != -1:
                delta = self.prev_val_loss - val_loss
                if 0 < delta < self.loss_decrement / (epoch + 1):
                    break
            
            self.prev_val_loss = val_loss
        
        # 计算本地更新
        self.local_update = parameters_to_vector(self.model.parameters()) - init_params

        # 更新训练时间
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        
        return self.local_update
    
    def get_validation_loss(self, dataloader):
        """计算给定数据集上的验证损失"""
        self.model.eval()
        total_loss = 0
        samples = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                output = self.model(x)
                loss = self.loss(output, y)
                
                samples += y.shape[0]
                total_loss += loss.item() * y.shape[0]
        
        return total_loss / samples if samples > 0 else float('inf')
