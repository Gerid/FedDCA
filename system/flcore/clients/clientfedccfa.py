import copy
import torch
import time
import numpy as np
from collections import Counter
from typing import Iterator
from torch.nn import Parameter, MSELoss, CosineSimilarity, CrossEntropyLoss
from torch.utils.data import DataLoader

from ..clients.clientbase import Client
from utils.data_utils import read_client_data


class clientFedCCFA(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # 保存所有必要的参数
        self.args = args
        self.learning_rate = args.local_learning_rate
        self.model = copy.deepcopy(args.model)
        
        # FedCCFA特有参数
        self.clf_keys = []  # 分类器层的键
        
        # 根据惩罚类型设置原型损失
        if hasattr(args, 'penalize') and args.penalize == "L2":
            self.proto_criterion = MSELoss().to(self.device)
        else:
            self.proto_criterion = CrossEntropyLoss().to(self.device)
        
        # 初始化原型和标签分布
        self.local_protos = {}
        self.global_protos = []
        self.p_clf_params = []
        self.label_distribution = torch.zeros(args.num_classes, dtype=torch.int)
        self.proto_weight = 0.0
        
        # 优化器配置
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=args.momentum if hasattr(args, 'momentum') else 0.9,
            weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 0
        )
        
        # 损失函数
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
    
    def update_label_distribution(self):
        """更新客户端的标签分布"""
        distribution = Counter(self.train_samples['y'])
        for label, label_size in distribution.items():
            self.label_distribution[label] = label_size
        self.label_distribution = np.array(self.label_distribution)
        prob = self.label_distribution / len(self.train_samples['y'])
        
        # 计算熵以自适应调整原型损失权重
        non_zero_probs = prob[prob > 0]
        entropy = -np.sum(non_zero_probs * np.log(non_zero_probs))
        
        if hasattr(self.args, 'gamma') and self.args.gamma != 0:
            # 根据熵自适应调整原型损失权重
            self.proto_weight = entropy / self.args.gamma
        else:
            self.proto_weight = self.args.lambda_proto if hasattr(self.args, 'lambda_proto') else 0.1
    
    def set_rep_params(self, new_params: Iterator[Parameter]):
        """
        设置模型的表示层参数
        
        Args:
            new_params: 新的参数
        """
        rep_params = [param for name, param in self.model.named_parameters() if name not in self.clf_keys]
        for new_param, local_param in zip(new_params, rep_params):
            local_param.data = new_param.data.clone()
    
    def set_clf_params(self, new_params: Iterator[Parameter]):
        """
        设置模型的分类器参数
        
        Args:
            new_params: 新的参数
        """
        clf_params = [param for name, param in self.model.named_parameters() if name in self.clf_keys]
        for new_param, local_param in zip(new_params, clf_params):
            local_param.data = new_param.data.clone()
    
    def set_label_params(self, label, new_params: Iterator[Parameter]):
        """
        为特定标签设置分类器参数
        
        Args:
            label: 要设置的标签
            new_params: 新的参数
        """
        clf_params = [param for name, param in self.model.named_parameters() if name in self.clf_keys]
        for new_param, local_param in zip(new_params, clf_params):
            local_param.data[label] = new_param.data.clone()
    
    def get_clf_parameters(self):
        """
        获取本地模型的分类器参数
        
        Returns:
            分类器参数列表
        """
        clf_params = [param for name, param in self.model.named_parameters() if name in self.clf_keys]
        return clf_params
    
    def train_with_protos(self, _round):
        """
        使用原型进行解耦训练
        首先训练分类器，然后训练表示层
        
        Args:
            _round: 当前训练轮次
        """
        if self.p_clf_params:
            self.set_clf_params(self.p_clf_params)
            
        self.model.train()
        
        # 创建训练数据加载器
        trainloader = self.load_train_data()
        
        # ------------- 开始训练分类器 -------------
        for name, param in self.model.named_parameters():
            if name in self.clf_keys:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.clf_lr if hasattr(self.args, 'clf_lr') else self.learning_rate,
            weight_decay=self.args.weight_decay if hasattr(self.args, 'weight_decay') else 0,
            momentum=self.args.momentum if hasattr(self.args, 'momentum') else 0.9
        )
        
        # 训练分类器
        clf_epochs = self.args.clf_epochs if hasattr(self.args, 'clf_epochs') else 1
        for epoch in range(clf_epochs):
            for _, (x, y) in enumerate(trainloader):
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                optimizer.step()
        
        # ------------- 开始训练表示层 -------------
        for name, param in self.model.named_parameters():
            if name in self.clf_keys:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.rep_lr if hasattr(self.args, 'rep_lr') else self.learning_rate,
            weight_decay=self.args.weight_decay if hasattr(self.args, 'weight_decay') else 0,
            momentum=self.args.momentum if hasattr(self.args, 'momentum') else 0.9
        )
        
        cos = CosineSimilarity(dim=2).to(self.device)
        
        # 训练表示层
        rep_epochs = self.args.rep_epochs if hasattr(self.args, 'rep_epochs') else 1
        for epoch in range(rep_epochs):
            for _, (x, y) in enumerate(trainloader):
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                optimizer.zero_grad()
                
                # 获取特征和输出
                if hasattr(self.model, 'extract_features'):
                    outputs, features = self.model.extract_features(x, return_features=True)
                else:
                    # 假设模型没有提供特征提取函数，我们使用标准输出
                    outputs = self.model(x)
                    features = None
                    
                loss_sup = self.criterion(outputs, y)
                
                # 如果有全局原型和特征，应用原型损失
                loss_proto = 0.0
                if len(self.global_protos) > 0 and _round >= 20 and features is not None:
                    if self.proto_weight != 0:
                        if hasattr(self.args, 'penalize') and self.args.penalize == "L2":
                            # L2对齐
                            protos = features.clone().detach()
                            for i in range(len(y)):
                                label = y[i].item()
                                if label < len(self.global_protos):  # 确保标签在范围内
                                    protos[i] = self.global_protos[label].detach()
                            loss_proto = self.proto_criterion(features, protos)
                        else:
                            # 对比对齐
                            temperature = self.args.temperature if hasattr(self.args, 'temperature') else 0.5
                            features = features.unsqueeze(1)
                            batch_global_protos = torch.stack([p for p in self.global_protos if p is not None])
                            batch_global_protos = batch_global_protos.repeat(len(y), 1, 1)
                            logits = cos(features, batch_global_protos)
                            loss_proto = self.proto_criterion(logits / temperature, y)
                
                loss = loss_sup + self.proto_weight * loss_proto
                loss.backward()
                optimizer.step()
        
        # 更新本地原型
        with torch.no_grad():
            # 只有当模型支持特征提取时才更新原型
            if hasattr(self.model, 'extract_features'):
                self.local_protos = self.get_local_protos(self.model)
                self.global_protos = [self.local_protos[label] if label in self.local_protos 
                                     else None for label in range(self.args.num_classes)]
    
    def balance_train(self):
        """
        使用平衡的训练集训练分类器
        """
        self.model.train()
        
        # 获取平衡采样的训练集
        balanced_trainloader = self.class_balance_sample()
        
        # 只训练分类器层
        for name, param in self.model.named_parameters():
            if name in self.clf_keys:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.balanced_clf_lr if hasattr(self.args, 'balanced_clf_lr') else self.learning_rate,
            weight_decay=self.args.weight_decay if hasattr(self.args, 'weight_decay') else 0,
            momentum=self.args.momentum if hasattr(self.args, 'momentum') else 0.9
        )
        
        # 训练分类器
        balanced_epochs = self.args.balanced_epochs if hasattr(self.args, 'balanced_epochs') else 1
        for epoch in range(balanced_epochs):
            for _, (x, y) in enumerate(balanced_trainloader):
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                optimizer.step()
    
    def class_balance_sample(self):
        """
        创建类平衡的训练加载器
        
        Returns:
            平衡采样的数据加载器
        """
        # 获取训练数据
        train_data = self.load_train_data().dataset
        
        # 按类别组织索引
        indices_by_label = {}
        labels = np.array(train_data.targets if hasattr(train_data, 'targets') else [y for _, y in train_data])
        
        for idx, label in enumerate(labels):
            if label not in indices_by_label:
                indices_by_label[label] = []
            indices_by_label[label].append(idx)
        
        # 找出每类样本的最小数量
        min_size = min([len(indices) for indices in indices_by_label.values()])
        min_size = min(min_size, 5)  # 限制最小采样数
        
        # 从每类中随机采样相同数量的样本
        balanced_indices = []
        for label, indices in indices_by_label.items():
            if len(indices) > 0:
                balanced_indices.extend(np.random.choice(indices, min_size, replace=False))
        
        # 随机打乱索引
        np.random.shuffle(balanced_indices)
        
        # 创建平衡的数据加载器
        from torch.utils.data import Subset
        balanced_dataset = Subset(train_data, balanced_indices)
        balanced_loader = DataLoader(
            balanced_dataset, 
            batch_size=self.batch_size,
            shuffle=True
        )
        
        return balanced_loader
    
    def get_local_protos(self, model):
        """
        获取每个类的本地原型
        
        Args:
            model: 本地模型
            
        Returns:
            每个类的原型平均值
        """
        proto_dict = {}
        train_loader = DataLoader(self.load_train_data().dataset, batch_size=min(2048, len(self.train_samples)), shuffle=True)
        
        with torch.no_grad():
            for _, (x, y) in enumerate(train_loader):
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                # 获取特征
                if hasattr(model, 'extract_features'):
                    _, features = model.extract_features(x, return_features=True)
                else:
                    # 如果模型没有提供特征提取函数，使用一个简单的替代方法
                    # 注意：这可能不是理想的方法，但是为了兼容性而提供
                    outputs = model(x)
                    features = outputs  # 这里应该使用模型的中间特征，这里仅是示例
                
                # 按标签组织特征
                for i in range(len(y)):
                    label = y[i].item()
                    if label in proto_dict:
                        proto_dict[label].append(features[i, :])
                    else:
                        proto_dict[label] = [features[i, :]]
        
        # 计算每个类的原型平均值
        proto_mean = {}
        for label, proto_list in proto_dict.items():
            if proto_list:  # 确保列表不为空
                proto_mean[label] = torch.mean(torch.stack(proto_list), dim=0)
        
        return proto_mean
    
    def fine_tune(self):
        """
        微调分类器
        """
        self.model.train()
        
        trainloader = self.load_train_data()
        
        # 只训练分类器层
        for name, param in self.model.named_parameters():
            if name in self.clf_keys:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.p_lr if hasattr(self.args, 'p_lr') else self.learning_rate / 10,
            weight_decay=self.args.weight_decay if hasattr(self.args, 'weight_decay') else 0,
            momentum=self.args.momentum if hasattr(self.args, 'momentum') else 0.9
        )
        
        # 微调5个epoch
        for epoch in range(5):
            for _, (x, y) in enumerate(trainloader):
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                optimizer.step()
    
    def train(self):
        """
        为兼容基类，提供标准训练方法
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
    
    def load_train_data(self, batch_size=None):
        """加载训练数据"""
        if not batch_size:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True, data_dir=self.drift_data_dir, 
                                     iteration=self.current_iteration if self.use_drift_dataset else None)
        return DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
