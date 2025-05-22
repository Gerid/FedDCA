import numpy as np
import torch
import torch.nn as nn
import copy
import random
from torch.utils.data import DataLoader
from typing import List, Tuple

class DCM:
    def __init__(self,
                cid: int,
                loss: nn.Module,
                train_data: List[Tuple], 
                batch_size: int, 
                rand_percent: int, 
                layer_idx: int = 0,
                eta: float = 1.0,
                device: str = 'cpu', 
                threshold: float = 0.1,
                num_pre_loss: int = 10) -> None:
        """
        Initialize DCM module

        Args:
            cid: Client ID. 
            loss: The loss function. 
            train_data: The reference of the local training data.
            batch_size: Weight learning batch size.
            rand_percent: The percent of the local training data to sample.
            layer_idx: Control the weight range. By default, all the layers are selected. Default: 0
            eta: Weight learning rate. Default: 1.0
            device: Using cuda or cpu. Default: 'cpu'
            threshold: Train the weight until the standard deviation of the recorded losses is less than a given threshold. Default: 0.1
            num_pre_loss: The number of the recorded losses to be considered to calculate the standard deviation. Default: 10

        Returns:
            None.
        """

        self.cid = cid
        self.loss = loss
        self.train_data = train_data
        self.batch_size = batch_size
        self.rand_percent = rand_percent
        self.layer_idx = layer_idx
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.device = device

        self.weights = None # Learnable local aggregation weights.
        self.start_phase = True


    def adaptive_local_aggregation(self, 
                            global_model: nn.Module,
                            local_model: nn.Module) -> None:
        """
        Generates the Dataloader for the randomly sampled local training data and 
        preserves the lower layers of the update. 

        Args:
            global_model: The received global/aggregated model. 
            local_model: The trained local model. 

        Returns:
            None.
        """

        # randomly sample partial local training data
        rand_ratio = self.rand_percent / 100
        rand_num = int(rand_ratio*len(self.train_data))
        rand_idx = random.randint(0, len(self.train_data)-rand_num)
        rand_loader = DataLoader(self.train_data[rand_idx:rand_idx+rand_num], self.batch_size, drop_last=True)


        # obtain the references of the parameters
        params_g = list(global_model.parameters())
        params = list(local_model.parameters())

        # deactivate ALA at the 1st communication iteration
        if torch.sum(params_g[0] - params[0]) == 0:
            return

        # preserve all the updates in the lower layers
        for param, param_g in zip(params[:-self.layer_idx], params_g[:-self.layer_idx]):
            param.data = param_g.data.clone()


        # temp local model only for weight learning
        model_t = copy.deepcopy(local_model)
        params_t = list(model_t.parameters())

        # only consider higher layers
        params_p = params[-self.layer_idx:]
        params_gp = params_g[-self.layer_idx:]
        params_tp = params_t[-self.layer_idx:]

        # frozen the lower layers to reduce computational cost in Pytorch
        for param in params_t[:-self.layer_idx]:
            param.requires_grad = False

        # used to obtain the gradient of higher layers
        # no need to use optimizer.step(), so lr=0
        optimizer = torch.optim.SGD(params_tp, lr=0)

        # initialize the weight to all ones in the beginning
        if self.weights == None:
            self.weights = [torch.ones_like(param.data).to(self.device) for param in params_p]

        # initialize the higher layers in the temp local model
        for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp,
                                                self.weights):
            param_t.data = param + (param_g - param) * weight

        # weight learning
        losses = []  # record losses
        cnt = 0  # weight training iteration counter
        while True:
            for x, y in rand_loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                output = model_t(x)
                loss_value = self.loss(output, y) # modify according to the local objective
                loss_value.backward()

                # update weight in this batch
                for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                        params_gp, self.weights):
                    weight.data = torch.clamp(
                        weight - self.eta * (param_t.grad * (param_g - param)), 0, 1)

                # update temp local model in this batch
                for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                        params_gp, self.weights):
                    param_t.data = param + (param_g - param) * weight

            losses.append(loss_value.item())
            cnt += 1

            # only train one epoch in the subsequent iterations
            if not self.start_phase:
                break

            # train the weight until convergence
            if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold:
                print('Client:', self.cid, '\tStd:', np.std(losses[-self.num_pre_loss:]),
                    '\tALA epochs:', cnt)
                break

        self.start_phase = False

        # obtain initialized local model
        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone()

class VariationalWassersteinClustering:
    def __init__(self, num_clusters, max_iter=100, tol=1e-4, alpha=0.1):
        self.num_clusters = num_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha  # 权重衰减率
        self.y = None  # 聚类中心
        self.h = None  # 权重
        self.clusters = {}  # 存储聚类信息，使用字典管理聚类
    
    def initialize(self, data):
        """
        初始化聚类中心 y 和权重 h
        """
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(data)
        self.y = kmeans.cluster_centers_
        self.h = np.zeros(self.num_clusters)
    
    def assign_clusters(self, data):
        """
        将数据点分配到最近的聚类
        """
        distances = np.dot(data, self.y.T) + self.h  # 计算 ⟨z_i, y_j⟩ + h_j
        assignments = np.argmin(distances, axis=1)
        return assignments
    
    def update_centers(self, data, assignments, mu):
        """
        更新聚类中心 y_j
        """
        for j in range(self.num_clusters):
            cluster_data = data[assignments == j]
            if len(cluster_data) > 0:
                self.y[j] = np.sum(cluster_data * mu[assignments == j, np.newaxis], axis=0) / np.sum(mu[assignments == j])
    
    def update_weights(self, data, assignments, mu):
        """
        更新权重 h_j 使用简单的梯度下降
        """
        for j in range(self.num_clusters):
            cluster_mu = mu[assignments == j]
            grad = np.sum(cluster_mu) - np.sum(cluster_mu)
            self.h[j] -= self.alpha * grad
    
    def fit(self, data, mu, clients):
        """
        执行VWC算法，数据点为`data`，每个数据点对应一个客户端
        """
        self.initialize(data)
        for iteration in range(self.max_iter):
            assignments = self.assign_clusters(data)
            old_y = self.y.copy()
            old_h = self.h.copy()

            # 更新每个聚类的代理表征和中心模型
            for i, assignment in enumerate(assignments):
                cluster_id = assignment
                if cluster_id not in self.clusters:
                    self.clusters[cluster_id] = Cluster(cluster_id)
                self.clusters[cluster_id].add_client(clients[i])
            
            # 更新代理表征和中心模型
            for cluster in self.clusters.values():
                cluster.update_proxy_representation()
                cluster.update_model()

            # 更新聚类中心和权重
            self.update_centers(data, assignments, mu)
            self.update_weights(data, assignments, mu)

            # 检查收敛
            if np.linalg.norm(self.y - old_y) < self.tol and np.linalg.norm(self.h - old_h) < self.tol:
                print(f"VWC converged at iteration {iteration}")
                break



class Cluster:
    def __init__(self, cluster_id, proxy_representation=None, model=None):
        self.cluster_id = cluster_id  # 聚类ID
        self.proxy_representation = proxy_representation  # 聚类的代理表征（可以是张量）
        self.model = model  # 聚类的中心模型
        self.clients = []  # 聚类中的客户端列表

    def add_client(self, client):
        """将客户端添加到聚类中"""
        self.clients.append(client)

    def remove_client(self, client):
        """从聚类中移除客户端"""
        self.clients.remove(client)

    def update_proxy_representation(self):
        """根据聚类中的客户端更新代理表征"""
        if len(self.clients) == 0:
            return
        # 计算聚类内所有客户端表征的平均值作为代理表征
        representations = [client.intermediate_output for client in self.clients]
        self.proxy_representation = torch.mean(torch.stack(representations), dim=0)

    def update_model(self):
        """根据聚类内的客户端模型更新中心模型"""
        if len(self.clients) == 0:
            return
        # 聚合模型（这里假设有一个方法来聚合模型）
        self.model = self.aggregate_models([client.model for client in self.clients])

    def aggregate_models(self, models):
        """聚合多个模型"""
        # 假设你有一个方法来聚合模型参数，通常是简单的平均
        model_params = [model.state_dict() for model in models]
        avg_state_dict = {key: torch.zeros_like(value) for key, value in model_params[0].items()}
        
        for param in avg_state_dict:
            avg_state_dict[param] = torch.mean(
                torch.stack([model_param[param].float() for model_param in model_params]), dim=0
            )
        
        # 将平均后的参数加载到新模型中
        avg_model = models[0]  # 假设用第一个模型的结构来构造平均模型
        avg_model.load_state_dict(avg_state_dict)
        return avg_model
