import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
import numpy as np
import warnings

# 忽略PCA中的警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

class VariationalWassersteinClustering(nn.Module):
    def __init__(self, num_clients, num_clusters, proxy_dim, pca_dim=4, sinkhorn_reg=0.2, temperature=0.5):
        super().__init__()
        self.num_clients = num_clients
        self.num_clusters = num_clusters
        self.pca_dim = pca_dim
        self.sinkhorn_reg = sinkhorn_reg  # 增大正则化系数，促进更均衡分配
        self.temperature = temperature  # 降低温度参数，使分配更加明确
        
        # 改进1: 更好的中心初始化，使用更大的初始扰动
        self.centers = nn.Parameter(torch.randn(num_clusters, pca_dim) * 0.5)
        
        # 改进2: 使用更大的随机初始化logits，促进更多样化的初始分配
        self.logits = nn.Parameter(torch.randn(num_clients, num_clusters) * 0.5)
        
        # 改进3: 添加均衡度量指标追踪
        self.cluster_balance = 0.0

    def reduce_dim(self, proxy_points):
        # proxy_points: [num_clients, num_samples, feat_dim]
        num_clients, num_samples, feat_dim = proxy_points.shape
        # 降维维度不能超过样本数和特征数
        real_dim = min(self.pca_dim, num_samples-1, feat_dim)
        reduced = []
        for i in range(num_clients):
            # PCA在CPU上做，结果转回原device
            pca = PCA(n_components=real_dim)
            arr = proxy_points[i].detach().cpu().numpy()
            arr_pca = pca.fit_transform(arr)
            arr_pca = torch.tensor(arr_pca, dtype=torch.float32, device=proxy_points.device)
            # 若降维后不足pca_dim，pad到pca_dim
            if arr_pca.shape[1] < self.pca_dim:
                arr_pca = F.pad(arr_pca, (0, self.pca_dim-arr_pca.shape[1]))
            reduced.append(arr_pca)
        # [num_clients, num_samples, pca_dim]
        return torch.stack(reduced)

    def forward(self, proxy_points):
        # proxy_points: [num_clients, num_samples, feat_dim]
        x = self.reduce_dim(proxy_points)  # [num_clients, num_samples, pca_dim]
        # 用均值代表每个client分布
        x_mean = x.mean(dim=1)  # [num_clients, pca_dim]
        
        # 计算距离 [num_clients, num_clusters]
        dist = torch.cdist(x_mean, self.centers)
        
        # 距离取负，这样距离越小越有可能分配到该集群
        neg_dist = -dist / self.temperature  # 改进4: 使用温度参数调节softmax平滑度
        
        # 使用负距离直接影响分配概率
        assignment_logits = self.logits + neg_dist
        
        # softmax分配概率
        probs = F.softmax(assignment_logits, dim=1)
        
        # 计算聚类均衡性指标（熵）
        cluster_probs = probs.mean(dim=0)  # 每个簇的平均分配概率
        ideal_prob = 1.0 / self.num_clusters  # 理想情况下每个簇的分配概率
        
        # 计算分配熵，值越高表示分配越均衡
        entropy = -torch.sum(cluster_probs * torch.log(cluster_probs + 1e-10))
        max_entropy = -torch.sum(torch.ones_like(cluster_probs) * ideal_prob * 
                                torch.log(torch.ones_like(cluster_probs) * ideal_prob + 1e-10))
        self.cluster_balance = entropy / max_entropy  # 归一化熵作为均衡指标
        
        # 改进5: 修改损失函数，添加簇间距离最大化项和集群均衡项
        pairwise_dist = torch.cdist(self.centers, self.centers)
        # 将对角线设为最大值，不考虑自身距离
        mask = torch.eye(self.num_clusters, device=pairwise_dist.device) * 1e10
        pairwise_dist = pairwise_dist + mask
        
        # 簇间最小距离的负值（越小越好，所以取负）
        min_dist = -torch.min(pairwise_dist)
        
        # 集群非均衡惩罚（使用Gini系数）
        cluster_assignments = torch.argmax(probs, dim=1)
        counts = torch.bincount(cluster_assignments, minlength=self.num_clusters).float()
        proportions = counts / counts.sum()
        gini = torch.sum(proportions * (1 - proportions))
          # 完整损失：距离损失 + 熵正则化 + 簇间距离最大化 + 集群均衡惩罚
        distance_loss = (probs * dist).sum()
        entropy_reg = -self.sinkhorn_reg * entropy
        
        # 改进：增强均衡惩罚，以处理客户端分配不到某些簇的问题
        # 使用均方差作为不均衡度量，标准差越小越均衡
        # 先计算每个簇的客户端数量
        client_counts = torch.bincount(cluster_assignments, minlength=self.num_clusters).float()
        # 计算均值和标准差
        mean_count = client_counts.mean()
        std_count = torch.sqrt(torch.mean((client_counts - mean_count) ** 2))
        # 不均衡惩罚项
        imbalance_penalty = std_count / (mean_count + 1e-10)  # 归一化标准差
        
        # 组合损失
        loss = distance_loss + entropy_reg + 0.2 * min_dist + 0.5 * gini + 0.8 * imbalance_penalty
        
        return probs, loss

    def get_cluster_assignments(self):
        with torch.no_grad():
            # 使用与forward中相同的逻辑计算分配
            dummy_data = torch.zeros(self.num_clients, 1, self.pca_dim, device=self.centers.device)
            probs, _ = self.forward(dummy_data)            # 打印每个集群的分配数量以便调试
            assignments = torch.argmax(probs, dim=1)
            cluster_counts = torch.bincount(assignments, minlength=self.num_clusters)
            print(f"Cluster distribution: {cluster_counts.tolist()}")
            print(f"Cluster balance score (0-1): {self.cluster_balance.item():.4f}")
            return assignments    

    def fit(self, proxy_points, lr=0.01, max_iter=100, tol=1e-5):
        # 改进6: 使用Adam优化器，更大的学习率和更多迭代次数
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        prev_loss = float('inf')
        
        # 保存训练过程中的损失值和分配情况，用于调试
        losses = []
        balance_scores = []
        cluster_distributions = []
        
        # 在训练前尝试做一次K-means++初始化
        with torch.no_grad():
            x = self.reduce_dim(proxy_points)
            x_mean = x.mean(dim=1)  # [num_clients, pca_dim]
            centers = self._kmeans_pp_initialization(x_mean, self.num_clusters)
            self.centers.data.copy_(centers)
        
        # 强制初始平均分配
        if self.num_clients >= self.num_clusters:
            clients_per_cluster = self.num_clients // self.num_clusters
            remainder = self.num_clients % self.num_clusters
            
            new_logits = torch.ones_like(self.logits) * (-10.0)
            client_idx = 0
            
            for cluster_id in range(self.num_clusters):
                cluster_size = clients_per_cluster + (1 if cluster_id < remainder else 0)
                for _ in range(cluster_size):
                    if client_idx < self.num_clients:
                        new_logits[client_idx, cluster_id] = 10.0
                        client_idx += 1
            
            self.logits.data.copy_(new_logits)
        
        # 在训练前尝试做一次K-means++初始化
        with torch.no_grad():
            x = self.reduce_dim(proxy_points)
            x_mean = x.mean(dim=1)  # [num_clients, pca_dim]
            centers = self._kmeans_pp_initialization(x_mean, self.num_clusters)
            self.centers.data.copy_(centers)
        
        # 强制初始平均分配
        if self.num_clients >= self.num_clusters:
            clients_per_cluster = self.num_clients // self.num_clusters
            remainder = self.num_clients % self.num_clusters
            
            new_logits = torch.ones_like(self.logits) * (-10.0)
            client_idx = 0
            
            for cluster_id in range(self.num_clusters):
                cluster_size = clients_per_cluster + (1 if cluster_id < remainder else 0)
                for _ in range(cluster_size):
                    if client_idx < self.num_clients:
                        new_logits[client_idx, cluster_id] = 10.0
                        client_idx += 1
            
            self.logits.data.copy_(new_logits)
        
        # 在训练前尝试做一次K-means++初始化
        with torch.no_grad():
            x = self.reduce_dim(proxy_points)
            x_mean = x.mean(dim=1)  # [num_clients, pca_dim]
            centers = self._kmeans_pp_initialization(x_mean, self.num_clusters)
            self.centers.data.copy_(centers)
        
        # 强制初始平均分配
        if self.num_clients >= self.num_clusters:
            clients_per_cluster = self.num_clients // self.num_clusters
            remainder = self.num_clients % self.num_clusters
            
            new_logits = torch.ones_like(self.logits) * (-10.0)
            client_idx = 0
            
            for cluster_id in range(self.num_clusters):
                cluster_size = clients_per_cluster + (1 if cluster_id < remainder else 0)
                for _ in range(cluster_size):
                    if client_idx < self.num_clients:
                        new_logits[client_idx, cluster_id] = 10.0
                        client_idx += 1
            
            self.logits.data.copy_(new_logits)
        
        for i in range(max_iter):
            optimizer.zero_grad()
            probs, loss = self.forward(proxy_points)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            balance_scores.append(self.cluster_balance.item())
            
            if i % 10 == 0:
                # 打印当前集群分配情况以监控训练进度
                with torch.no_grad():
                    assignments = torch.argmax(probs, dim=1)
                    cluster_counts = torch.bincount(assignments, minlength=self.num_clusters)
                    print(f"Iter {i}, Loss: {loss.item():.4f}, Balance: {self.cluster_balance.item():.4f}, Counts: {cluster_counts.tolist()}")
                    cluster_distributions.append(cluster_counts.tolist())
            
            # 更严格的早停条件，考虑改善不显著且均衡性良好
            if prev_loss - loss.item() < tol and self.cluster_balance.item() > 0.8 and i > 20:
                print(f"Early stopping at iteration {i}, good balance achieved")
                break
                
            # 如果均衡性太差但已经迭代了一定次数，尝试温度退火
            if i > 20 and i % 10 == 0 and self.cluster_balance.item() < 0.5:
                self.temperature = max(0.1, self.temperature * 0.8)  # 降低温度使概率分布更sharper
                print(f"Adjusting temperature to {self.temperature:.4f}")
                
            prev_loss = loss.item()
            scheduler.step()
        
        # 训练完成后检查集群分配情况
        assignments = self.get_cluster_assignments()
        
        # 如果所有样本都被分到一个集群，尝试强制重新分配
        if torch.unique(assignments).size(0) == 1 or self.cluster_balance.item() < 0.5:
            print("Warning: Poor clustering balance! Attempting forced reassignment...")
            # 强制分配到不同集群
            self._force_diverse_clusters(proxy_points)
            assignments = self.get_cluster_assignments()
            
        return assignments
        
    def _force_diverse_clusters(self, proxy_points):
        """强制进行多样化的集群分配"""
        with torch.no_grad():
            x = self.reduce_dim(proxy_points)
            x_mean = x.mean(dim=1)  # [num_clients, pca_dim]
            
            # 使用K-means++初始化集群中心
            centers = self._kmeans_pp_initialization(x_mean, self.num_clusters)
            self.centers.data.copy_(centers)
            
            # 改进7: 使用更强的强制分配策略
            if self.num_clients >= self.num_clusters:
                # 计算到每个中心的距离
                dist = torch.cdist(x_mean, self.centers)
                
                # 排序确定分配
                sorted_indices = torch.argsort(dist, dim=1)
                
                # 初始化计数器跟踪每个集群的分配数量
                cluster_counts = torch.zeros(self.num_clusters, device=dist.device)
                
                # 为每个客户端选择集群
                new_assignments = torch.zeros(self.num_clients, dtype=torch.long, device=dist.device)
                
                # 先分配第一批以确保每个集群至少有一个客户端
                for cluster_id in range(min(self.num_clusters, self.num_clients)):
                    # 找到距离该中心最近的未分配客户端
                    min_dist_idx = torch.argmin(dist[:, cluster_id])
                    new_assignments[min_dist_idx] = cluster_id
                    cluster_counts[cluster_id] += 1
                    # 将已分配客户端的距离设为无穷大，防止重复分配
                    dist[min_dist_idx, :] = float('inf')
                
                # 分配剩余客户端到最小的集群
                for client_idx in range(self.num_clients):
                    if new_assignments[client_idx] == 0 and cluster_counts[0] > 0:  # 已被分配
                        continue
                    # 找到当前最小的集群
                    min_cluster = torch.argmin(cluster_counts)
                    new_assignments[client_idx] = min_cluster
                    cluster_counts[min_cluster] += 1
                
                # 根据新分配设置logits
                new_logits = torch.ones_like(self.logits) * (-100.0)  # 先设置所有logits为一个很小的值
                for client_idx in range(self.num_clients):
                    cluster_id = new_assignments[client_idx]
                    new_logits[client_idx, cluster_id] = 100.0  # 为分配的集群设置一个很大的值
                
                self.logits.data.copy_(new_logits)
                
                print(f"Forced assignment successful, cluster distribution: {cluster_counts.tolist()}")
    
    def _kmeans_pp_initialization(self, data, k):
        """使用K-means++算法初始化聚类中心"""
        n_samples = data.size(0)
        device = data.device
        
        # 随机选择第一个中心点
        centers = torch.zeros(k, data.size(1), device=device)
        first_idx = torch.randint(0, n_samples, (1,)).item()
        centers[0] = data[first_idx]
        
        # 选择其余中心点
        for i in range(1, k):
            # 计算每个点到已选中心的最小距离
            dists = torch.cdist(data, centers[:i])
            min_dists, _ = torch.min(dists, dim=1)
            
            # 概率与距离平方成正比
            probs = min_dists ** 2
            probs /= probs.sum()
            
            # 按概率选择下一个中心
            next_idx = torch.multinomial(probs, 1).item()
            centers[i] = data[next_idx]
            
        return centers

# 示例用法
if __name__ == "__main__":
    # 假设有10个客户端，每个客户端有100个代理数据点，维度为64
    num_clients = 10
    num_samples = 100
    proxy_dim = 64
    num_clusters = 3
    
    # 随机生成代理数据点（实际中应从客户端数据生成）
    proxy_points = torch.randn(num_clients, num_samples, proxy_dim)
    
    # 初始化VWC模型
    vwc_model = VariationalWassersteinClustering(
        num_clients=num_clients, 
        num_clusters=num_clusters, 
        proxy_dim=proxy_dim, 
        pca_dim=4,
        sinkhorn_reg=0.1
    )
    
    # 训练VWC模型
    cluster_assignments = vwc_model.fit(proxy_points)
    print("Cluster Assignments:", cluster_assignments)
