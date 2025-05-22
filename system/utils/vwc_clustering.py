import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

class VariationalWassersteinClustering(nn.Module):
    def __init__(self, num_clients, num_clusters, proxy_dim, pca_dim=4, sinkhorn_reg=0.01):
        super().__init__()
        self.num_clients = num_clients
        self.num_clusters = num_clusters
        self.pca_dim = pca_dim
        self.sinkhorn_reg = sinkhorn_reg
        
        # 随机初始化集群中心，更好的初始值
        self.centers = nn.Parameter(torch.randn(num_clusters, pca_dim) * 0.1)
        
        # 为logits添加一些随机性，避免初始时所有客户端分配相同
        self.logits = nn.Parameter(torch.randn(num_clients, num_clusters) * 0.01)

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
        neg_dist = -dist
        
        # 使用负距离直接影响分配概率
        assignment_logits = self.logits + neg_dist
        
        # softmax分配概率
        probs = F.softmax(assignment_logits, dim=1)
        
        # 计算损失：距离损失 + 熵正则化
        loss = (probs * dist).sum() - self.sinkhorn_reg * (probs * torch.log(probs+1e-10)).sum()
        
        return probs, loss

    def get_cluster_assignments(self):
        with torch.no_grad():
            # 使用与forward中相同的逻辑计算分配
            dummy_data = torch.zeros(self.num_clients, 1, self.pca_dim, device=self.centers.device)
            probs, _ = self.forward(dummy_data)
            # 打印每个集群的分配数量以便调试
            assignments = torch.argmax(probs, dim=1)
            cluster_counts = torch.bincount(assignments, minlength=self.num_clusters)
            print(f"Cluster distribution: {cluster_counts.tolist()}")
            return assignments

    def fit(self, proxy_points, lr=0.01, max_iter=50, tol=1e-4):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        prev_loss = float('inf')
        
        # 保存训练过程中的损失值，用于调试
        losses = []
        
        for i in range(max_iter):
            optimizer.zero_grad()
            probs, loss = self.forward(proxy_points)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if i % 10 == 0:
                # 打印当前集群分配情况以监控训练进度
                with torch.no_grad():
                    assignments = torch.argmax(probs, dim=1)
                    cluster_counts = torch.bincount(assignments, minlength=self.num_clusters)
                    print(f"Iter {i}, Loss: {loss.item():.4f}, Cluster counts: {cluster_counts.tolist()}")
            
            # 更严格的早停条件
            if prev_loss - loss.item() < tol and i > 10:
                print(f"Early stopping at iteration {i}")
                break
                
            prev_loss = loss.item()
        
        # 训练完成后检查集群分配情况
        assignments = self.get_cluster_assignments()
        
        # 如果所有样本都被分到一个集群，尝试强制重新分配
        if torch.unique(assignments).size(0) == 1:
            print("Warning: All clients assigned to the same cluster! Attempting forced reassignment...")
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
            
            # 根据到新中心的距离重置logits
            dist = torch.cdist(x_mean, self.centers)
            new_logits = -dist + torch.randn_like(self.logits) * 0.1  # 添加随机扰动
            self.logits.data.copy_(new_logits)
    
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
        sinkhorn_reg=0.01
    )
    
    # 训练VWC模型
    cluster_assignments = vwc_model.fit(proxy_points)
    print("Cluster Assignments:", cluster_assignments)