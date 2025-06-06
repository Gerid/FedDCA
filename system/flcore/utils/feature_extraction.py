import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import traceback
from scipy import stats

def generate_proxy_data_gmm(features, num_samples=100, min_components=1, max_components=5):
    """
    使用高斯混合模型(GMM)生成代理数据点
    
    参数:
        features: 原始特征数据，形状为(n_samples, n_features)
        num_samples: 要生成的代理数据点数量
        min_components: GMM的最小组件数
        max_components: GMM的最大组件数
        
    返回:
        生成的代理数据点，形状为(num_samples, n_features)
    """
    try:
        # 确保输入数据格式正确
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
            
        if features.ndim == 1:
            features = features.reshape(-1, 1)
            
        # 获取样本数和特征维度
        n_samples, n_dim = features.shape
        
        # 如果样本数太少，直接使用Bootstrap采样
        if n_samples < 3:
            print(f"样本数({n_samples})过少，使用Bootstrap采样")
            indices = np.random.choice(n_samples, size=num_samples, replace=True)
            return features[indices]
        
        # 如果维度大于样本数，需要先降维
        if n_dim > n_samples:
            print(f"维度({n_dim})大于样本数({n_samples})，使用PCA降维")
            try:
                # 降维到样本数-1维度，但至少保留2维
                target_dim = max(2, min(n_samples - 1, 20))  # 限制最大降维到20维
                pca = PCA(n_components=target_dim)
                features_reduced = pca.fit_transform(features)
                
                # 在降维空间中使用GMM
                gmm_samples = gmm_sample(features_reduced, num_samples, min_components, max_components)
                
                # 投影回原始空间
                samples = pca.inverse_transform(gmm_samples)
                return samples
                
            except Exception as e:
                print(f"PCA降维或GMM失败: {str(e)}")
                print(traceback.format_exc())
                # 失败时使用原始特征的Bootstrap采样
                indices = np.random.choice(n_samples, size=num_samples, replace=True)
                return features[indices]
        
        # 正常情况：直接使用GMM
        return gmm_sample(features, num_samples, min_components, max_components)
        
    except Exception as e:
        print(f"GMM处理完全失败: {str(e)}")
        print(traceback.format_exc())
        
        # 最后的回退：直接返回原始特征或其Bootstrap采样
        if len(features) >= num_samples:
            return features[:num_samples]
        else:
            indices = np.random.choice(len(features), size=num_samples, replace=True)
            return features[indices]

def gmm_sample(features, num_samples, min_components=1, max_components=5):
    """
    使用GMM生成样本
    
    参数:
        features: 特征数据
        num_samples: 要生成的样本数量
        min_components, max_components: GMM组件数范围
    """
    n_samples = features.shape[0]
    
    # 动态确定组件数量，但不超过样本数的一半
    n_components = min(max_components, max(min_components, n_samples // 4))
    
    # 使用BIC准则选择最佳的组件数
    best_gmm = None
    best_bic = np.inf
    
    # 在合理范围内尝试不同的组件数
    for n_comp in range(min_components, min(n_components + 1, n_samples // 2 + 1)):
        try:
            gmm = GaussianMixture(
                n_components=n_comp,
                covariance_type='full',  # 使用完全协方差矩阵以捕获特征间相关性
                random_state=42,
                reg_covar=1e-3  # 添加正则化以增加数值稳定性
            )
            gmm.fit(features)
            bic = gmm.bic(features)
            
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
        except Exception as e:
            print(f"GMM拟合失败(n_comp={n_comp}): {str(e)}")
            continue
    
    # 如果所有组件数都失败，使用简单的单组件GMM
    if best_gmm is None:
        try:
            best_gmm = GaussianMixture(
                n_components=1,
                covariance_type='diag',  # 使用对角协方差矩阵，更简单更稳定
                random_state=42,
                reg_covar=1e-2  # 更强的正则化
            )
            best_gmm.fit(features)
        except Exception as e:
            print(f"单组件GMM拟合失败: {str(e)}")
            # 如果GMM完全失败，回退到Bootstrap
            indices = np.random.choice(n_samples, size=num_samples, replace=True)
            return features[indices]
    
    # 使用拟合好的GMM生成样本
    try:
        samples, _ = best_gmm.sample(num_samples)
        return samples
    except Exception as e:
        print(f"GMM采样失败: {str(e)}")
        # 采样失败时使用Bootstrap
        indices = np.random.choice(n_samples, size=num_samples, replace=True)
        return features[indices]

def generate_proxy_data_kde(features, num_samples=100):
    """
    使用KDE生成代理数据点（作为备选方法）
    
    参数:
        features: 原始特征数据
        num_samples: 要生成的代理数据点数量
        
    返回:
        生成的代理数据点
    """
    try:
        # 确保输入数据格式正确
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
            
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        # 获取样本数和特征维度
        n_samples, n_dim = features.shape
        
        # 如果样本数太少，直接使用Bootstrap采样
        if n_samples < 5:
            indices = np.random.choice(n_samples, size=num_samples, replace=True)
            return features[indices]
        
        # 如果维度大于样本数，需要先降维
        if n_dim > n_samples:
            # 降维到样本数-1维度，但至少保留2维
            target_dim = max(2, min(n_samples - 1, 20))
            pca = PCA(n_components=target_dim)
            features_reduced = pca.fit_transform(features)
            
            # 在降维空间中使用KDE
            kde = stats.gaussian_kde(features_reduced.T)
            sampled_reduced = kde.resample(num_samples).T
            
            # 投影回原始空间
            sampled = pca.inverse_transform(sampled_reduced)
            return sampled
        
        # 正常情况：直接使用KDE
        kde = stats.gaussian_kde(features.T)
        sampled = kde.resample(num_samples).T
        return sampled
        
    except Exception as e:
        print(f"KDE处理失败: {str(e)}")
        # 失败时使用原始特征的Bootstrap采样
        indices = np.random.choice(len(features), size=num_samples, replace=True)
        return features[indices]

def collect_label_conditional_proxy_data(clients, method='gmm', gmm_samples=100, min_components=1, max_components=5):
    """
    收集每个客户端按标签条件分组的代理数据
    
    参数:
        clients: 客户端列表
        method: 代理数据生成方法，'gmm'或'kde'
        gmm_samples: 要生成的样本数量
        min_components, max_components: GMM组件数范围
        
    返回:
        dict: 按客户端ID和标签组织的代理数据字典，格式为 {client_id: {label: features}}
    """
    label_conditional_proxy_data = {}
    
    for client in clients:
        # 从客户端获取按标签分组的特征
        features_by_label = client.get_intermediate_outputs_with_labels()
        
        if not features_by_label:
            print(f"警告: 客户端 {client.id} 没有可用的标签条件特征")
            continue
        
        # 为当前客户端创建标签条件代理数据
        label_conditional_proxy_data[client.id] = {}
        
        # 对每个标签的特征生成代理数据点
        for label, features in features_by_label.items():
            # 确保特征数据格式正确
            if isinstance(features, torch.Tensor):
                features = features.detach().cpu().numpy()
            
            # 跳过空的特征集
            if features.size == 0 or features.shape[0] == 0:
                continue
                  # 生成该标签的代理数据
            try:
                if method == 'gmm':
                    # 使用GMM生成代理数据
                    sampled = generate_proxy_data_gmm(
                        features, 
                        num_samples=gmm_samples, 
                        min_components=min_components, 
                        max_components=max_components
                    )
                elif method == 'kde':
                    # 使用KDE生成代理数据
                    sampled = generate_proxy_data_kde(features, num_samples=gmm_samples)
                else:
                    # 未知方法，使用GMM作为默认选择
                    print(f"未知的代理数据生成方法 '{method}'，使用GMM")
                    sampled = generate_proxy_data_gmm(features, num_samples=gmm_samples)
                
                label_conditional_proxy_data[client.id][label] = sampled
                
            except Exception as e:
                print(f"处理客户端 {client.id} 标签 {label} 时出错: {str(e)}")
                # 出错时使用原始特征
                label_conditional_proxy_data[client.id][label] = features
    
    return label_conditional_proxy_data

def improved_dimension_reduction(features, target_dim=None):
    """
    改进的降维策略，结合多种方法以适应不同场景
    
    参数:
        features: 要降维的特征
        target_dim: 目标维度，如果为None则自动确定
        
    返回:
        降维后的特征
    """
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
        
    n_samples, n_dim = features.shape
    
    if target_dim is None:
        # 自动确定目标维度，但不超过样本数-1
        target_dim = min(max(2, n_samples - 1), 20)  # 至少2维，最多20维
    
    # 如果已经满足维度要求，无需降维
    if n_dim <= target_dim:
        return features
    
    try:
        # 使用PCA进行降维
        pca = PCA(n_components=target_dim)
        features_reduced = pca.fit_transform(features)
        return features_reduced
    except Exception as e:
        print(f"PCA降维失败: {str(e)}")
        
        try:
            # 如果PCA失败，尝试使用随机投影
            from sklearn.random_projection import GaussianRandomProjection
            rp = GaussianRandomProjection(n_components=target_dim, random_state=42)
            features_reduced = rp.fit_transform(features)
            return features_reduced
        except Exception as e:
            print(f"随机投影降维失败: {str(e)}")
            
            # 所有降维方法都失败，只能截断保留前几个维度
            return features[:, :target_dim]
