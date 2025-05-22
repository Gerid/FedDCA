"""
服务器端概念漂移管理模块

该模块负责:
1. 为所有客户端创建和分配共享概念
2. 保存概念配置到文件以确保可重复性
3. 管理概念漂移点和漂移类型
"""
import os
import json
import copy
import numpy as np
import torch
from torch.utils.data import TensorDataset
import pickle


def initialize_shared_concepts(server):
    """
    初始化服务器端共享的概念漂移组合

    创建一组共享的概念，并为每个客户端分配概念，
    确保所有客户端使用相同的概念集合，以便进行公平的聚类分析
    
    Args:
        server: 服务器实例
    
    Returns:
        dict: 包含所有概念配置的字典
    """
    from utils.concept_drift_simulation import create_shared_concepts, initialize_drift_patterns
    
    # 首先检查是否存在保存的概念配置文件
    config_file = os.path.join('results', 'drift_concepts', 'concept_config.json')
    
    # 如果配置目录不存在，创建它
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    
    # 尝试读取现有配置
    if os.path.exists(config_file):
        print(f"加载现有概念配置: {config_file}")
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            # 从配置中提取主要信息
            shared_concepts = config.get('shared_concepts', [])
            if not shared_concepts:
                raise ValueError("配置文件缺少shared_concepts字段")
                
            print(f"读取到 {len(shared_concepts)} 个共享概念")
        except Exception as e:
            print(f"读取配置文件失败: {str(e)}，将创建新的配置")
            shared_concepts = None
    else:
        shared_concepts = None
    
    # 如果没有现有配置或读取失败，创建新的配置
    if shared_concepts is None:
        print("创建新的共享概念配置")
        # 创建共享概念
        shared_concepts = create_shared_concepts(num_concepts=5, num_classes=100, seed=42)
        shared_drift_patterns = initialize_drift_patterns()
        
        # 设置漂移点 - 在指定轮次训练中均匀分布5个漂移点
        iterations = getattr(server, 'max_iterations', 200)
        num_drifts = 5
        drift_points = [iterations * (i + 1) // (num_drifts + 1) for i in range(num_drifts)]
        
        # 为每个客户端分配概念和漂移类型
        client_concepts = {}
        client_drift_types = {}
        
        for client in server.clients:
            drift_types = ['sudden', 'gradual', 'recurring']
            client_id_hash = hash(f"client_{client.id}") % 3
            drift_type = drift_types[client_id_hash]
            client_drift_types[str(client.id)] = drift_type
            
            # 为每个客户端分配概念
            num_client_concepts = np.random.randint(2, 4)  # 分配2-3个概念
            np.random.seed(hash(f"client_{client.id}_concepts") % 10000)
            concept_indices = np.random.choice(len(shared_concepts), num_client_concepts, replace=False).tolist()
            client_concepts[str(client.id)] = concept_indices
            
        # 创建配置对象
        config = {
            'shared_concepts': shared_concepts,
            'drift_patterns': shared_drift_patterns,
            'drift_iterations': drift_points,
            'client_concepts': client_concepts,
            'client_drift_types': client_drift_types,
            'num_concepts': len(shared_concepts),
            'max_iterations': iterations
        }
        
        # 保存配置到文件
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
        print(f"共享概念配置已保存到: {config_file}")
        print(f"- 共 {len(shared_concepts)} 个概念")
        print(f"- 漂移点: {drift_points}")
    
    # 将配置传递给所有客户端
    distribute_concepts_to_clients(server, shared_concepts, config)
    
    return config

def distribute_concepts_to_clients(server, shared_concepts, config):
    """
    将共享概念分配给所有客户端
    
    Args:
        server: 服务器实例
        shared_concepts: 共享概念列表
        config: 完整的配置字典
    """
    client_concepts = config['client_concepts']
    client_drift_types = config['client_drift_types']
    drift_iterations = config['drift_iterations']
    
    print(f"为 {len(server.clients)} 个客户端分配共享概念配置...")
    
    for client in server.clients:
        client_id = str(client.id)
        
        # 设置客户端的概念
        if client_id in client_concepts:
            concept_indices = client_concepts[client_id]
            client_specific_concepts = [shared_concepts[idx] for idx in concept_indices]
            client.shared_concepts = client_specific_concepts
        else:
            print(f"警告: 客户端 {client_id} 没有预定义的概念分配")
            # 随机分配概念作为后备方案
            num_concepts = np.random.randint(2, 4)
            concept_indices = np.random.choice(len(shared_concepts), num_concepts, replace=False)
            client_specific_concepts = [shared_concepts[idx] for idx in concept_indices]
            client.shared_concepts = client_specific_concepts
        
        # 设置客户端的漂移类型
        if client_id in client_drift_types:
            client.drift_type = client_drift_types[client_id]
        else:
            drift_types = ['sudden', 'gradual', 'recurring']
            client.drift_type = drift_types[hash(f"client_{client.id}") % 3]
        
        # 设置其他漂移相关属性
        client.drift_points = drift_iterations
        client.use_shared_concepts = True
        client.gradual_window = 10
        client.recurring_period = np.random.randint(20, 41)
        
        # 启用模拟漂移
        client.simulate_drift = True
        
        # 如果有漂移数据目录，也设置它
        if hasattr(server, 'drift_data_dir'):
            client.drift_data_dir = server.drift_data_dir

def save_concept_progress(server, round_idx=None):
    """
    保存当前训练轮次的概念漂移进展状态
    
    Args:
        server: 服务器实例
        round_idx: 当前训练轮次
    """
    if not hasattr(server, 'current_iteration'):
        return
    
    progress_file = os.path.join('results', 'drift_concepts', 'concept_progress.json')
    
    # 创建或更新进度对象
    progress = {
        'current_iteration': server.current_iteration,
        'round': round_idx,
        'client_concepts': {}
    }
    
    # 记录每个客户端当前使用的概念
    for client in server.selected_clients:
        if hasattr(client, 'current_concept_id'):
            progress['client_concepts'][str(client.id)] = client.current_concept_id
    
    # 保存进度到文件
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)

def get_client_concept_mapping(server):
    """
    获取客户端到概念的映射，用于分析
    
    Args:
        server: 服务器实例
    
    Returns:
        dict: 客户端到当前概念的映射
    """
    mapping = {}
    
    for client in server.clients:
        if hasattr(client, 'current_concept_id'):
            mapping[str(client.id)] = client.current_concept_id
    
    return mapping

def analyze_concept_alignment(server):
    """
    分析聚类分配与概念的一致性
    
    Args:
        server: 服务器实例
    
    Returns:
        dict: 包含一致性分析的字典
    """
    if not hasattr(server, 'clusters'):
        return None
    
    # 获取客户端到概念的映射
    concept_mapping = get_client_concept_mapping(server)
    
    # 如果没有足够的概念信息，返回None
    if not concept_mapping:
        return None
    
    # 计算集群与概念的对应关系
    cluster_concept_matrix = {}
    for client_id, cluster_id in server.clusters.items():
        if client_id in concept_mapping:
            concept_id = concept_mapping[client_id]
            
            if cluster_id not in cluster_concept_matrix:
                cluster_concept_matrix[cluster_id] = {}
            
            if concept_id not in cluster_concept_matrix[cluster_id]:
                cluster_concept_matrix[cluster_id][concept_id] = 0
                
            cluster_concept_matrix[cluster_id][concept_id] += 1
    
    # 计算每个集群中主导概念
    dominant_concepts = {}
    for cluster_id, concepts in cluster_concept_matrix.items():
        dominant_concept = max(concepts.items(), key=lambda x: x[1])
        dominant_concepts[cluster_id] = {
            'concept_id': dominant_concept[0],
            'count': dominant_concept[1],
            'total': sum(concepts.values()),
            'percentage': dominant_concept[1] / sum(concepts.values()) * 100
        }
    
    # 计算总体对齐率
    total_aligned = sum(info['count'] for info in dominant_concepts.values())
    total_clients = sum(info['total'] for info in dominant_concepts.values())
    overall_alignment = total_aligned / total_clients if total_clients > 0 else 0
    
    return {
        'cluster_concept_matrix': cluster_concept_matrix,
        'dominant_concepts': dominant_concepts,
        'overall_alignment': overall_alignment
    }
