import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import random
from collections import defaultdict

def load_data_from_npz(file_path):
    """
    从.npz文件加载数据
    
    Args:
        file_path: .npz文件路径
        
    Returns:
        x: 特征数据
        y: 标签数据
    """
    try:
        with open(file_path, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            # 尝试加载不同的键名，我们已经修改了保存格式
            if 'x' in data:
                x = data['x']
                y = data['y']
            elif 'data' in data:
                data_dict = data['data'].item()
                x = data_dict['x']
                y = data_dict['y']
            else:
                raise ValueError(f"未知的数据格式: {list(data.keys())}")
                
            return x, y
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return None, None

def find_duplicate_images(train_images, test_images, threshold=0.0):
    """
    查找训练集和测试集中的重复图像
    
    Args:
        train_images: 训练图像数组，形状为 [N, H, W, C]
        test_images: 测试图像数组，形状为 [M, H, W, C]
        threshold: 像素差异阈值，低于此阈值被认为是重复的
        
    Returns:
        duplicates: 列表，包含重复图像的索引对 (train_idx, test_idx)
    """
    duplicates = []
    
    # 对于大型数据集，这个过程可能很慢，可以考虑使用随机采样
    max_samples = min(len(train_images), 1000)
    if len(train_images) > max_samples:
        train_indices = random.sample(range(len(train_images)), max_samples)
        train_subset = train_images[train_indices]
    else:
        train_subset = train_images
        train_indices = list(range(len(train_images)))
    
    max_test_samples = min(len(test_images), 1000)
    if len(test_images) > max_test_samples:
        test_indices = random.sample(range(len(test_images)), max_test_samples)
        test_subset = test_images[test_indices]
    else:
        test_subset = test_images
        test_indices = list(range(len(test_images)))
    
    print(f"比较 {len(train_subset)} 个训练样本和 {len(test_subset)} 个测试样本")
    
    # 将图像展平为向量，以便进行像素比较
    train_vectors = train_subset.reshape(len(train_subset), -1)
    test_vectors = test_subset.reshape(len(test_subset), -1)
    
    # 对每个测试图像，与所有训练图像比较
    for i, test_vec in enumerate(test_vectors):
        for j, train_vec in enumerate(train_vectors):
            # 计算欧氏距离
            diff = np.mean(np.abs(test_vec - train_vec))
            if diff < threshold:
                duplicates.append((train_indices[j], test_indices[i]))
    
    return duplicates

def test_data_redundancy(data_dir, client_id, iteration_id=0):
    """
    测试指定客户端在指定迭代中的训练集和测试集是否有重复样本
    
    Args:
        data_dir: 数据目录路径
        client_id: 客户端ID
        iteration_id: 迭代ID
        
    Returns:
        duplicate_ratio: 重复样本的比例
    """
    # 构建路径
    iter_path = os.path.join(data_dir, f"iteration_{iteration_id}")
    train_file = os.path.join(iter_path, "train", f"{client_id}.npz")
    test_file = os.path.join(iter_path, "test", f"{client_id}.npz")
    
    # 加载数据
    train_x, train_y = load_data_from_npz(train_file)
    test_x, test_y = load_data_from_npz(test_file)
    
    if train_x is None or test_x is None:
        print(f"无法加载客户端 {client_id} 的数据")
        return None
    
    print(f"客户端 {client_id} 的训练集大小: {len(train_x)}")
    print(f"客户端 {client_id} 的测试集大小: {len(test_x)}")
    
    # 查找重复图像
    # 对于完全相同的副本，阈值可以设为0
    # 对于稍有不同的图像，可以使用较小的阈值，例如5.0
    duplicates = find_duplicate_images(train_x, test_x, threshold=5.0)
    
    duplicate_ratio = len(duplicates) / len(test_x) if len(test_x) > 0 else 0
    print(f"找到 {len(duplicates)} 个重复样本，占测试集的 {duplicate_ratio:.2%}")
    
    # 如果有重复，可视化一些例子
    if duplicates and len(duplicates) > 0:
        n_examples = min(5, len(duplicates))
        fig, axes = plt.subplots(n_examples, 2, figsize=(10, 2*n_examples))
        
        for i in range(n_examples):
            train_idx, test_idx = duplicates[i]
            
            if n_examples == 1:
                axes[0].imshow(train_x[train_idx])
                axes[0].set_title(f"训练样本 {train_idx}")
                axes[0].axis('off')
                
                axes[1].imshow(test_x[test_idx])
                axes[1].set_title(f"测试样本 {test_idx}")
                axes[1].axis('off')
            else:
                axes[i, 0].imshow(train_x[train_idx])
                axes[i, 0].set_title(f"训练样本 {train_idx}")
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(test_x[test_idx])
                axes[i, 1].set_title(f"测试样本 {test_idx}")
                axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"duplicate_samples_client_{client_id}.png")
        plt.close()
    
    return duplicate_ratio

def test_all_clients(data_dir, num_clients=20, iteration_id=0):
    """
    测试所有客户端的数据冗余情况
    
    Args:
        data_dir: 数据目录路径
        num_clients: 客户端数量
        iteration_id: 迭代ID
    """
    results = {}
    
    for client_id in range(num_clients):
        print(f"\n--- 测试客户端 {client_id} ---")
        duplicate_ratio = test_data_redundancy(data_dir, client_id, iteration_id)
        results[client_id] = duplicate_ratio
    
    # 统计总体情况
    valid_ratios = [r for r in results.values() if r is not None]
    if valid_ratios:
        avg_ratio = sum(valid_ratios) / len(valid_ratios)
        max_ratio = max(valid_ratios)
        min_ratio = min(valid_ratios)
        
        print("\n--- 总体数据冗余情况 ---")
        print(f"平均重复比例: {avg_ratio:.2%}")
        print(f"最大重复比例: {max_ratio:.2%} (客户端 {list(results.keys())[list(results.values()).index(max_ratio)]})")
        print(f"最小重复比例: {min_ratio:.2%} (客户端 {list(results.keys())[list(results.values()).index(min_ratio)]})")
    
    return results

if __name__ == "__main__":
    # 测试原始数据生成方法的数据冗余情况
    print("测试原始数据生成方法的数据冗余情况...")
    original_data_dir = "Cifar100_clustered"
    if os.path.exists(original_data_dir):
        original_results = test_all_clients(original_data_dir)
    else:
        print(f"目录 {original_data_dir} 不存在，跳过原始数据测试")
        original_results = {}
    
    # 测试修复后数据生成方法的数据冗余情况
    print("\n测试修复后数据生成方法的数据冗余情况...")
    fixed_data_dir = "Cifar100_clustered_fixed"
    if os.path.exists(fixed_data_dir):
        fixed_results = test_all_clients(fixed_data_dir)
    else:
        print(f"目录 {fixed_data_dir} 不存在，跳过修复数据测试")
        fixed_results = {}
    
    # 比较结果
    if original_results and fixed_results:
        print("\n--- 原始方法与修复方法比较 ---")
        original_avg = sum([r for r in original_results.values() if r is not None]) / len(original_results)
        fixed_avg = sum([r for r in fixed_results.values() if r is not None]) / len(fixed_results)
        
        improvement = (original_avg - fixed_avg) / original_avg if original_avg > 0 else 0
        print(f"原始方法平均重复比例: {original_avg:.2%}")
        print(f"修复方法平均重复比例: {fixed_avg:.2%}")
        print(f"改进率: {improvement:.2%}")
