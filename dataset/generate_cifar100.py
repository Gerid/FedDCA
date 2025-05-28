import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
import json

from utils.dataset_utils import check, separate_data, split_data, save_file


random.seed(1)
np.random.seed(1)
num_clients = 20
num_classes = 100
dir_path = "Cifar100/"


# Allocate data to users
def generate_cifar100(data_dir, client_count, class_count, is_niid, is_balanced, partition_type):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    # Setup directory for train/test data
    config_path = data_dir + "config.json"
    train_path = data_dir + "train/"
    test_path = data_dir + "test/"

    if check(config_path, train_path, test_path, client_count, is_niid, is_balanced, partition_type):
        return
        
    # Get Cifar100 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(
        root=data_dir+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(
        root=data_dir+"rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    X, y, statistic = separate_data((dataset_image, dataset_label), client_count, class_count, 
                                    is_niid, is_balanced, partition_type, class_per_client=20)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, client_count, class_count, 
        statistic, is_niid, is_balanced, partition_type)


def generate_cifar100_with_clusters(data_dir, client_count, class_count, is_niid, is_balanced, partition_type, num_concepts=5, iterations=200, num_drifts=5):
    """
    为联邦学习生成带有概念漂移的数据集，使客户端自然形成聚类结构
    
    参数:
        data_dir: 数据存储路径
        client_count: 客户端数量
        class_count: 类别数量
        is_niid: 是否为非独立同分布
        is_balanced: 是否平衡分配
        partition_type: 分区方式 (dir/pathological)
        num_concepts: 概念数量，每个概念对应一个不同的条件分布
        iterations: 迭代/轮次数量，每轮表示一个时间段
        num_drifts: 在整个迭代过程中发生的漂移次数
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # 获取CIFAR-100数据
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(
        root=data_dir+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(
        root=data_dir+"rawdata", train=False, download=True, transform=transform)

    # 加载所有数据
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        train_images, train_labels = train_data
    for _, test_data in enumerate(testloader, 0):
        test_images, test_labels = test_data

    # 转换为numpy数组
    train_images = train_images.cpu().detach().numpy()
    train_labels = train_labels.cpu().detach().numpy()
    test_images = test_images.cpu().detach().numpy()
    test_labels = test_labels.cpu().detach().numpy()
    
    # 为每个概念创建不同的条件分布
    concept_distributions = {}
    for concept_id in range(num_concepts):
        # 为每个概念选择一部分偏好的类别
        preferred_classes_count = np.random.randint(10, 30)  # 每个概念偏好10-30个类别
        preferred_classes = np.random.choice(class_count, size=preferred_classes_count, replace=False)
        
        # 为这个概念分配类别偏好的权重
        class_weights = np.ones(class_count) * 0.1  # 基础权重
        class_weights[preferred_classes] = np.random.uniform(0.5, 1.0, size=len(preferred_classes))
        # 归一化权重
        class_weights = class_weights / np.sum(class_weights)
        
        concept_distributions[concept_id] = {
            'preferred_classes': preferred_classes.tolist(),
            'class_weights': class_weights.tolist()
        }
    
    # 为每个客户端分配2-3个概念
    client_concepts = {}
    for client_id in range(client_count):
        # 随机选择2-3个概念
        num_concepts_per_client = np.random.randint(2, 4)  # 2-3个概念
        client_concepts[client_id] = np.random.choice(num_concepts, size=num_concepts_per_client, replace=False).tolist()
    
    # 为每个客户端分配漂移类型
    drift_types = ['sudden', 'gradual', 'recurring']
    client_drift_types = {}
    for client_id in range(client_count):
        client_drift_types[client_id] = np.random.choice(drift_types)
    
    # 生成5个固定的漂移时间点，均匀分布在整个迭代过程中
    drift_iterations = np.linspace(1, iterations-10, num_drifts, dtype=int)
    print(f"漂移将发生在以下迭代: {drift_iterations}")
    
    # 为每个客户端生成漂移轨迹（概念切换序列）
    client_concept_trajectories = {}
    for client_id in range(client_count):
        # 获取该客户端的概念列表
        available_concepts = client_concepts[client_id]
        
        # 初始化轨迹，记录每个时间点客户端属于哪个概念
        trajectory = np.zeros(iterations, dtype=int)
        
        # 设置初始概念
        initial_concept_idx = 0  # 从第一个概念开始
        trajectory[0] = available_concepts[initial_concept_idx]
        
        # 根据漂移类型和漂移点生成概念轨迹
        drift_type = client_drift_types[client_id]
        
        if drift_type == 'sudden':
            # 突变漂移：在漂移点突然切换到不同概念
            current_concept_idx = initial_concept_idx
            
            for i in range(1, iterations):
                if i in drift_iterations:
                    # 在漂移点切换到下一个概念
                    current_concept_idx = (current_concept_idx + 1) % len(available_concepts)
                
                trajectory[i] = available_concepts[current_concept_idx]
                
        elif drift_type == 'gradual':
            # 渐进漂移：在漂移点周围逐渐从一个概念过渡到另一个概念
            gradient_window = 10  # 渐变窗口大小
            current_concept_idx = initial_concept_idx
            
            for i in range(1, iterations):
                if i in drift_iterations:
                    # 在漂移点切换到下一个概念
                    current_concept_idx = (current_concept_idx + 1) % len(available_concepts)
                
                # 如果在渐变窗口内，概率性地分配概念
                if any(abs(i - drift_iter) < gradient_window for drift_iter in drift_iterations):
                    # 找出最近的漂移点
                    nearest_drift = min(drift_iterations, key=lambda drift: abs(i - drift))
                    distance = abs(i - nearest_drift)
                    
                    if i < nearest_drift:  # 漂移点之前
                        prev_idx = (current_concept_idx - 1) % len(available_concepts)
                        # 随着接近漂移点，越来越可能使用当前概念
                        prob_current = 1 - (distance / gradient_window)
                        if np.random.random() < prob_current:
                            trajectory[i] = available_concepts[current_concept_idx]
                        else:
                            trajectory[i] = available_concepts[prev_idx]
                    else:  # 漂移点之后
                        next_idx = (current_concept_idx + 1) % len(available_concepts)
                        # 随着远离漂移点，越来越可能使用下一个概念
                        prob_next = 1 - (distance / gradient_window)
                        if np.random.random() < prob_next:
                            trajectory[i] = available_concepts[next_idx]
                        else:
                            trajectory[i] = available_concepts[current_concept_idx]
                else:
                    trajectory[i] = available_concepts[current_concept_idx]
                
        else:  # recurring
            # 周期漂移：在不同的概念之间循环切换
            period_length = np.random.randint(20, 40)  # 周期长度20-40
            
            for i in range(1, iterations):
                if i in drift_iterations:
                    # 重新计算周期，使其在漂移点发生变化
                    period_length = np.random.randint(20, 40)
                
                # 根据周期确定当前概念索引
                current_concept_idx = (initial_concept_idx + (i // period_length)) % len(available_concepts)
                trajectory[i] = available_concepts[current_concept_idx]
        
        client_concept_trajectories[client_id] = trajectory.tolist()
    
    # 保存概念信息和客户端漂移轨迹
    concept_info = {
        'num_concepts': num_concepts,
        'concept_distributions': concept_distributions,
        'client_concepts': client_concepts,
        'client_drift_types': client_drift_types,
        'drift_iterations': drift_iterations.tolist(),
        'client_concept_trajectories': client_concept_trajectories
    }
    
    if not os.path.exists(data_dir + "drift_info/"):
        os.makedirs(data_dir + "drift_info/")
    
    # 将numpy数组转换为列表以便JSON序列化
    concept_info_serializable = json.loads(
        json.dumps(concept_info, cls=NumpyArrayEncoder)
    )
    
    with open(data_dir + "drift_info/concept_config.json", 'w', encoding='utf-8') as f:
        json.dump(concept_info_serializable, f, indent=4)
    
    print(f"每个客户端的概念分配: {client_concepts}")
    print(f"每个客户端的漂移类型: {client_drift_types}")
    
    # 对每个迭代进行数据分配
    for iteration in range(iterations):
        print(f"\n生成迭代 {iteration} 的数据")
        
        # 创建当前迭代的目录
        iteration_dir = os.path.join(data_dir, f"iteration_{iteration}")
        if not os.path.exists(iteration_dir):
            os.makedirs(iteration_dir)
        
        iteration_train_path = os.path.join(iteration_dir, "train/")
        iteration_test_path = os.path.join(iteration_dir, "test/")
        iteration_config_path = os.path.join(iteration_dir, "config.json")
        
        if not os.path.exists(iteration_train_path):
            os.makedirs(iteration_train_path)
        if not os.path.exists(iteration_test_path):
            os.makedirs(iteration_test_path)
        
        # 为当前迭代分配基础数据
        X, y, statistic = separate_data(
            (train_images, train_labels), 
            client_count, 
            class_count, 
            is_niid, 
            is_balanced, 
            partition_type, 
            class_per_client=30  # 增加每个客户端可能拥有的类别数
        )
        
        # 根据每个客户端当前的概念调整数据分布
        for client_id in range(client_count):
            # 获取当前迭代中客户端所属的概念
            current_concept = client_concept_trajectories[client_id][iteration]
            concept_dist = concept_distributions[current_concept]
            
            # 获取该概念的类别权重
            class_weights = np.array(concept_dist['class_weights'])
            
            # 调整客户端数据以反映概念分布
            client_x = X[client_id]
            client_y = y[client_id]
            
            # 计算当前数据集中每个类别的数量
            class_counts = np.zeros(class_count)
            for cls in range(class_count):
                class_counts[cls] = np.sum(client_y == cls)
            
            # 计算目标类别分布（基于类别权重）
            total_samples = len(client_y)
            target_counts = class_weights * total_samples
            
            # 调整每个类别的样本数量
            for cls in range(class_count):
                current_count = class_counts[cls]
                target_count = target_counts[cls]
                
                if current_count > target_count:
                    # 需要减少样本
                    reduction = int(current_count - target_count)
                    if reduction > 0 and current_count > reduction:
                        # 找出该类别的样本索引
                        cls_indices = np.where(client_y == cls)[0]
                        # 随机选择要删除的样本
                        to_remove = np.random.choice(cls_indices, min(reduction, len(cls_indices)), replace=False)
                        # 创建保留掩码
                        keep_mask = np.ones(len(client_y), dtype=bool)
                        keep_mask[to_remove] = False
                        # 更新数据
                        client_x = client_x[keep_mask]
                        client_y = client_y[keep_mask]
                
                elif current_count < target_count:
                    # 需要增加样本
                    addition = int(target_count - current_count)
                    if addition > 0 and current_count > 0:
                        # 找出该类别的样本索引
                        cls_indices = np.where(client_y == cls)[0]
                        # 随机选择要复制的样本
                        to_add = np.random.choice(cls_indices, addition, replace=True)
                        # 添加复制的样本
                        client_x = np.concatenate([client_x, client_x[to_add]])
                        client_y = np.concatenate([client_y, client_y[to_add]])
            
            # 更新客户端数据
            X[client_id] = client_x
            y[client_id] = client_y
            
            # 检查是否是漂移迭代点
            if iteration in drift_iterations:
                print(f"客户端 {client_id} 在迭代 {iteration} 发生 {client_drift_types[client_id]} 漂移，概念从 {client_concept_trajectories[client_id][iteration-1]} 变为 {current_concept}")
        
        # 划分训练和测试数据
        train_data, test_data = {}, {}
        
        for c in range(client_count):
            # 分割数据为训练集和测试集
            if len(X[c]) > 0:
                split_idx = int(0.8 * len(X[c]))
                X_train, X_test = X[c][:split_idx], X[c][split_idx:]
                y_train, y_test = y[c][:split_idx], y[c][split_idx:]
            else:
                X_train, X_test = np.array([]), np.array([])
                y_train, y_test = np.array([]), np.array([])
            
            train_data[c] = {'x': X_train, 'y': y_train}
            test_data[c] = {'x': X_test, 'y': y_test}
        
        # 调试日志，检查 train_data 和 test_data 的结构
        print(f"调试: 迭代 {iteration} 的 train_data 结构: {train_data}")
        print(f"调试: 迭代 {iteration} 的 test_data 结构: {test_data}")
        
        # 保存此迭代的数据
        save_file(
            iteration_config_path, 
            iteration_train_path, 
            iteration_test_path, 
            train_data, 
            test_data, 
            client_count, 
            class_count, 
            statistic, 
            is_niid, 
            is_balanced, 
            partition_type
        )
        
        print(f"迭代 {iteration} 的数据生成完成")
    
    print("\n所有迭代的数据生成完成，概念漂移已应用到数据集中")
    print(f"共有 {num_concepts} 个概念，每个客户端涉及2-3个概念")
    print(f"在 {iterations} 轮迭代中，在以下时间点发生了 {num_drifts} 次漂移: {drift_iterations}")
    
    # 额外添加客户端-概念分配的描述性统计
    concept_client_map = {i: [] for i in range(num_concepts)}
    for client_id, concepts in client_concepts.items():
        for concept in concepts:
            concept_client_map[concept].append(client_id)
    
    print("\n各概念包含的客户端:")
    for concept, clients in concept_client_map.items():
        print(f"概念 {concept}: {clients}")
    
    # 保存最终聚类结果，用于后续分析
    final_iteration = iterations - 1
    final_client_concepts = {}
    for client_id in range(client_count):
        final_client_concepts[str(client_id)] = int(client_concept_trajectories[client_id][final_iteration])
    
    with open(data_dir + "drift_info/final_client_concepts.json", 'w', encoding='utf-8') as f:
        json.dump(final_client_concepts, f, indent=4)
    
    return concept_info

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return json.JSONEncoder.default(self, o)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        niid = True if sys.argv[1] == "noniid" else False
    else:
        niid = False

    if len(sys.argv) > 2:
        balance = False if sys.argv[2] == "balance" else False
    else:
        balance = False

    partition = "dir"


    generate_cifar100(
                data_dir="Cifar100/",  # 数据存储路径
                client_count=20,                 # 客户端数量
                class_count=100,                 # 类别数量
                is_niid=True,                    # 非独立同分布设置
                is_balanced=False,               # 非平衡分配
                partition_type="dir")          
    #         )
    #     else:
    #         cluster_info = generate_cifar100_with_clusters(
    #             data_dir="Cifar100_clustered/",  # 数据存储路径
    #             client_count=20,                 # 客户端数量
    #             class_count=100,                 # 类别数量
    #             is_niid=True,                    # 非独立同分布设置
    #             is_balanced=False,               # 非平衡分配
    #             partition_type="dir",            # 分区方式
    #             num_concepts=5,                  # 概念数量
    #             iterations=200,                  # 时间迭代数量
    #             num_drifts=5                     # 漂移次数
    #         )
    # else:
    #     print("need dataset arg")