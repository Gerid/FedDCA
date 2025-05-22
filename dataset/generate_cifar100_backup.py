import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms

from utils.dataset_utils import check, separate_data, split_data, save_file


random.seed(1)
np.random.seed(1)
num_clients = 20
num_classes = 100
dir_path = "Cifar100/"


# Allocate data to users
def generate_cifar100(dir_path, num_clients, num_classes, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return
        
    # Get Cifar100 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)
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

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition, class_per_client=20)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)


def generate_cifar100_with_drift(dir_path, num_clients, num_classes, niid, balance, partition, iterations=5):
    """
    为联邦学习中的概念漂移场景生成CIFAR-100数据集
    
    参数:
        dir_path: 数据存储路径
        num_clients: 客户端数量
        num_classes: 类别数量
        niid: 是否为非独立同分布
        balance: 是否平衡分配
        partition: 分区方式 (dir/pathological)
        iterations: 迭代/轮次数量，每轮表示一个时间段
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # 设置基本配置
    base_config_path = dir_path + "config.json"
    
    # 获取CIFAR-100数据
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)

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
    
    # 为每个客户端生成漂移模式
    # 1. 决定漂移类型（逐步漂移、突变漂移、循环漂移）
    drift_types = ['gradual', 'sudden', 'recurring']
    client_drift_types = np.random.choice(drift_types, size=num_clients)
    
    # 2. 为每个客户端决定漂移发生的时间点
    change_points = {}
    for c in range(num_clients):
        if client_drift_types[c] == 'gradual':
            # 逐步漂移：多个时间点
            num_changes = np.random.randint(2, iterations//2)
            change_points[c] = sorted(np.random.choice(range(1, iterations), size=num_changes, replace=False))
        elif client_drift_types[c] == 'sudden':
            # 突变漂移：一个时间点
            change_points[c] = [np.random.randint(1, iterations)]
        else:  # recurring
            # 循环漂移：重复出现的多个时间点
            first_change = np.random.randint(1, iterations//3)
            change_points[c] = [first_change, first_change + iterations//2]
    
    print(f"生成的客户端漂移类型: {client_drift_types}")
    print(f"生成的漂移时间点: {change_points}")
    
    # 保存漂移信息
    drift_info = {
        'client_drift_types': client_drift_types.tolist(),
        'change_points': change_points
    }
    
    if not os.path.exists(dir_path + "drift_info/"):
        os.makedirs(dir_path + "drift_info/")
    
    import json
    with open(dir_path + "drift_info/drift_config.json", 'w') as f:
        json.dump(drift_info, f)
    
    # 对每个迭代进行数据分配
    for iteration in range(iterations):
        print(f"\n生成迭代 {iteration} 的数据")
        
        # 创建当前迭代的目录
        iteration_dir = os.path.join(dir_path, f"iteration_{iteration}")
        if not os.path.exists(iteration_dir):
            os.makedirs(iteration_dir)
        
        iteration_train_path = os.path.join(iteration_dir, "train/")
        iteration_test_path = os.path.join(iteration_dir, "test/")
        iteration_config_path = os.path.join(iteration_dir, "config.json")
        
        if not os.path.exists(iteration_train_path):
            os.makedirs(iteration_train_path)
        if not os.path.exists(iteration_test_path):
            os.makedirs(iteration_test_path)
        
        # 生成此迭代的训练和测试数据
        current_train_images = train_images.copy()
        current_train_labels = train_labels.copy()
        
        # 应用概念漂移
        for c in range(num_clients):
            # 检查是否是当前客户端的漂移时间点
            if iteration in change_points.get(c, []):
                print(f"客户端 {c} 在迭代 {iteration} 发生概念漂移")
                
                # 基于漂移类型执行不同的数据变换
                if client_drift_types[c] == 'gradual':
                    # 逐步漂移：逐步改变类别分布
                    intensity = change_points[c].index(iteration) / len(change_points[c])
                    # 实现逐步漂移的数据变换逻辑
                elif client_drift_types[c] == 'sudden':
                    # 突变漂移：显著改变类别分布
                    # 实现突变漂移的数据变换逻辑
                else:  # recurring
                    # 循环漂移：在不同时间点切换回先前的分布
                    # 实现循环漂移的数据变换逻辑
        
        # 为当前迭代分配数据
        X, y, statistic = separate_data(
            (current_train_images, current_train_labels), 
            num_clients, 
            num_classes, 
            niid, 
            balance, 
            partition, 
            class_per_client=20
        )
        
        # 对于已经发生漂移的客户端，根据漂移类型修改其数据分布
        for c in range(num_clients):
            if any(cp <= iteration for cp in change_points.get(c, [])):
                # 根据漂移类型应用特定的数据转换
                drift_type = client_drift_types[c]
                if drift_type == 'gradual':
                    # 模拟渐进式类别偏好变化
                    intensity = sum(cp <= iteration for cp in change_points[c]) / len(change_points[c])
                    # 随机选择一部分类别进行偏好调整
                    num_classes_to_adjust = int(num_classes * 0.2)  # 调整20%的类别
                    classes_to_adjust = np.random.choice(num_classes, num_classes_to_adjust, replace=False)
                    
                    for cls in classes_to_adjust:
                        cls_indices = np.where(y[c] == cls)[0]
                        if len(cls_indices) > 0:
                            # 根据强度减少或增加某类样本
                            adjust_count = int(len(cls_indices) * intensity * 0.5)
                            if adjust_count > 0:
                                # 随机决定增加还是减少
                                if np.random.random() > 0.5:
                                    # 增加样本（通过复制现有样本）
                                    sample_indices = np.random.choice(cls_indices, adjust_count)
                                    X[c] = np.concatenate([X[c], X[c][sample_indices]])
                                    y[c] = np.concatenate([y[c], y[c][sample_indices]])
                                else:
                                    # 减少样本
                                    remove_indices = np.random.choice(cls_indices, min(adjust_count, len(cls_indices)-1), replace=False)
                                    keep_mask = np.ones(len(X[c]), dtype=bool)
                                    keep_mask[remove_indices] = False
                                    X[c] = X[c][keep_mask]
                                    y[c] = y[c][keep_mask]
                
                elif drift_type == 'sudden':
                    # 模拟突然的类别分布变化
                    # 随机选择一部分类别被其他类别替代
                    num_classes_to_replace = int(num_classes * 0.3)  # 替换30%的类别
                    classes_to_replace = np.random.choice(num_classes, num_classes_to_replace, replace=False)
                    
                    for cls in classes_to_replace:
                        cls_indices = np.where(y[c] == cls)[0]
                        if len(cls_indices) > 0:
                            # 随机选择一个新类别替换
                            new_cls = np.random.choice([i for i in range(num_classes) if i != cls])
                            y[c][cls_indices] = new_cls
                
                elif drift_type == 'recurring':
                    # 在不同时间点循环切换类别分布
                    cycle_phase = sum(cp <= iteration for cp in change_points[c]) % 2
                    
                    if cycle_phase == 1:
                        # 交换两组类别的样本比例
                        group1 = np.random.choice(num_classes, num_classes//2, replace=False)
                        group2 = np.array([i for i in range(num_classes) if i not in group1])
                        
                        # 找出属于两组的样本索引
                        group1_indices = np.where(np.isin(y[c], group1))[0]
                        group2_indices = np.where(np.isin(y[c], group2))[0]
                        
                        # 随机交换一部分样本的类别
                        if len(group1_indices) > 0 and len(group2_indices) > 0:
                            swap_count = min(len(group1_indices), len(group2_indices)) // 2
                            g1_to_swap = np.random.choice(group1_indices, swap_count, replace=False)
                            g2_to_swap = np.random.choice(group2_indices, swap_count, replace=False)
                            
                            # 交换样本
                            temp_X = X[c][g1_to_swap].copy()
                            temp_y = y[c][g1_to_swap].copy()
                            
                            X[c][g1_to_swap] = X[c][g2_to_swap]
                            y[c][g1_to_swap] = y[c][g2_to_swap]
                            
                            X[c][g2_to_swap] = temp_X
                            y[c][g2_to_swap] = temp_y
        
        # 生成此迭代的测试数据
        train_data, test_data = {}, {}
        
        # 为每个客户端准备训练和测试数据
        for c in range(num_clients):
            # 分割数据为训练集和测试集
            X_train, X_test = X[c][:int(0.8*len(X[c]))], X[c][int(0.8*len(X[c])):]
            y_train, y_test = y[c][:int(0.8*len(y[c]))], y[c][int(0.8*len(y[c])):]
            
            train_data[c] = {'x': X_train, 'y': y_train}
            test_data[c] = {'x': X_test, 'y': y_test}
        
        # 保存此迭代的数据
        save_file(
            iteration_config_path, 
            iteration_train_path, 
            iteration_test_path, 
            train_data, 
            test_data, 
            num_clients, 
            num_classes, 
            statistic, 
            niid, 
            balance, 
            partition
        )
        
        print(f"迭代 {iteration} 的数据生成完成")
    
    print("\n所有迭代的数据生成完成，概念漂移已应用到数据集中")
    
    # 生成一个无漂移的基准数据集
    print("\n生成无漂移的基准数据集...")
    generate_cifar100(dir_path + "no_drift/", num_clients, num_classes, niid, balance, partition)
    
    return drift_info


def generate_cifar100_with_clusters(dir_path, num_clients, num_classes, niid, balance, partition, num_clusters=3, iterations=5):
    """
    为联邦学习生成带有概念漂移的数据集，使客户端自然形成聚类结构
    
    参数:
        dir_path: 数据存储路径
        num_clients: 客户端数量
        num_classes: 类别数量
        niid: 是否为非独立同分布
        balance: 是否平衡分配
        partition: 分区方式 (dir/pathological)
        num_clusters: 希望形成的聚类数量
        iterations: 迭代/轮次数量，每轮表示一个时间段
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # 获取CIFAR-100数据
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)

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
    
    # 将客户端分配到聚类
    clients_per_cluster = num_clients // num_clusters
    remaining_clients = num_clients % num_clusters
    
    # 按聚类分配客户端
    client_clusters = {}
    client_id = 0
    
    for cluster_id in range(num_clusters):
        # 计算当前聚类的客户端数量
        cluster_size = clients_per_cluster + (1 if cluster_id < remaining_clients else 0)
        for _ in range(cluster_size):
            client_clusters[client_id] = cluster_id
            client_id += 1
    
    # 为每个聚类分配不同的漂移模式
    drift_patterns = {}
    
    for cluster_id in range(num_clusters):
        # 为每个聚类选择一种主要漂移类型
        if cluster_id % 3 == 0:
            drift_type = 'gradual'
            # 逐步漂移：多个渐进时间点
            num_changes = np.random.randint(2, iterations//2)
            change_points = sorted(np.random.choice(range(1, iterations), size=num_changes, replace=False))
            
            # 选择该聚类偏好的类别子集
            preferred_classes = np.random.choice(num_classes, size=num_classes//3, replace=False)
            
            drift_patterns[cluster_id] = {
                'type': drift_type,
                'change_points': change_points,
                'preferred_classes': preferred_classes,
                'intensity_factor': np.random.uniform(0.5, 1.5)  # 漂移强度因子
            }
            
        elif cluster_id % 3 == 1:
            drift_type = 'sudden'
            # 突变漂移：一个显著变化点
            change_point = np.random.randint(1, iterations-1)
            
            # 选择该聚类在变化后偏好的类别子集
            before_classes = np.random.choice(num_classes, size=num_classes//4, replace=False)
            after_classes = np.random.choice([c for c in range(num_classes) if c not in before_classes], 
                                            size=num_classes//4, replace=False)
            
            drift_patterns[cluster_id] = {
                'type': drift_type,
                'change_point': change_point,
                'before_classes': before_classes,
                'after_classes': after_classes,
                'swap_ratio': np.random.uniform(0.6, 0.9)  # 类别交换比例
            }
            
        else:
            drift_type = 'recurring'
            # 循环漂移：在两个分布之间交替
            first_change = np.random.randint(1, iterations//3)
            period = np.random.randint(1, 3)  # 周期长度
            
            # 生成两组交替的类别偏好
            group1 = np.random.choice(num_classes, size=num_classes//3, replace=False)
            group2 = np.random.choice([c for c in range(num_classes) if c not in group1], 
                                      size=num_classes//3, replace=False)
            
            drift_patterns[cluster_id] = {
                'type': drift_type,
                'first_change': first_change,
                'period': period,
                'group1': group1,
                'group2': group2,
                'swap_intensity': np.random.uniform(0.4, 0.8)  # 交换强度
            }
    
    # 为每个客户端生成漂移信息，加入少量随机性以保持聚类内的相似性和聚类间的差异性
    client_drift_info = {}
    
    for client_id, cluster_id in client_clusters.items():
        pattern = drift_patterns[cluster_id].copy()
        
        # 添加少量随机扰动，确保同一聚类内客户端相似但不完全相同
        if pattern['type'] == 'gradual':
            # 小幅调整变化点和强度因子
            change_points = pattern['change_points'].copy()
            if len(change_points) > 1 and np.random.random() < 0.3:
                # 30%概率调整一个变化点
                idx = np.random.randint(len(change_points))
                change_points[idx] = max(1, min(iterations-1, 
                                              change_points[idx] + np.random.randint(-1, 2)))
                change_points = sorted(change_points)
            
            intensity = pattern['intensity_factor'] * np.random.uniform(0.9, 1.1)
            
            client_drift_info[client_id] = {
                'type': pattern['type'],
                'change_points': change_points,
                'preferred_classes': pattern['preferred_classes'],
                'intensity_factor': intensity
            }
            
        elif pattern['type'] == 'sudden':
            # 小幅调整变化点和交换比例
            change_point = pattern['change_point'] + np.random.randint(-1, 2)
            change_point = max(1, min(iterations-1, change_point))
            
            swap_ratio = pattern['swap_ratio'] * np.random.uniform(0.9, 1.1)
            swap_ratio = min(1.0, swap_ratio)
            
            client_drift_info[client_id] = {
                'type': pattern['type'],
                'change_point': change_point,
                'before_classes': pattern['before_classes'],
                'after_classes': pattern['after_classes'],
                'swap_ratio': swap_ratio
            }
            
        else:  # recurring
            # 小幅调整首次变化点和周期
            first_change = pattern['first_change'] + np.random.randint(-1, 2)
            first_change = max(1, min(iterations-1, first_change))
            
            period = pattern['period']  # 保持周期一致以维持聚类相似性
            swap_intensity = pattern['swap_intensity'] * np.random.uniform(0.9, 1.1)
            swap_intensity = min(1.0, swap_intensity)
            
            client_drift_info[client_id] = {
                'type': pattern['type'],
                'first_change': first_change,
                'period': period,
                'group1': pattern['group1'],
                'group2': pattern['group2'],
                'swap_intensity': swap_intensity
            }
    
    # 保存聚类和漂移信息
    cluster_info = {
        'num_clusters': num_clusters,
        'client_clusters': client_clusters,
        'drift_patterns': drift_patterns,
        'client_drift_info': client_drift_info
    }
    
    if not os.path.exists(dir_path + "drift_info/"):
        os.makedirs(dir_path + "drift_info/")
    
    import json
    # 将numpy数组转换为列表以便JSON序列化
    cluster_info_serializable = json.loads(
        json.dumps(cluster_info, cls=NumpyArrayEncoder)
    )
    
    with open(dir_path + "drift_info/cluster_config.json", 'w') as f:
        json.dump(cluster_info_serializable, f, indent=4)
    
    print(f"客户端聚类分配: {client_clusters}")
    
    # 对每个迭代进行数据分配
    for iteration in range(iterations):
        print(f"\n生成迭代 {iteration} 的数据")
        
        # 创建当前迭代的目录
        iteration_dir = os.path.join(dir_path, f"iteration_{iteration}")
        if not os.path.exists(iteration_dir):
            os.makedirs(iteration_dir)
        
        iteration_train_path = os.path.join(iteration_dir, "train/")
        iteration_test_path = os.path.join(iteration_dir, "test/")
        iteration_config_path = os.path.join(iteration_dir, "config.json")
        
        if not os.path.exists(iteration_train_path):
            os.makedirs(iteration_train_path)
        if not os.path.exists(iteration_test_path):
            os.makedirs(iteration_test_path)
        
        # 为当前迭代分配数据
        X, y, statistic = separate_data(
            (train_images, train_labels), 
            num_clients, 
            num_classes, 
            niid, 
            balance, 
            partition, 
            class_per_client=20
        )
        
        # 对每个客户端应用漂移
        for client_id, drift_info in client_drift_info.items():
            drift_type = drift_info['type']
            
            # 根据漂移类型和当前迭代决定是否应用漂移
            apply_drift = False
            
            if drift_type == 'gradual':
                change_points = drift_info['change_points']
                # 检查当前迭代是否是变化点
                if iteration in change_points:
                    apply_drift = True
                    # 计算当前强度（基于当前是第几个变化点）
                    current_intensity = (change_points.index(iteration) + 1) / len(change_points)
                    # 应用强度因子
                    current_intensity *= drift_info['intensity_factor']
                    
                    # 应用渐进漂移
                    preferred_classes = drift_info['preferred_classes']
                    
                    # 增加首选类别样本，减少非首选类别样本
                    for cls in range(num_classes):
                        cls_indices = np.where(y[client_id] == cls)[0]
                        if len(cls_indices) > 0:
                            if cls in preferred_classes:
                                # 增加首选类别样本
                                boost_count = int(len(cls_indices) * current_intensity * 0.5)
                                if boost_count > 0:
                                    # 复制现有样本
                                    sample_indices = np.random.choice(cls_indices, boost_count)
                                    X[client_id] = np.concatenate([X[client_id], X[client_id][sample_indices]])
                                    y[client_id] = np.concatenate([y[client_id], y[client_id][sample_indices]])
                            else:
                                # 减少非首选类别样本
                                reduce_count = int(len(cls_indices) * current_intensity * 0.3)
                                if reduce_count > 0 and len(cls_indices) > reduce_count:
                                    # 移除部分样本
                                    remove_indices = np.random.choice(cls_indices, reduce_count, replace=False)
                                    keep_mask = np.ones(len(X[client_id]), dtype=bool)
                                    keep_mask[remove_indices] = False
                                    X[client_id] = X[client_id][keep_mask]
                                    y[client_id] = y[client_id][keep_mask]
                                    
            elif drift_type == 'sudden':
                # 检查是否达到突变点
                if iteration == drift_info['change_point']:
                    apply_drift = True
                    
                    # 应用突变漂移
                    before_classes = drift_info['before_classes']
                    after_classes = drift_info['after_classes']
                    swap_ratio = drift_info['swap_ratio']
                    
                    # 将一定比例的before_classes替换为after_classes
                    for i, before_cls in enumerate(before_classes):
                        if i < len(after_classes):  # 确保有对应的after类别
                            after_cls = after_classes[i]
                            before_indices = np.where(y[client_id] == before_cls)[0]
                            
                            if len(before_indices) > 0:
                                # 确定要交换的样本数量
                                swap_count = int(len(before_indices) * swap_ratio)
                                if swap_count > 0:
                                    swap_indices = np.random.choice(before_indices, swap_count, replace=False)
                                    # 将标签从before_cls更改为after_cls
                                    y[client_id][swap_indices] = after_cls
                    
            elif drift_type == 'recurring':
                # 检查是否应该切换状态
                first_change = drift_info['first_change']
                period = drift_info['period']
                
                if iteration >= first_change:
                    # 计算当前周期中的位置
                    cycle_position = (iteration - first_change) % (period * 2)
                    # 如果在周期的前半部分，使用group1；在后半部分，使用group2
                    current_state = 0 if cycle_position < period else 1
                    
                    # 只在状态切换点应用漂移
                    if (iteration - first_change) % period == 0:
                        apply_drift = True
                        
                        group1 = drift_info['group1']
                        group2 = drift_info['group2']
                        swap_intensity = drift_info['swap_intensity']
                        
                        # 当前活跃组和非活跃组
                        active_group = group1 if current_state == 0 else group2
                        inactive_group = group2 if current_state == 0 else group1
                        
                        # 增加活跃组类别样本
                        for cls in active_group:
                            cls_indices = np.where(y[client_id] == cls)[0]
                            if len(cls_indices) > 0:
                                boost_count = int(len(cls_indices) * swap_intensity * 0.5)
                                if boost_count > 0:
                                    sample_indices = np.random.choice(cls_indices, boost_count)
                                    X[client_id] = np.concatenate([X[client_id], X[client_id][sample_indices]])
                                    y[client_id] = np.concatenate([y[client_id], y[client_id][sample_indices]])
                        
                        # 减少非活跃组类别样本
                        for cls in inactive_group:
                            cls_indices = np.where(y[client_id] == cls)[0]
                            if len(cls_indices) > 1:  # 保留至少一个样本
                                reduce_count = int(len(cls_indices) * swap_intensity * 0.5)
                                if reduce_count > 0 and len(cls_indices) > reduce_count:
                                    remove_indices = np.random.choice(cls_indices, reduce_count, replace=False)
                                    keep_mask = np.ones(len(X[client_id]), dtype=bool)
                                    keep_mask[remove_indices] = False
                                    X[client_id] = X[client_id][keep_mask]
                                    y[client_id] = y[client_id][keep_mask]
            
            if apply_drift:
                print(f"客户端 {client_id} (聚类 {client_clusters[client_id]}) 在迭代 {iteration} 发生 {drift_type} 漂移")
        
        # 划分训练和测试数据
        train_data, test_data = {}, {}
        
        for c in range(num_clients):
            # 分割数据为训练集和测试集
            X_train, X_test = X[c][:int(0.8*len(X[c]))], X[c][int(0.8*len(X[c])):]
            y_train, y_test = y[c][:int(0.8*len(y[c]))], y[c][int(0.8*len(y[c])):]
            
            train_data[c] = {'x': X_train, 'y': y_train}
            test_data[c] = {'x': X_test, 'y': y_test}
        
        # 保存此迭代的数据
        save_file(
            iteration_config_path, 
            iteration_train_path, 
            iteration_test_path, 
            train_data, 
            test_data, 
            num_clients, 
            num_classes, 
            statistic, 
            niid, 
            balance, 
            partition
        )
        
        print(f"迭代 {iteration} 的数据生成完成")
    
    print("\n所有迭代的数据生成完成，概念漂移已应用到数据集中")
    
    return cluster_info

# 用于序列化包含numpy数组的JSON
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)
print(f"  应用逐步漂移，强度为 {intensity:.2f}")
                                        print(f"  应用突变漂移")

                    print(f"  应用循环漂移")

if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else True
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_cifar100_with_clusters(dir_path, num_clients, num_classes, niid, balance, partition)