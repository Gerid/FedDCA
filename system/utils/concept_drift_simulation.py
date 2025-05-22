"""
概念漂移模拟模块

此模块提供了联邦学习环境中模拟概念漂移的工具函数，
包括概念定义、漂移类型和数据变换等功能。

主要功能：
1. 创建和管理共享概念
2. 实现不同类型的概念漂移（突变、渐进、周期性）
3. 应用数据变换实现漂移效果
"""

import numpy as np
import torch
import copy
from torch.utils.data import TensorDataset
import os
import json
import pickle

# 全局变量 - 由所有客户端共享的概念集合
GLOBAL_CONCEPTS = None
GLOBAL_DRIFT_PATTERNS = None

def create_shared_concepts(num_concepts=5, num_classes=100, seed=42):
    """
    创建所有客户端共享的概念集合
    
    Args:
        num_concepts: 概念数量
        num_classes: 数据集类别数量
        seed: 随机种子，确保可重复性
        
    Returns:
        concepts: 概念列表
    """
    global GLOBAL_CONCEPTS
    
    # 如果已经创建过概念，则直接返回
    if GLOBAL_CONCEPTS is not None:
        return GLOBAL_CONCEPTS
    
    # 设置随机种子以确保可重复性
    np.random.seed(seed)
    
    concepts = []
    
    for i in range(num_concepts):
        # 确定当前概念偏好的类别数量 (10-30)
        num_preferred_classes = np.random.randint(10, 31)
        
        # 随机选择偏好的类别
        preferred_classes = np.random.choice(num_classes, num_preferred_classes, replace=False)
        
        # 为偏好类别创建标签映射（条件分布变换）
        mapping_type = np.random.choice(['swap', 'shift', 'random'])
        label_mapping = {}
        
        if mapping_type == 'swap':
            # 交换类别: 成对交换标签
            perm_classes = preferred_classes.copy()
            np.random.shuffle(perm_classes)
            for j in range(0, len(preferred_classes) - 1, 2):
                if j+1 < len(preferred_classes):
                    label_mapping[int(preferred_classes[j])] = int(perm_classes[j+1])
                    label_mapping[int(preferred_classes[j+1])] = int(perm_classes[j])
            
        elif mapping_type == 'shift':
            # 偏移类别: 将标签向前偏移一定数量
            shift = np.random.randint(1, 50)
            for cls in preferred_classes:
                label_mapping[int(cls)] = int((cls + shift) % num_classes)
                
        else:  # random
            # 随机映射: 随机分配新的类别
            targets = np.random.choice(num_classes, len(preferred_classes), replace=False)
            for j, cls in enumerate(preferred_classes):
                label_mapping[int(cls)] = int(targets[j])
        
        # 为这些类别分配权重 (主要用于选择这些类别样本)
        class_weights = {}
        for c in range(num_classes):
            if c in preferred_classes:
                # 偏好类别给予较高权重
                class_weights[c] = 0.5 + 0.5 * np.random.random()
            else:
                # 非偏好类别给予较低权重
                class_weights[c] = 0.1
                
        # 创建概念
        concept = {
            'id': i,
            'label_mapping': label_mapping,       # 关键: 使用标签映射实现条件分布变化
            'class_weights': class_weights,       # 用于控制样本选择的偏好
            'preferred_classes': preferred_classes.tolist(),
            'mapping_type': mapping_type
        }
        
        concepts.append(concept)
    
    # 存储全局概念
    GLOBAL_CONCEPTS = concepts
    return concepts

def initialize_drift_patterns():
    """
    初始化可用的漂移模式
    
    定义各种不同类型的漂移模式，可以根据迭代次数进行切换
    
    Returns:
        drift_patterns: 漂移模式字典
    """
    global GLOBAL_DRIFT_PATTERNS
    
    # 如果已经创建过漂移模式，则直接返回
    if GLOBAL_DRIFT_PATTERNS is not None:
        return GLOBAL_DRIFT_PATTERNS
    
    # 定义多种漂移模式
    drift_patterns = {
        # 1. 标签漂移 - 将某些类别的标签互换
        'label_drift_mild': {
            'label_mapping': {0: 1, 1: 0, 10: 11, 11: 10}  # 交换一些相似类别
        },
        'label_drift_moderate': {
            'label_mapping': {i: (i+5)%100 for i in range(0, 20)}  # 更多类别发生变化
        },
        'label_drift_severe': {
            'label_mapping': {i: (i+50)%100 for i in range(0, 100)}  # 大规模标签变化
        },
        
        # 2. 样本分布漂移 - 改变类别的先验概率
        'prior_drift_mild': {
            'class_probs': {str(i): (1.5 if i < 10 else 0.5)/100 for i in range(100)}
        },
        'prior_drift_severe': {
            'class_probs': {str(i): (3.0 if i < 5 else 0.1)/100 for i in range(100)}
        },
        
        # 3. 协变量漂移 - 改变输入特征的分布
        'covariate_noise': {
            'transform': {'type': 'gaussian_noise', 'params': {'severity': 2}}
        },
        'covariate_blur': {
            'transform': {'type': 'gaussian_blur', 'params': {'severity': 2}}
        },
        'covariate_brightness': {
            'transform': {'type': 'brightness', 'params': {'severity': 3}}
        },
        'covariate_contrast': {
            'transform': {'type': 'contrast', 'params': {'severity': 2}}
        },
        'covariate_elastic': {
            'transform': {'type': 'elastic_transform', 'params': {'severity': 2}}
        },
        'covariate_saturate': {
            'transform': {'type': 'saturate', 'params': {'severity': 3}}
        },
        
        # 4. 组合漂移
        'combined_mild': {
            'label_mapping': {0: 5, 5: 0, 10: 15, 15: 10},
            'class_probs': {str(i): (2.0 if i % 10 == 0 else 0.8)/100 for i in range(100)},
            'transform': {'type': 'gaussian_noise', 'params': {'severity': 1}}
        },
        'combined_severe': {
            'label_mapping': {i: (i+25)%100 for i in range(0, 50)},
            'class_probs': {str(i): (3.0 if i < 10 else 0.2)/100 for i in range(100)},
            'transform': {'type': 'combined', 'params': {
                'severity': 3,
                'sequence': ['gaussian_noise', 'contrast', 'brightness']
            }}
        }
    }
    
    GLOBAL_DRIFT_PATTERNS = drift_patterns
    return drift_patterns

def assign_client_concepts(client_id, all_concepts=None, num_concepts_per_client=None, seed=None):
    """
    为客户端分配概念

    Args:
        client_id: 客户端ID
        all_concepts: 所有可用概念的列表
        num_concepts_per_client: 每个客户端分配的概念数量，如果为None则随机分配2-3个
        seed: 随机种子，确保可重复性
        
    Returns:
        client_concepts: 分配给客户端的概念列表
    """
    if all_concepts is None:
        all_concepts = create_shared_concepts()
    
    if seed is not None:
        np.random.seed(seed + client_id)  # 为每个客户端设置不同但可复现的随机种子
    
    # 确定要分配给客户端的概念数量
    if num_concepts_per_client is None:
        num_client_concepts = np.random.randint(2, 4)  # 分配2-3个概念
    else:
        num_client_concepts = min(num_concepts_per_client, len(all_concepts))
    
    # 随机选择概念
    concept_indices = np.random.choice(len(all_concepts), num_client_concepts, replace=False)
    
    # 分配概念给客户端
    client_concepts = [all_concepts[idx] for idx in concept_indices]
    
    return client_concepts

def apply_concept_distribution(data, labels, concept):
    """
    根据当前概念调整类别分布和标签
    
    主要功能:
    1. 基于概念的标签映射，改变数据的条件分布 p(y|x)
    2. 基于类别权重，调整样本的类别分布 p(y)
    
    Args:
        data: 输入数据张量
        labels: 标签张量
        concept: 概念定义，包含标签映射和类别偏好权重
        
    Returns:
        tuple: (调整后的数据, 调整后的标签)
    """
    # 如果概念没有定义，直接返回原始数据
    if concept is None:
        return data, labels
        
    # 复制标签，避免修改原始数据
    new_labels = labels.clone()
    
    # 1. 首先应用标签映射 - 改变条件分布 p(y|x)
    if 'label_mapping' in concept and concept['label_mapping']:
        label_mapping = concept['label_mapping']
        
        # 应用映射，将特定类别的标签转换为新标签
        for orig_label, new_label in label_mapping.items():
            mask = (labels == orig_label)
            if mask.any():
                new_labels[mask] = new_label
    
    # 2. 然后基于类别权重调整样本数量 - 调整类别分布 p(y)
    if 'class_weights' in concept and concept['class_weights']:
        class_weights = concept['class_weights']
        total_classes = 100  # CIFAR-100的类别数
        
        # 按类别组织样本索引
        class_indices = {}
        for i, label in enumerate(new_labels):  # 注意这里使用的是已经映射后的标签
            label_int = int(label.item())
            if label_int not in class_indices:
                class_indices[label_int] = []
            class_indices[label_int].append(i)
            
        # 基于权重计算各类别目标样本数
        total_samples = len(new_labels)
        target_counts = {}
        all_weights = sum(class_weights.values())
        
        for class_idx in range(total_classes):
            weight = class_weights.get(class_idx, 0.1)  # 默认非偏好类别权重为0.1
            target_counts[class_idx] = max(1, int(total_samples * (weight / all_weights)))
            
        # 创建新数据集
        final_data = []
        final_labels = []
        
        # 对每个类别进行抽样，调整到目标样本数
        for class_idx in range(total_classes):
            target_count = target_counts.get(class_idx, 0)
            if target_count > 0 and class_idx in class_indices and len(class_indices[class_idx]) > 0:
                # 对于需要增加的类别，允许重复抽样
                indices = np.random.choice(
                    class_indices[class_idx],
                    size=target_count,
                    replace=True
                )
                
                for idx in indices:
                    final_data.append(data[idx])
                    final_labels.append(new_labels[idx])
        
        if len(final_data) > 0:
            return torch.stack(final_data), torch.stack(final_labels)
            
    # 如果只进行了标签映射但没有调整分布，或者没有成功创建新数据集
    return data, new_labels

def get_current_concept(client_concepts, drift_type, current_iter, drift_points, window_size=10, period=30):
    """
    根据当前迭代和漂移类型确定要使用的概念
    
    支持三种漂移类型:
    1. 突变漂移(sudden drift): 在漂移点立即切换到新概念
    2. 渐进漂移(gradual drift): 在漂移点周围窗口期内，逐渐从一个概念过渡到另一个
    3. 周期漂移(recurring drift): 按照周期在可用概念间循环切换
    
    Args:
        client_concepts: 客户端可用的概念列表
        drift_type: 漂移类型 ('sudden', 'gradual', 'recurring')
        current_iter: 当前迭代轮次
        drift_points: 漂移点列表
        window_size: 渐进漂移的窗口大小
        period: 周期漂移的周期长度
        
    Returns:
        当前应用的概念或混合概念
    """
    # 如果客户端没有概念，返回None
    num_concepts = len(client_concepts)
    if num_concepts == 0:
        return None
    elif num_concepts == 1:
        return copy.deepcopy(client_concepts[0])
        
    # 根据漂移类型确定当前概念
    if drift_type == 'sudden':
        # 突变漂移: 在漂移点立即切换概念
        concept_idx = 0
        for i, point in enumerate(drift_points):
            if current_iter >= point:
                concept_idx = (i + 1) % num_concepts
        return copy.deepcopy(client_concepts[concept_idx])
        
    elif drift_type == 'gradual':
        # 渐进漂移: 在漂移点周围一段时间内，逐渐过渡到新概念
        # 确定基础概念索引
        base_idx = 0
        target_idx = 0
        transition_prob = 0.0
        in_transition = False
        
        for i, point in enumerate(drift_points):
            if current_iter >= point:
                base_idx = (i + 1) % num_concepts
                
            # 检查是否在过渡窗口内
            if point - window_size <= current_iter < point:
                base_idx = i % num_concepts
                target_idx = (i + 1) % num_concepts
                # 计算过渡概率
                transition_prob = (current_iter - (point - window_size)) / window_size
                in_transition = True
                break
        
        # 如果在过渡期，创建混合概念 - 结合两个概念的标签映射
        if in_transition and transition_prob > 0:
            base_concept = client_concepts[base_idx]
            target_concept = client_concepts[target_idx]
            
            # 创建混合概念
            mixed_concept = copy.deepcopy(base_concept)
            
            # 混合标签映射
            if 'label_mapping' in base_concept and 'label_mapping' in target_concept:
                base_mapping = base_concept['label_mapping']
                target_mapping = target_concept['label_mapping']
                
                # 选择标签映射的混合策略
                if np.random.random() < transition_prob:
                    # 随着转换概率增加，更多地使用目标概念的映射
                    for label, new_label in target_mapping.items():
                        if np.random.random() < transition_prob:
                            mixed_concept['label_mapping'][label] = new_label
            
            # 混合类别权重
            if 'class_weights' in base_concept and 'class_weights' in target_concept:
                base_weights = base_concept['class_weights']
                target_weights = target_concept['class_weights']
                mixed_weights = {}
                
                # 根据过渡概率加权混合两个概念的权重
                for class_idx in range(100):  # CIFAR-100
                    base_w = base_weights.get(class_idx, 0.1)
                    target_w = target_weights.get(class_idx, 0.1)
                    mixed_weights[class_idx] = base_w * (1 - transition_prob) + target_w * transition_prob
                    
                mixed_concept['class_weights'] = mixed_weights
                
            return mixed_concept
        else:
            # 不在过渡期，使用当前概念
            return copy.deepcopy(client_concepts[base_idx])
            
    elif drift_type == 'recurring':
        # 周期漂移: 按固定周期循环使用概念
        concept_idx = (current_iter // period) % num_concepts
        return copy.deepcopy(client_concepts[concept_idx])
        
    else:
        # 默认使用第一个概念
        return copy.deepcopy(client_concepts[0])

def apply_image_transforms(image, transforms):
    """
    应用各种图像变换
    
    Args:
        image: 输入图像(numpy数组)
        transforms: 变换参数
        
    Returns:
        变换后的图像
    """
    try:
        import skimage as sk
        from skimage.filters import gaussian
        import cv2
        from scipy.ndimage import zoom as scizoom
        from scipy.ndimage.interpolation import map_coordinates
    except ImportError:
        print("警告: 缺少图像处理依赖库。安装 scikit-image, opencv-python 和 scipy 以启用完整功能。")
        return image * 255.0
    
    # 确保输入图像为numpy数组并且值范围为[0,1]
    if isinstance(image, np.ndarray) and image.max() > 1.0:
        image = image / 255.0
    
    transform_type = transforms.get('type', 'none')
    params = transforms.get('params', {})
    severity = params.get('severity', 1)
    
    # 根据变换类型应用不同处理
    if transform_type == 'none':
        return image * 255.0  # 返回值范围[0,255]
        
    elif transform_type == 'gaussian_noise':
        # 高斯噪声
        c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
        noisy = image + np.random.normal(size=image.shape, scale=c)
        return np.clip(noisy, 0, 1) * 255.0
        
    elif transform_type == 'shot_noise':
        # 散粒噪声
        c = [60, 25, 12, 5, 3][severity - 1]
        noisy = np.random.poisson(image * c) / c
        return np.clip(noisy, 0, 1) * 255.0
        
    elif transform_type == 'impulse_noise':
        # 脉冲噪声
        c = [.03, .06, .09, 0.17, 0.27][severity - 1]
        noisy = sk.util.random_noise(image, mode='s&p', amount=c)
        return np.clip(noisy, 0, 1) * 255.0
        
    elif transform_type == 'speckle_noise':
        # 斑点噪声
        c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]
        noisy = image + image * np.random.normal(size=image.shape, scale=c)
        return np.clip(noisy, 0, 1) * 255.0
        
    elif transform_type == 'gaussian_blur':
        # 高斯模糊
        c = [1, 2, 3, 4, 6][severity - 1]
        blurred = gaussian(image, sigma=c/10, multichannel=True)
        return np.clip(blurred, 0, 1) * 255.0
        
    elif transform_type == 'contrast':
        # 对比度调整
        c = [0.7, 0.6, 0.5, 0.4, 0.3][severity - 1]
        means = np.mean(image, axis=(0, 1), keepdims=True)
        contrasted = (image - means) * c + means
        return np.clip(contrasted, 0, 1) * 255.0
        
    elif transform_type == 'brightness':
        # 亮度调整
        c = [.1, .2, .3, .4, .5][severity - 1]
        brightened = image + c
        return np.clip(brightened, 0, 1) * 255.0
        
    elif transform_type == 'saturate':
        # 饱和度调整
        c = [0.3, 0.5, 1.5, 2.5, 3.0][severity - 1]
        if len(image.shape) == 3 and image.shape[2] == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = hsv[:, :, 1] * c
            saturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            return np.clip(saturated, 0, 1) * 255.0
        return image * 255.0
        
    elif transform_type == 'elastic_transform':
        # 弹性变换
        def elastic_transform_impl(image, severity=1):
            imshape = image.shape
            c = [(imshape[0]*0.05, imshape[0]*0.03, imshape[0]*0.02),
                 (imshape[0]*0.07, imshape[0]*0.05, imshape[0]*0.03),
                 (imshape[0]*0.09, imshape[0]*0.07, imshape[0]*0.05),
                 (imshape[0]*0.11, imshape[0]*0.09, imshape[0]*0.07),
                 (imshape[0]*0.15, imshape[0]*0.11, imshape[0]*0.09)][severity - 1]
            
            shape = image.shape
            dx = gaussian(np.random.uniform(-1, 1, size=shape[:2]), c[1], mode='reflect') * c[0]
            dy = gaussian(np.random.uniform(-1, 1, size=shape[:2]), c[1], mode='reflect') * c[0]

            x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
            indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
                
            return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
        
        return np.clip(elastic_transform_impl(image, severity), 0, 1) * 255.0
    
    elif transform_type == 'combined':
        # 组合多种变换
        transform_sequence = params.get('sequence', ['gaussian_noise', 'contrast'])
        transformed = image.copy()
        
        for t in transform_sequence:
            sub_transform = {'type': t, 'params': {'severity': severity}}
            transformed = apply_image_transforms(transformed/255.0, sub_transform) / 255.0
            
        return np.clip(transformed, 0, 1) * 255.0
        
    else:
        # 未知变换类型，返回原始图像
        return image * 255.0

def get_current_drift_pattern(drift_patterns, drift_schedule, drift_type, current_iter, drift_points):
    """
    根据当前迭代次数获取应用的漂移模式
    
    除了从预定义的drift_schedule中获取模式外，
    还会根据当前迭代与漂移点的关系，动态添加适当的变换
    
    Args:
        drift_patterns: 所有可用的漂移模式
        drift_schedule: 漂移时间表
        drift_type: 漂移类型
        current_iter: 当前迭代次数
        drift_points: 漂移点列表
        
    Returns:
        当前应用的漂移模式
    """
    if drift_schedule is None:
        return None
        
    # 查找当前迭代所处的漂移阶段
    base_pattern = None
    
    for schedule in drift_schedule:
        if schedule['iterations'][0] <= current_iter < schedule['iterations'][1]:
            pattern_name = schedule['pattern']
            if pattern_name is None:
                base_pattern = None
            else:
                base_pattern = copy.deepcopy(drift_patterns.get(pattern_name, None))
            break
    
    # 检查当前迭代是否接近漂移点
    if drift_points:
        # 查找最近的漂移点
        closest_drift_point = None
        min_distance = float('inf')
        
        for point in drift_points:
            distance = abs(current_iter - point)
            if distance < min_distance:
                min_distance = distance
                closest_drift_point = point
        
        # 如果非常接近漂移点(±3轮)，增加强烈的变换
        if min_distance <= 3:
            # 如果没有基础模式，创建一个新的
            if base_pattern is None:
                base_pattern = {}
            
            # 根据漂移类型添加额外变换
            if drift_type == 'sudden':
                # 突变漂移在漂移点添加强烈的组合变换
                if current_iter >= closest_drift_point:
                    base_pattern['transform'] = {
                        'type': 'combined',
                        'params': {
                            'severity': 3,
                            'sequence': ['gaussian_noise', 'contrast', 'brightness']
                        }
                    }
            
            elif drift_type == 'gradual':
                # 渐进漂移添加逐渐增强的变换
                if current_iter < closest_drift_point:
                    # 漂移前，轻微变换
                    severity = max(1, 3 - min_distance)
                    base_pattern['transform'] = {
                        'type': 'gaussian_noise',
                        'params': {'severity': severity}
                    }
                else:
                    # 漂移后，更强变换
                    severity = max(1, 4 - min_distance)
                    base_pattern['transform'] = {
                        'type': 'combined',
                        'params': {
                            'severity': severity,
                            'sequence': ['gaussian_blur', 'contrast']
                        }
                    }
            
            elif drift_type == 'recurring':
                # 周期漂移在漂移点使用随机变换
                transforms = ['gaussian_noise', 'gaussian_blur', 'contrast', 'brightness', 'elastic_transform']
                selected = np.random.choice(transforms)
                base_pattern['transform'] = {
                    'type': selected,
                    'params': {'severity': np.random.randint(1, 4)}
                }
    
    return base_pattern

def apply_drift_transformation(dataset, client_concepts, drift_patterns, drift_schedule, current_iter, client_id, 
                               drift_points=None, drift_type=None, gradual_window=10, recurring_period=30):
    """
    根据当前迭代和预设模式对数据集应用概念漂移变换
    
    Args:
        dataset: 原始数据集
        client_concepts: 客户端可用的概念列表
        drift_patterns: 可用的漂移模式
        drift_schedule: 漂移时间表
        current_iter: 当前迭代次数
        client_id: 客户端ID
        drift_points: 漂移点列表
        drift_type: 漂移类型
        gradual_window: 渐进漂移的窗口大小
        recurring_period: 周期漂移的周期长度
        
    Returns:
        transformed_dataset: 应用了漂移变换的数据集
    """
    import torch
    from torch.utils.data import TensorDataset
    
    # 如果漂移点未指定，使用默认值
    if drift_points is None:
        max_iterations = 200
        num_drifts = 5
        drift_points = [max_iterations * (i + 1) // (num_drifts + 1) for i in range(num_drifts)]
    
    # 如果漂移类型未指定，基于客户端ID确定
    if drift_type is None:
        drift_types = ['sudden', 'gradual', 'recurring']
        client_id_hash = hash(f"client_{client_id}") % 3
        drift_type = drift_types[client_id_hash]
    
    # 获取当前概念
    current_concept = get_current_concept(
        client_concepts, drift_type, current_iter, drift_points, 
        window_size=gradual_window, period=recurring_period
    )
    
    # 获取当前漂移模式
    current_pattern = get_current_drift_pattern(
        drift_patterns, drift_schedule, drift_type, current_iter, drift_points
    )
    
    if current_concept is None and current_pattern is None:
        return dataset  # 如果没有漂移模式，返回原始数据集
        
    # 从数据集提取数据和标签
    all_data = []
    all_labels = []
    for img, label in dataset:
        all_data.append(img)
        all_labels.append(label)
        
    if len(all_data) == 0:
        return dataset  # 空数据集，直接返回
        
    all_data = torch.stack(all_data)
    all_labels = torch.tensor(all_labels)
    
    # 应用基于概念的类别偏好调整
    if current_concept is not None:
        all_data, all_labels = apply_concept_distribution(all_data, all_labels, current_concept)
    
    # 如果有明确的漂移模式，则应用额外的变换
    if current_pattern is not None:
        # 应用标签变换 (label shift)
        if 'label_mapping' in current_pattern:
            mapping = current_pattern['label_mapping']
            new_labels = all_labels.clone()
            for old_label, new_label in mapping.items():
                new_labels[all_labels == old_label] = new_label
            all_labels = new_labels
            
        # 应用样本分布变换 (prior probability shift)
        if 'class_probs' in current_pattern:
            class_probs = current_pattern['class_probs']
            new_data = []
            new_labels = []
            
            # 按类别组织样本
            class_indices = {}
            for i, label in enumerate(all_labels):
                label_int = int(label.item())
                if label_int not in class_indices:
                    class_indices[label_int] = []
                class_indices[label_int].append(i)
                
            # 根据新的类别概率分布抽样
            total_samples = len(all_labels)
            for class_idx, prob in class_probs.items():
                class_idx = int(class_idx)
                if class_idx in class_indices and len(class_indices[class_idx]) > 0:
                    # 计算该类别应该有多少样本
                    sample_count = max(1, int(total_samples * prob))
                    
                    # 从该类别中随机抽样(可重复)
                    indices = np.random.choice(
                        class_indices[class_idx], 
                        size=sample_count, 
                        replace=True
                    )
                    
                    for idx in indices:
                        new_data.append(all_data[idx])
                        new_labels.append(all_labels[idx])
            
            # 如果样本数量发生变化，更新数据
            if len(new_data) > 0:
                all_data = torch.stack(new_data)
                all_labels = torch.stack(new_labels)
        
        # 应用数据增强和协变量漂移
        if 'transform' in current_pattern:
            transforms = current_pattern['transform']
            
            # 转换PyTorch张量为NumPy数组，方便处理
            np_images = all_data.permute(0, 2, 3, 1).cpu().numpy() if all_data.dim() == 4 else all_data.cpu().numpy()
            
            # 应用变换
            transformed_images = []
            for img in np_images:
                transformed = apply_image_transforms(img, transforms)
                transformed_images.append(transformed)
            
            # 转换回PyTorch张量格式
            transformed_tensor = torch.tensor(np.array(transformed_images))
            
            # 确保格式正确 - 转为BCHW (如果是图像)
            if transformed_tensor.dim() == 4 and transformed_tensor.shape[-1] == 3:  # BHWC格式
                transformed_tensor = transformed_tensor.permute(0, 3, 1, 2)  # 转换为BCHW格式
            
            all_data = transformed_tensor
            
    # 返回转换后的数据集
    return TensorDataset(all_data, all_labels)

def save_concepts_to_disk(all_concepts, save_dir):
    """
    将概念保存到磁盘
    
    Args:
        all_concepts: 所有概念的列表
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存所有概念的集合
    with open(os.path.join(save_dir, 'all_concepts.pkl'), 'wb') as f:
        pickle.dump(all_concepts, f)
    
    # 保存每个概念的详细信息到JSON文件
    for i, concept in enumerate(all_concepts):
        concept_path = os.path.join(save_dir, f'concept_{i}.json')
        
        # 将numpy整数类型转换为Python整数类型，以便JSON序列化
        concept_copy = copy.deepcopy(concept)
        if 'label_mapping' in concept_copy:
            new_mapping = {}
            for k, v in concept_copy['label_mapping'].items():
                new_mapping[int(k)] = int(v)
            concept_copy['label_mapping'] = new_mapping
            
        with open(concept_path, 'w') as f:
            json.dump(concept_copy, f, indent=2)
    
    print(f"已将所有概念保存到 {save_dir}")

def load_concepts_from_disk(load_dir):
    """
    从磁盘加载概念
    
    Args:
        load_dir: 加载目录
        
    Returns:
        all_concepts: 所有概念的列表
    """
    global GLOBAL_CONCEPTS
    
    # 尝试加载所有概念的集合
    try:
        with open(os.path.join(load_dir, 'all_concepts.pkl'), 'rb') as f:
            all_concepts = pickle.load(f)
        
        GLOBAL_CONCEPTS = all_concepts
        return all_concepts
    except:
        # 如果pickle文件不可用，尝试从JSON文件加载
        all_concepts = []
        i = 0
        
        while True:
            concept_path = os.path.join(load_dir, f'concept_{i}.json')
            if not os.path.exists(concept_path):
                break
                
            with open(concept_path, 'r') as f:
                concept = json.load(f)
                all_concepts.append(concept)
            
            i += 1
        
        if all_concepts:
            GLOBAL_CONCEPTS = all_concepts
            return all_concepts
            
    # 如果无法加载，创建新的概念
    print(f"无法从 {load_dir} 加载概念，创建新的概念")
    return create_shared_concepts()

def analyze_client_drift_paths(client_concepts, num_iterations=200, drift_types=None, client_id=None, drift_points=None):
    """
    分析客户端的漂移路径
    
    Args:
        client_concepts: 客户端可用的概念列表
        num_iterations: 总迭代次数
        drift_types: 漂移类型列表，如果为None则使用默认值
        client_id: 客户端ID
        drift_points: 漂移点列表
        
    Returns:
        drift_path: 客户端的漂移路径，格式为 {iteration: concept_id}
    """
    drift_path = {}
    
    # 如果未指定漂移类型，使用默认值
    if drift_types is None:
        drift_types = ['sudden', 'gradual', 'recurring']
    
    # 如果未指定客户端ID，使用0
    if client_id is None:
        client_id = 0
    
    # 确定漂移类型
    drift_type = drift_types[client_id % len(drift_types)]
    
    # 如果漂移点未指定，使用默认值
    if drift_points is None:
        drift_points = [num_iterations * (i + 1) // 6 for i in range(5)]
    
    # 设置参数
    gradual_window = 10
    recurring_period = np.random.randint(20, 41)
    
    # 分析每个迭代的概念
    for iter_num in range(num_iterations):
        current_concept = get_current_concept(
            client_concepts, drift_type, iter_num, drift_points, 
            window_size=gradual_window, period=recurring_period
        )
        
        if current_concept is not None:
            drift_path[iter_num] = current_concept['id']
        else:
            drift_path[iter_num] = -1  # 无概念
    
    return drift_path
