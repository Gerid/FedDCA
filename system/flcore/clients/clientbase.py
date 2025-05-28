import copy
import torch
import torch.nn as nn
import numpy as np
import os
import json
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        # 添加对概念漂移数据集的支持
        self.current_iteration = 0
        self.drift_data_dir = args.drift_data_dir if hasattr(args, 'drift_data_dir') else None
        self.use_drift_dataset = args.use_drift_dataset if hasattr(args, 'use_drift_dataset') else False
        self.max_iterations = args.max_iterations if hasattr(args, 'max_iterations') else 200
          # 模拟漂移相关参数
        self.simulate_drift = args.simulate_drift if hasattr(args, 'simulate_drift') else False
        self.increment_iteration = args.increment_iteration if hasattr(args, 'increment_iteration') else True
        self.shared_concepts = []
        self.client_concepts = []
        self.current_concept_id = -1
        self.current_concept = None
        self.drift_patterns = None
        self.drift_schedule = None
        self.drift_args = None
        self.use_shared_concepts = False
        self.gradual_window = 10
        self.recurring_period = 30
        self.all_concepts = None
        self.drift_type = None
        self.drift_points = None
        self.drift_patterns = None
        self.drift_schedule = None
        self.drift_args = None
        self.use_shared_concepts = False
        self.gradual_window = 10
        self.recurring_period = 30
        self.all_concepts = None
        self.drift_type = None
        self.drift_points = None

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs["train_slow"]
        self.send_slow = kwargs["send_slow"]
        self.train_time_cost = {"num_rounds": 0, "total_cost": 0.0}
        self.send_time_cost = {"num_rounds": 0, "total_cost": 0.0}

        self.privacy = args.privacy        
        self.dp_sigma = args.dp_sigma

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay    

    def update_iteration(self, new_iteration):
        """更新当前迭代计数器，用于概念漂移数据集"""
        if hasattr(self, 'current_iteration'):
            self.current_iteration = new_iteration
            
    def update_concept(self, concept):
        """更新当前使用的概念"""
        self.current_concept = concept
        if concept is not None and 'id' in concept:
            self.current_concept_id = concept['id']
            
    def load_train_data(self, batch_size=None):
        """加载训练数据，支持概念漂移数据集"""
        if batch_size == None:
            batch_size = self.batch_size
        
        if hasattr(self, 'use_drift_dataset') and self.use_drift_dataset and hasattr(self, 'drift_data_dir') and self.drift_data_dir:
            return self.load_drift_data(is_train=True, batch_size=batch_size)
        else:
            train_data = read_client_data(self.dataset, self.id, is_train=True)
            
            # 如果启用了模拟漂移
            if hasattr(self, 'simulate_drift') and self.simulate_drift:
                train_data = self.apply_drift_transformation(train_data)
                
            return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        """加载测试数据，支持概念漂移数据集"""
        if batch_size == None:
            batch_size = self.batch_size
            
        if hasattr(self, 'use_drift_dataset') and self.use_drift_dataset and hasattr(self, 'drift_data_dir') and self.drift_data_dir:
            return self.load_drift_data(is_train=False, batch_size=batch_size)
        else:
            test_data = read_client_data(self.dataset, self.id, is_train=False)
            
            # 如果启用了模拟漂移，也应用于测试数据
            if hasattr(self, 'simulate_drift') and self.simulate_drift:
                test_data = self.apply_drift_transformation(test_data)
                
            return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
            
    def load_drift_data(self, is_train=True, batch_size=None):
        """
        从概念漂移数据集加载特定迭代的数据
        
        参数:
            is_train: 是否加载训练集，False则加载测试集
            batch_size: 批次大小，如果为None则使用默认批次大小
        """
        import os
        import json
        import numpy as np
        from torch.utils.data import DataLoader, TensorDataset
        
        if batch_size is None:
            batch_size = self.batch_size
        
        try:
            # 构建当前迭代的路径
            iter_path = os.path.join(self.drift_data_dir, f"iteration_{self.current_iteration}")
            
            # 确保迭代不超出范围
            if self.current_iteration >= self.max_iterations:
                print(f"Warning: Iteration {self.current_iteration} exceeds max {self.max_iterations}, resetting to 0")
                self.current_iteration = 0
                iter_path = os.path.join(self.drift_data_dir, f"iteration_{self.current_iteration}")
            
            # 构建数据文件路径
            data_type = "train" if is_train else "test"
            json_path = os.path.join(iter_path, data_type, f"{self.id}.json")
            npz_path = os.path.join(iter_path, data_type, f"{self.id}.npz")
            
            # 优先尝试加载 JSON，其次尝试 NPZ
            data = None
            if os.path.exists(json_path):
                # 从 JSON 加载
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif os.path.exists(npz_path):
                # 尝试从NPZ加载并检查必需键
                try:
                    loaded = np.load(npz_path)
                    if 'x' in loaded.files and 'y' in loaded.files:
                        data = {'x': loaded['x'], 'y': loaded['y']}
                    else:
                        raise KeyError("'x' or 'y' not found in NPZ archive")
                except KeyError as e:
                    print(f"Warning: NPZ data missing for client {self.id}: {str(e)}, falling back to default loader")
                    # 如果NPZ加载失败，回退到默认加载方式
                    if is_train:
                        train_data = read_client_data(self.dataset, self.id, is_train=True)
                        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)
                    else:
                        test_data = read_client_data(self.dataset, self.id, is_train=False)
                        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
            else:
                print(f"Warning: Data file not found for client {self.id}, paths: {json_path} or {npz_path}")
                # 回退到默认数据加载
                if is_train:
                    train_data = read_client_data(self.dataset, self.id, is_train=True)
                    return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)
                else:
                    test_data = read_client_data(self.dataset, self.id, is_train=False)
                    return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
            
            if data is not None:
                # 加载数据后的统一处理
                import torch
                x = torch.tensor(data['x'], dtype=torch.float32)
                y = torch.tensor(data['y'], dtype=torch.long)
                
                # 创建数据集和数据加载器
                dataset = TensorDataset(x, y)
                drop_last = True if is_train else False
                dataloader = DataLoader(
                    dataset, 
                    batch_size=batch_size, 
                    shuffle=True,
                    drop_last=drop_last
                )
                
                return dataloader
            
        except Exception as e:
            print(f"Error loading drift data for client {self.id}: {str(e)}")
            # 回退到默认数据加载
            if is_train:
                train_data = read_client_data(self.dataset, self.id, is_train=True)
                return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)
            else:
                test_data = read_client_data(self.dataset, self.id, is_train=False)
                return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average="micro")

        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y

    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(
            item,
            os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"),
        )

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(
            os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt")
        )

    def apply_drift_transformation(self, dataset):
        """根据当前迭代和预设模式对数据集应用概念漂移变换
        
        实现了三种主要的概念漂移类型：
        1. 突变漂移(sudden drift)：在漂移点立即切换到新概念
        2. 渐进漂移(gradual drift)：在漂移点周围窗口期内，逐渐从一个概念过渡到另一个
        3. 周期漂移(recurring drift)：按照周期在可用概念间循环切换
        
        Args:
            dataset: 原始数据集
            
        Returns:
            transformed_dataset: 应用了漂移变换的数据集
        """
        # 如果尚未初始化漂移参数，则初始化
        if not hasattr(self, 'drift_patterns') or self.drift_patterns is None:
            self.initialize_drift_patterns()
            
        # 如果客户端概念未初始化，则初始化
        if not hasattr(self, 'client_concepts') or self.client_concepts is None or len(self.client_concepts) == 0:
            self.initialize_client_concepts()

        current_concept = self.get_current_concept()
        
        if current_concept is None:
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
            all_data, all_labels = self.apply_concept_distribution(all_data, all_labels, current_concept)
                
        # 返回转换后的数据集
        return TensorDataset(all_data, all_labels)

    def apply_concept_distribution(self, data, labels, concept):
        """
        应用概念分布变换 - 按照当前概念，修改标签和数据分布
        
        Args:
            data: 原始数据
            labels: 原始标签
            concept: 当前概念，含有标签映射和类别权重
            
        Returns:
            转换后的数据和标签
        """
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
        
    def get_current_concept(self):
        """根据当前迭代和漂移类型确定要使用的概念
        
        支持三种漂移类型:
        1. 突变漂移(sudden drift): 在漂移点立即切换到新概念
        2. 渐进漂移(gradual drift): 在漂移点周围窗口期内，逐渐从一个概念过渡到另一个
        3. 周期漂移(recurring drift): 按照周期在可用概念间循环切换
        
        返回:
            当前应用的概念或混合概念
        """
        import numpy as np
        import copy
        
        # 尝试使用外部模块的函数来获取当前概念
        if hasattr(self, 'use_shared_concepts') and self.use_shared_concepts:
            try:
                from utils.concept_drift_simulation import get_current_concept as external_get_concept
                
                # 如果没有初始化客户端概念，先初始化
                if not hasattr(self, 'client_concepts') or self.client_concepts is None:
                    self.initialize_client_concepts()
                
                drift_type = getattr(self, 'drift_type', 'sudden')
                drift_points = getattr(self, 'drift_points', [40, 80, 120, 160])
                window_size = getattr(self, 'gradual_window', 10)
                period = getattr(self, 'recurring_period', 30)
                
                # 使用外部函数获取当前概念
                current_concept = external_get_concept(
                    self.client_concepts, 
                    drift_type, 
                    self.current_iteration, 
                    drift_points, 
                    window_size, 
                    period
                )
                
                # 记录当前概念ID (用于跟踪和分析)
                if current_concept is not None and 'id' in current_concept:
                    self.current_concept_id = current_concept['id']
                    
                return current_concept
                
            except ImportError:
                print(f"Warning: Failed to import external concept drift module for client {self.id}")
        
        # 如果没有初始化客户端概念，先初始化
        if not hasattr(self, 'client_concepts') or self.client_concepts is None or len(self.client_concepts) == 0:
            self.initialize_client_concepts()
            
        current_iter = self.current_iteration
        
        # 确定漂移类型
        drift_type = getattr(self, 'drift_type', 'sudden')
        
        # 获取漂移点列表
        drift_points = getattr(self, 'drift_points', [40, 80, 120, 160])  # 默认漂移点
        
        # 获取客户端可用概念列表
        available_concepts = self.client_concepts
        num_concepts = len(available_concepts)
        
        if num_concepts == 0:
            return None
        elif num_concepts == 1:
            # 记录当前概念ID (用于跟踪和分析)
            if 'id' in available_concepts[0]:
                self.current_concept_id = available_concepts[0]['id']
            return copy.deepcopy(available_concepts[0])
        
        # 根据漂移类型确定当前概念
        if drift_type == 'sudden':
            # 突变漂移: 在漂移点立即切换概念
            concept_idx = 0
            for i, point in enumerate(drift_points):
                if current_iter >= point:
                    concept_idx = (i + 1) % num_concepts
                    
            # 记录当前概念ID
            if 'id' in available_concepts[concept_idx]:
                self.current_concept_id = available_concepts[concept_idx]['id']
                
            return copy.deepcopy(available_concepts[concept_idx])
            
        elif drift_type == 'gradual':
            # 渐进漂移: 在漂移点周围一段时间内，逐渐过渡到新概念
            window_size = getattr(self, 'gradual_window', 10)  # 过渡窗口大小
            
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
            
            # 记录当前概念ID
            if 'id' in available_concepts[base_idx]:
                self.current_concept_id = available_concepts[base_idx]['id']
                
            # 如果在过渡期，创建混合概念 - 结合两个概念的标签映射
            if in_transition and transition_prob > 0:
                base_concept = available_concepts[base_idx]
                target_concept = available_concepts[target_idx]
                
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
                return copy.deepcopy(available_concepts[base_idx])
                
        elif drift_type == 'recurring':
            # 周期漂移: 按固定周期循环使用概念
            period = getattr(self, 'recurring_period', 30)  # 默认30轮一个周期
            concept_idx = (current_iter // period) % num_concepts
            
            # 记录当前概念ID
            if 'id' in available_concepts[concept_idx]:
                self.current_concept_id = available_concepts[concept_idx]['id']
                
            return copy.deepcopy(available_concepts[concept_idx])
            
        else:
            # 默认使用第一个概念
            if 'id' in available_concepts[0]:
                self.current_concept_id = available_concepts[0]['id']
                
            return copy.deepcopy(available_concepts[0])
            
    def initialize_drift_patterns(self):
        """初始化概念漂移模式
        
        定义各种不同类型的漂移模式，可以根据迭代次数进行切换
        """
        import numpy as np
        
        # 定义多种漂移模式
        self.drift_patterns = {
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
        
        # 基于客户端ID确定漂移类型，以确保客户端具有不同类型的漂移
        drift_types = ['sudden', 'gradual', 'recurring']
        client_id_hash = hash(f"client_{self.id}") % 3
        self.drift_type = drift_types[client_id_hash]
        
        # 设置漂移点 - 在200轮训练中均匀分布5个漂移点
        iterations = getattr(self, 'max_iterations', 200)
        num_drifts = 5
        self.drift_points = [iterations * (i + 1) // (num_drifts + 1) for i in range(num_drifts)]
        
        # 渐进漂移的窗口大小
        self.gradual_window = 10
        
        # 设置漂移时间表
        self.drift_schedule = [
            {'iterations': [0, self.drift_points[0]], 'pattern': 'label_drift_mild'},
            {'iterations': [self.drift_points[0], self.drift_points[1]], 'pattern': 'prior_drift_mild'},
            {'iterations': [self.drift_points[1], self.drift_points[2]], 'pattern': 'covariate_noise'},
            {'iterations': [self.drift_points[2], self.drift_points[3]], 'pattern': 'covariate_brightness'},
            {'iterations': [self.drift_points[3], self.drift_points[4]], 'pattern': 'combined_mild'},
            {'iterations': [self.drift_points[4], iterations], 'pattern': 'combined_mild'}
        ]
        
    def initialize_client_concepts(self):
        """初始化客户端概念
        
        优先使用服务器分配的共享概念，如果没有则自行创建
        共享概念可确保所有客户端使用相同的概念集合，便于分析
        """
        import numpy as np
        
        # 首先检查是否有服务器分配的共享概念
        if hasattr(self, 'shared_concepts') and self.shared_concepts is not None and len(self.shared_concepts) > 0:
            print(f"客户端 {self.id} 使用服务器分配的共享概念")
            self.client_concepts = self.shared_concepts
            
            # 记录当前概念ID (用于跟踪和分析)
            self.current_concept_id = 0
            if len(self.client_concepts) > 0 and 'id' in self.client_concepts[0]:
                self.current_concept_id = self.client_concepts[0]['id']
                
            return
        
        # 如果没有共享概念，则尝试使用外部模块创建
        try:
            # print(f"客户端 {self.id} 尝试从外部模块加载共享概念")
            from utils.concept_drift_simulation import create_shared_concepts, assign_client_concepts
            
            # 创建全局共享概念
            all_concepts = create_shared_concepts(num_concepts=5, num_classes=100, seed=42)
            
            # 为当前客户端分配概念
            seed_value = hash(f"client_{self.id}_seed") % 10000
            self.client_concepts = assign_client_concepts(
                client_id=self.id,
                all_concepts=all_concepts,
                seed=seed_value
            )
            
            # 记录当前概念ID (用于跟踪和分析)
            self.current_concept_id = 0
            if len(self.client_concepts) > 0 and 'id' in self.client_concepts[0]:
                self.current_concept_id = self.client_concepts[0]['id']
                
            return
            
        except (ImportError, Exception) as e:
            print(f"无法从外部模块加载概念，将使用内部方法: {str(e)}")
            
            # 总概念数
            num_concepts = 5
            # 总类别数 (CIFAR-100)
            num_classes = 100
            
            # 如果概念已经初始化过，则跳过
            if hasattr(self, 'all_concepts') and self.all_concepts is not None:
                # 只为当前客户端选择概念
                self.client_concepts = []
                # 为客户端分配2-3个概念
                num_client_concepts = np.random.randint(2, 4)
                concept_indices = np.random.choice(len(self.all_concepts), num_client_concepts, replace=False)
                for idx in concept_indices:
                    self.client_concepts.append(self.all_concepts[idx])
                return
                
            # 创建5个不同的概念，每个概念代表不同的标签变换
            self.all_concepts = []
            
            for i in range(num_concepts):
                # 确定当前概念偏好的类别数量 (10-30)
                num_preferred_classes = np.random.randint(10, 31)
                
                # 随机选择偏好的类别
                preferred_classes = np.random.choice(num_classes, num_preferred_classes, replace=False)
                
                # 设置类别权重 - 偏好类别获得较高权重，非偏好类别获得较低权重
                class_weights = {}
                
                # 确定标签映射类型
                mapping_type = np.random.choice(['identical', 'partial_swap', 'random'], p=[0.5, 0.3, 0.2])
                
                # 创建标签映射
                label_mapping = {}
                
                if mapping_type == 'identical':
                    # 不改变标签
                    pass
                    
                elif mapping_type == 'partial_swap':
                    # 交换一些相似类别的标签
                    num_swaps = np.random.randint(5, 16)  # 交换5-15对标签
                    for _ in range(num_swaps):
                        label1 = np.random.randint(0, num_classes)
                        label2 = np.random.randint(0, num_classes)
                        label_mapping[label1] = label2
                        label_mapping[label2] = label1
                        
                elif mapping_type == 'random':
                    # 随机分配新标签
                    num_changes = np.random.randint(10, 51)  # 改变10-50个类别
                    labels_to_change = np.random.choice(num_classes, num_changes, replace=False)
                    new_labels = np.random.permutation(labels_to_change)
                    for orig, new in zip(labels_to_change, new_labels):
                        label_mapping[orig] = new
                
                # 设置类别权重
                for c in range(num_classes):
                    if c in preferred_classes:
                        # 偏好类别给予较高权重
                        class_weights[c] = np.random.uniform(1.0, 3.0)
                    else:
                        # 非偏好类别给予较低权重
                        class_weights[c] = 0.1
                        
                # 创建概念
                concept = {
                    'id': i,
                    'label_mapping': label_mapping,       # 关键变化: 使用标签映射
                    'class_weights': class_weights,       # 仅用于控制样本选择的偏好
                    'preferred_classes': preferred_classes.tolist(),
                    'mapping_type': mapping_type
                }
                
                self.all_concepts.append(concept)
                
            # 为当前客户端选择概念
            self.client_concepts = []
            # 为客户端分配2-3个概念
            num_client_concepts = np.random.randint(2, 4)
            concept_indices = np.random.choice(num_concepts, num_client_concepts, replace=False)
            for idx in concept_indices:
                self.client_concepts.append(self.all_concepts[idx])
                
            # 记录当前概念ID (用于跟踪和分析)
            self.current_concept_id = 0
            if len(self.client_concepts) > 0 and 'id' in self.client_concepts[0]:
                self.current_concept_id = self.client_concepts[0]['id']
