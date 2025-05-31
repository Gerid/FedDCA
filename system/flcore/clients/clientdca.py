import copy
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from ..clients.clientbase import Client
from utils.data_utils import read_client_data


class clientDCA(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # 保存所有必要的参数
        self.args = args
        self.learning_rate = args.local_learning_rate
        self.learning_rate_decay_gamma = args.learning_rate_decay_gamma

        # 添加迭代跟踪相关属性
        self.current_iteration = 0
        self.drift_data_dir = args.drift_data_dir if hasattr(args, 'drift_data_dir') else None
        self.use_drift_dataset = args.use_drift_dataset if hasattr(args, 'use_drift_dataset') else False
        self.max_iterations = args.max_iterations if hasattr(args, 'max_iterations') else 200
        
        # 添加标准数据集上的模拟漂移控制参数
        self.simulate_drift = args.simulate_drift if hasattr(args, 'simulate_drift') else True
        self.increment_iteration = args.increment_iteration if hasattr(args, 'increment_iteration') else True
        self.drift_patterns = None
        self.drift_schedule = None

        try:
            self.alpha = args.alpha
            self.beta = args.beta
            if args.model is None:
                raise ValueError("Model not provided in args")
            self.model = copy.deepcopy(args.model)
            self.global_model = copy.deepcopy(args.model)
            
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=self.learning_rate
            )
            self.optimizer_g = torch.optim.SGD(
                self.global_model.parameters(),
                lr=self.learning_rate
            )
            
            self.learning_rate_scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizer_g, 
                gamma=self.learning_rate_decay_gamma
            )

            self.catch_intermediate_output = True
            self.intermediate_output = None
            self.intermediate_outputs = []
            self.drift_interval = 20
            self.drift_args = None
            self.KL = nn.KLDivLoss()
            self.previous_loss = None
            self.drift_threshold = 0.1
            self.drift_detection_enabled = True
            
        except AttributeError as e:
            print(f"Error initializing client {id}: Missing required attribute - {str(e)}")
            raise
        except ValueError as e:
            print(f"Error initializing client {id}: {str(e)}")
            raise
        except RuntimeError as e:
            print(f"Runtime error initializing client {id}: {str(e)}")
            raise

    def receive_cluster_model(self, cluster_model):
        """Initialize the local model with the parameters of the cluster's centroid model."""
        if cluster_model is None:
            print(f"Warning: Received None cluster model for client {self.id}")
            return
            
        try:
            self.model = copy.deepcopy(cluster_model)
            self.global_model = copy.deepcopy(cluster_model)
            
            # 重新初始化优化器
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=self.learning_rate
            )
            self.optimizer_g = torch.optim.SGD(
                self.global_model.parameters(),
                lr=self.learning_rate
            )
            
            # 确保学习率调度器也被正确初始化
            if hasattr(self, 'learning_rate_scheduler_g'):
                self.learning_rate_scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer=self.optimizer_g,
                    gamma=self.learning_rate_decay_gamma
                )
                
        except (RuntimeError, ValueError) as e:
            print(f"Error in receive_cluster_model for client {self.id}: {str(e)}")
            # 如果复制失败，尝试保持当前模型状态
            if not hasattr(self, 'model') or self.model is None:
                print(f"Attempting to recover model state for client {self.id}")
                if hasattr(self, 'global_model') and self.global_model is not None:
                    self.model = copy.deepcopy(self.global_model)
                    self.optimizer = torch.optim.SGD(
                        self.model.parameters(),
                        lr=self.learning_rate
                    )

    def get_parameters(self):
        parameters = {}
        for name, param in self.model.state_dict().items():
            parameters[name] = param.detach().cpu().numpy()
        return parameters

    def forward_hook(self, _, __, layer_output):
        """注册前向传播钩子函数"""
        self.intermediate_outputs.append(layer_output)
        
    def register_hook(self):
        """注册特征层钩子"""
        try:
            feature_layer = self.model.base  # 使用属性访问而不是_modules
            if feature_layer:
                return feature_layer.register_forward_hook(self.forward_hook)
            return None
        except AttributeError:
            print(f"Warning: Model for client {self.id} has no 'base' layer")
            return None

    def calculate_intermediate_output_average(self):
        if self.intermediate_outputs:
            try:
                # 首先确保所有中间输出都在同一个设备上
                device_outputs = [output.to(self.device) for output in self.intermediate_outputs]
                
                # 处理不同批次大小的输出
                # 对每个输出先计算按第一维度的平均值，将形状从 [batch_size, feature_dim] 转换为 [1, feature_dim]
                device_outputs_normalized = [torch.mean(output, dim=0, keepdim=True) 
                                            for output in device_outputs]
                
                # 然后计算所有输出的平均值
                intermediate_output_avg = torch.mean(
                    torch.cat(device_outputs_normalized, dim=0), 0
                )
                self.intermediate_output = intermediate_output_avg.clone().detach()
                self.intermediate_outputs = []
            except (RuntimeError, ValueError) as e:
                print(f"Error calculating intermediate output: {str(e)}")
                self.intermediate_output = None
                self.intermediate_outputs = []
        else:
            self.intermediate_output = None
            
    def get_intermediate_outputs_with_labels(self):
        """
        获取模型中间层输出和对应的标签，按标签分组进行组织
        
        返回:
            features_by_label: 按标签分组的中间层特征字典，格式为 {label: [features]}
        """
        try:
            # 存储按标签组织的特征
            features_by_label = {}
            
            # 注册临时钩子
            hook_handle = None
            temp_outputs = []
            
            def temp_hook(module, input, output):
                temp_outputs.append(output.detach())
            
            # 尝试找到适合的特征层
            feature_layer = None
            if hasattr(self.model, 'base'):
                feature_layer = self.model.base
            elif hasattr(self.model, 'features'):
                feature_layer = self.model.features
            elif hasattr(self.model, 'encoder'):
                feature_layer = self.model.encoder
            
            if feature_layer is None:
                print(f"Warning: 无法找到适合的特征层, client_id: {self.id}")
                return {}
                
            hook_handle = feature_layer.register_forward_hook(temp_hook)
            
            # 加载训练数据
            loader = self.load_train_data()
            
            # 使用一小部分数据进行特征提取
            max_batches = 10  # 限制处理的批次数
            batch_count = 0
            
            with torch.no_grad():
                for x, y in loader:
                    batch_count += 1
                    if batch_count > max_batches:
                        break
                        
                    # 确保数据在正确的设备上
                    if isinstance(x, list):
                        x = [item.to(self.device) for item in x]
                    else:
                        x = x.to(self.device)
                    
                    # 前向传播，触发钩子函数
                    _ = self.model(x)
                    
                    # 获取最近生成的特征
                    if temp_outputs:
                        features = temp_outputs[-1]
                        
                        # 将特征按标签分组
                        for i, label in enumerate(y):
                            label_idx = label.item()
                            if label_idx not in features_by_label:
                                features_by_label[label_idx] = []
                            
                            # 添加这个样本的特征
                            if features.dim() > 2:  # 如果是卷积特征
                                # 全局平均池化
                                sample_feature = features[i].mean(dim=tuple(range(1, features.dim()-1)))
                            else:
                                sample_feature = features[i]
                            
                            features_by_label[label_idx].append(sample_feature.cpu())
                        
                        # 清空临时输出列表
                        temp_outputs.clear()
            
            # 移除钩子
            if hook_handle:
                hook_handle.remove()
                
            # 转换列表为张量
            for label in features_by_label:
                if features_by_label[label]:
                    features_by_label[label] = torch.stack(features_by_label[label])
            
            return features_by_label
            
        except Exception as e:
            print(f"Error in get_intermediate_outputs_with_labels for client {self.id}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
            # 移除钩子
            if 'hook_handle' in locals() and hook_handle:
                hook_handle.remove()
                
            return {}

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()
        hook_handle = None

        if self.catch_intermediate_output:
            hook_handle = self.register_hook()

        start_time = time.time()
        max_local_epochs = self.local_epochs
        
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        try:
            for _ in range(max_local_epochs):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)

                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))

                    output = self.model(x)
                    #output_g = self.global_model(x)
                    # loss = self.loss(output, y) * self.alpha + self.KL(
                    #     F.log_softmax(output, dim=1),
                    #     F.softmax(output_g, dim=1)
                    # ) * (1 - self.alpha)

                    loss = self.loss(output, y)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # 检测概念漂移
                    # if hasattr(self, 'drift_detection_enabled') and self.drift_detection_enabled:
                    #     if self.detect_drift(loss.item()):
                    #         print(f"Concept drift detected for client {self.id}")

        except (RuntimeError, ValueError) as e:
            print(f"Error during training for client {self.id}: {str(e)}")
            
        finally:
            if hook_handle is not None:
                hook_handle.remove()

            if self.catch_intermediate_output:
                self.calculate_intermediate_output_average()

            if hasattr(self, 'learning_rate_scheduler_g'):
                self.learning_rate_scheduler_g.step()

            self.train_time_cost["num_rounds"] += 1
            self.train_time_cost["total_cost"] += time.time() - start_time

    def test_metrics(self):
        testloader = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0

        try:
            with torch.no_grad():
                for x, y in testloader:
                    # if test_num >= 20:  # 限制测试样本数
                    #     break
                    if isinstance(x, list):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    
                    output = self.model(x)
                    test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                    test_num += y.shape[0]

        except RuntimeError as e:
            print(f"Error during testing for client {self.id}: {str(e)}")
            return 0, 0, 0

        return test_acc, test_num, 0

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()

        train_num = 0
        losses = 0
        
        try:
            with torch.no_grad():
                for x, y in trainloader:
                    # if train_num >= 20:  # 限制训练评估样本数
                    #     break
                    if isinstance(x, list):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    
                    output = self.model(x)
                    output_g = self.global_model(x)
                    loss = self.loss(output, y) * self.alpha + self.KL(
                        F.log_softmax(output, dim=1),
                        F.softmax(output_g, dim=1)
                    ) * (1 - self.alpha)

                    train_num += y.shape[0]
                    losses += loss.item() * y.shape[0]

        except RuntimeError as e:
            print(f"Error during training metrics for client {self.id}: {str(e)}")
            return 0, 0

        return losses, train_num

    def detect_drift(self, new_loss):
        """检测是否发生概念漂移"""
        if self.previous_loss is None:
            self.previous_loss = new_loss
            return False
            
        try:
            drift_coefficient = max(
                (new_loss - self.previous_loss) / self.previous_loss, 
                0
            )
            self.previous_loss = new_loss
            return drift_coefficient > self.drift_threshold
            
        except (ZeroDivisionError, ValueError) as e:
            print(f"Error detecting drift for client {self.id}: {str(e)}")
            return False    

    def reset_state(self):
        """在概念漂移点重置客户端状态"""
        # 重置损失跟踪
        self.previous_loss = None
        
        # 可以根据需要添加其他重置逻辑
        # 例如重置优化器状态等

    # def load_train_data(self, batch_size=None):
    #     """重写加载训练数据方法，以支持标准CIFAR-100上的模拟漂移
        
    #     根据当前迭代次数和预设的漂移模式修改数据标签和分布
    #     """
    #     if batch_size is None:
    #         batch_size = self.batch_size
        
    #     # 首先获取原始训练数据
    #     train_data = read_client_data(self.dataset, self.id, is_train=True)
        
    #     # 如果启用了标准数据集上的模拟漂移
    #     if hasattr(self, 'simulate_drift') and self.simulate_drift:
    #         # 应用漂移转simulate_drift换
    #         train_data = self.apply_drift_transformation(train_data)
            
    #         # 每次调用此方法时递增迭代计数
    #         if hasattr(self, 'increment_iteration') and self.increment_iteration:
    #             self.current_iteration = min(self.current_iteration + 1, self.max_iterations - 1)
                
    #     return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    # def load_test_data(self, batch_size=None):
    #     """重写加载训练数据方法，以支持标准CIFAR-100上的模拟漂移
        
    #     根据当前迭代次数和预设的漂移模式修改数据标签和分布
    #     """
    #     if batch_size is None:
    #         batch_size = self.batch_size

    #     # 首先获取原始训练数据
    #     test_data = read_client_data(self.dataset, self.id, is_train=False)

    #     # 如果启用了标准数据集上的模拟漂移
    #     if hasattr(self, 'simulate_drift') and self.simulate_drift:
    #         # 应用漂移转simulate_drift换
    #         test_data = self.apply_drift_transformation(test_data)

    #         # 每次调用此方法时递增迭代计数
    #         if hasattr(self, 'increment_iteration') and self.increment_iteration:
    #             self.current_iteration = min(
    #                 self.current_iteration + 1, self.max_iterations - 1)

    #     return DataLoader(test_data, batch_size, drop_last=True, shuffle=True)

    # def apply_drift_transformation(self, dataset):
    #     """根据当前迭代和预设模式对数据集应用概念漂移变换
        
    #     实现了三种主要的概念漂移类型：
    #     1. 突变漂移(sudden drift)：在漂移点立即切换到新概念
    #     2. 渐进漂移(gradual drift)：在漂移点周围窗口期内，逐渐从一个概念过渡到另一个
    #     3. 周期漂移(recurring drift)：按照周期在可用概念间循环切换
        
    #     Args:
    #         dataset: 原始数据集
            
    #     Returns:
    #         transformed_dataset: 应用了漂移变换的数据集
    #     """
        
    #     # 如果尚未初始化漂移参数，则初始化
    #     if not hasattr(self, 'drift_patterns') or self.drift_patterns is None:
    #         self.initialize_drift_patterns()
            


    #     current_concept = self.get_current_concept()
        
    #     if current_concept is None:
    #         return dataset  # 如果没有漂移模式，返回原始数据集
            
    #     # 从数据集提取数据和标签
    #     all_data = []
    #     all_labels = []
    #     for img, label in dataset:
    #         all_data.append(img)
    #         all_labels.append(label)
            
    #     if len(all_data) == 0:
    #         return dataset  # 空数据集，直接返回
            
    #     all_data = torch.stack(all_data)
    #     all_labels = torch.tensor(all_labels)
        
    #     # 应用基于概念的类别偏好调整
    #     if current_concept is not None:
    #         all_data, all_labels = self.apply_concept_distribution(all_data, all_labels, current_concept)
        
                
    #     # 返回转换后的数据集
    #     return TensorDataset(all_data, all_labels)
    
    def apply_concept_distribution(self, data, labels, concept):
        """根据当前概念调整类别分布和标签
        
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
        import torch
        import numpy as np
        
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
                if current_concept and 'id' in current_concept:
                    self.current_concept_id = current_concept['id']
                    
                return current_concept
                
            except (ImportError, Exception) as e:
                print(f"无法使用外部模块获取概念，将使用内部方法: {str(e)}")
        
        # 使用内部方法
        # 如果没有初始化客户端概念，先初始化
        if not hasattr(self, 'client_concepts') or self.client_concepts is None:
            self.initialize_client_concepts()
            
        # 如果客户端没有概念，返回None
        if not hasattr(self, 'client_concepts') or len(self.client_concepts) == 0:
            return None
            
        # 获取当前迭代轮次
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
        
        # 设置漂移类型 (sudden, gradual, recurring, 或混合)
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
        
        # 周期漂移的周期长度 (随机20-40轮)
        self.recurring_period = np.random.randint(20, 41)
        
        # 简化的漂移时间表，主要用于辅助变换
        self.drift_schedule = [
            {'iterations': [0, self.drift_points[0]], 'pattern': None},  
            {'iterations': [self.drift_points[0], self.drift_points[1]], 'pattern': 'covariate_noise'},
            {'iterations': [self.drift_points[1], self.drift_points[2]], 'pattern': 'covariate_blur'},
            {'iterations': [self.drift_points[2], self.drift_points[3]], 'pattern': 'covariate_brightness'},
            {'iterations': [self.drift_points[3], self.drift_points[4]], 'pattern': 'covariate_contrast'},
            {'iterations': [self.drift_points[4], iterations], 'pattern': 'combined_mild'}
        ]    
    
    def initialize_client_concepts(self):
        """初始化客户端概念
        
        优先使用服务器分配的共享概念，如果没有则自行创建
        共享概念可确保所有客户端使用相同的概念集合，便于分析
        """
        import numpy as np
        
        # 首先检查是否有服务器分配的共享概念
        if hasattr(self, 'shared_concepts') and self.shared_concepts is not None:
            print(f"客户端 {self.id} 使用服务器分配的共享概念")
            self.client_concepts = self.shared_concepts
            
            # 记录当前概念ID (用于跟踪和分析)
            self.current_concept_id = 0
            if len(self.client_concepts) > 0 and 'id' in self.client_concepts[0]:
                self.current_concept_id = self.client_concepts[0]['id']
            
            return
        
        # 如果没有共享概念，则尝试使用外部模块创建
        try:
            print(f"客户端 {self.id} 尝试从外部模块加载共享概念")
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
                
                # 为偏好类别创建标签映射（条件分布变换）
                # 有多种类型的映射策略:
                mapping_type = np.random.choice(['swap', 'shift', 'random'])
                label_mapping = {}
                
                if mapping_type == 'swap':
                    # 交换类别: 成对交换标签
                    perm_classes = preferred_classes.copy()
                    np.random.shuffle(perm_classes)
                    for j in range(0, len(preferred_classes) - 1, 2):
                        if j+1 < len(preferred_classes):
                            label_mapping[preferred_classes[j]] = int(perm_classes[j+1])
                            label_mapping[preferred_classes[j+1]] = int(perm_classes[j])
                    
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

    def get_current_drift_pattern(self):
        """根据当前迭代次数获取应用的漂移模式
        
        除了从预定义的drift_schedule中获取模式外，
        还会根据当前迭代与漂移点的关系，动态添加适当的变换
        
        Returns:
            当前应用的漂移模式
        """
        import copy
        import numpy as np
        
        if not hasattr(self, 'drift_schedule') or self.drift_schedule is None:
            return None
            
        # 查找当前迭代所处的漂移阶段
        current_iter = self.current_iteration
        base_pattern = None
        
        for schedule in self.drift_schedule:
            if schedule['iterations'][0] <= current_iter < schedule['iterations'][1]:
                pattern_name = schedule['pattern']
                if pattern_name is None:
                    base_pattern = None
                else:
                    base_pattern = copy.deepcopy(self.drift_patterns.get(pattern_name, None))
                break
        
        # 检查当前迭代是否接近漂移点
        if hasattr(self, 'drift_points'):
            # 查找最近的漂移点
            closest_drift_point = None
            min_distance = float('inf')
            
            for point in self.drift_points:
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
                if self.drift_type == 'sudden':
                    # 突变漂移在漂移点添加强烈的组合变换
                    if current_iter >= closest_drift_point:
                        base_pattern['transform'] = {
                            'type': 'combined',
                            'params': {
                                'severity': 3,
                                'sequence': ['gaussian_noise', 'contrast', 'brightness']
                            }
                        }
                
                elif self.drift_type == 'gradual':
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
                
                elif self.drift_type == 'recurring':
                    # 周期漂移在漂移点使用随机变换
                    transforms = ['gaussian_noise', 'gaussian_blur', 'contrast', 'brightness', 'elastic_transform']
                    selected = np.random.choice(transforms)
                    base_pattern['transform'] = {
                        'type': selected,
                        'params': {'severity': np.random.randint(1, 4)}
                    }
        
        return base_pattern
