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
        self.simulate_drift = args.simulate_drift if hasattr(args, 'simulate_drift') else False
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

    def load_train_data(self, batch_size=None):
        """重写加载训练数据方法，以支持标准CIFAR-100上的模拟漂移
        
        根据当前迭代次数和预设的漂移模式修改数据标签和分布
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        # 首先获取原始训练数据
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        
        # 如果启用了标准数据集上的模拟漂移
        if hasattr(self, 'simulate_drift') and self.simulate_drift:
            # 应用漂移转simulate_drift换
            train_data = self.apply_drift_transformation(train_data)
            
            # 每次调用此方法时递增迭代计数
            if hasattr(self, 'increment_iteration') and self.increment_iteration:
                self.current_iteration = min(self.current_iteration + 1, self.max_iterations - 1)
                
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)
        
    def apply_drift_transformation(self, dataset):
        """根据当前迭代和预设模式对数据集应用概念漂移变换
        
        Args:
            dataset: 原始数据集
            
        Returns:
            transformed_dataset: 应用了漂移变换的数据集
        """
        import torch
        from torch.utils.data import TensorDataset
        import numpy as np
        import copy
        
        # 如果尚未初始化漂移参数，则初始化
        if not hasattr(self, 'drift_patterns') or self.drift_patterns is None:
            self.initialize_drift_patterns()
            
        # 获取当前应用的漂移模式
        current_pattern = self.get_current_drift_pattern()
        if current_pattern is None:
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
                
        # 应用协变量漂移 (covariate shift)
        if 'transform' in current_pattern:
            transform_type = current_pattern['transform']['type']
            params = current_pattern['transform'].get('params', {})
            
            if transform_type == 'noise':
                # 添加高斯噪声
                noise_level = params.get('level', 0.1)
                noise = torch.randn_like(all_data) * noise_level
                all_data = torch.clamp(all_data + noise, 0, 1)
                
            elif transform_type == 'rotation':
                # 旋转图像
                angle = params.get('angle', 30)
                # 这里需要实现旋转逻辑，可能需要使用torchvision.transforms
                pass
                
            elif transform_type == 'brightness':
                # 亮度调整
                factor = params.get('factor', 0.5)
                all_data = torch.clamp(all_data * factor, 0, 1)
                
        # 返回转换后的数据集
        return TensorDataset(all_data, all_labels)
        
    def initialize_drift_patterns(self):
        """初始化概念漂移模式
        
        定义各种不同类型的漂移模式，可以根据迭代次数进行切换
        """
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
                'transform': {'type': 'noise', 'params': {'level': 0.1}}
            },
            'covariate_brightness': {
                'transform': {'type': 'brightness', 'params': {'factor': 0.7}}
            },
            
            # 4. 组合漂移
            'combined_drift': {
                'label_mapping': {0: 5, 5: 0, 10: 15, 15: 10},
                'class_probs': {str(i): (2.0 if i % 10 == 0 else 0.8)/100 for i in range(100)},
                'transform': {'type': 'noise', 'params': {'level': 0.05}}
            }
        }
        
        # 定义漂移变化时间表 - 每隔特定迭代次数应用不同的漂移模式
        self.drift_schedule = [
            {'iterations': [0, 50], 'pattern': None},  # 前50轮没有漂移
            {'iterations': [50, 100], 'pattern': 'label_drift_mild'},
            {'iterations': [100, 150], 'pattern': 'prior_drift_mild'},
            {'iterations': [150, 200], 'pattern': 'covariate_noise'},
            {'iterations': [200, 250], 'pattern': 'label_drift_moderate'},
            {'iterations': [250, 300], 'pattern': 'combined_drift'},
            {'iterations': [300, float('inf')], 'pattern': 'label_drift_severe'}
        ]
        
    def get_current_drift_pattern(self):
        """根据当前迭代次数获取应用的漂移模式"""
        if not hasattr(self, 'drift_schedule') or self.drift_schedule is None:
            return None
            
        # 查找当前迭代所处的漂移阶段
        current_iter = self.current_iteration
        for schedule in self.drift_schedule:
            if schedule['iterations'][0] <= current_iter < schedule['iterations'][1]:
                pattern_name = schedule['pattern']
                if pattern_name is None:
                    return None
                return self.drift_patterns.get(pattern_name, None)
                
        return None
