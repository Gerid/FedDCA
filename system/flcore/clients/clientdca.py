import copy
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from .clientbase import Client
from utils.data_utils import read_client_data
from torch.nn.utils import parameters_to_vector, vector_to_parameters


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
            
            self.clf_keys = []  # Will be set by the server if needed, or can be derived if model structure is fixed
            
            # 添加特征分布跟踪属性
            self.current_round_features = {}  # 当前轮次的标签条件特征分布
            self.previous_round_features = {}  # 上一轮次的标签条件特征分布
            self.feature_history = {}  # 特征分布历史记录

        except AttributeError as e:
            print(f"Error initializing client {id}: Missing required attribute - {str(e)}")
            raise
        except ValueError as e:
            print(f"Error initializing client {id}: {str(e)}")
            raise
        except RuntimeError as e:
            print(f"Runtime error initializing client {id}: {str(e)}")
            raise

    def set_clf_keys(self, clf_keys):
        """Sets the classifier keys, typically received from the server."""
        self.clf_keys = clf_keys

    def get_rep_parameters_vector(self):
        """
        获取特征提取器的输出特征表征 (类似FedCCFA的local proto)
        而不是表示层的参数向量
        
        返回:
            聚合后的特征表征向量，形状为 [feature_dim]
        """
        try:
            # 存储所有样本的特征表征
            all_features = []
            
            # 注册临时钩子来捕获特征
            hook_handle = None
            temp_outputs = []
            
            def temp_hook(module, input, output):
                temp_outputs.append(output.detach())
            
            # 尝试找到适合的特征层
            feature_layer = None
            if hasattr(self.model, 'base'):
                feature_layer = self.model.base
            elif hasattr(self.model, 'body'):
                feature_layer = self.model.body
            elif hasattr(self.model, 'features'):
                feature_layer = self.model.features
            elif hasattr(self.model, 'encoder'):
                feature_layer = self.model.encoder
            
            if feature_layer is None:
                print(f"Warning: Client {self.id} - 无法找到适合的特征层")
                return torch.tensor([], device=self.device)
                
            hook_handle = feature_layer.register_forward_hook(temp_hook)
            
            # 加载训练数据进行特征提取
            loader = self.load_train_data()
            
            # 限制处理的批次数以提高效率
            max_batches = 5  # 使用较少的批次进行特征提取
            batch_count = 0
            
            self.model.eval()  # 设置为评估模式
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
                        
                        # 处理不同维度的特征输出
                        if features.dim() > 2:  # 如果是卷积特征 [batch, channels, height, width]
                            # 全局平均池化
                            features = features.mean(dim=tuple(range(2, features.dim())))
                        
                        # 收集所有样本的特征
                        all_features.append(features.cpu())
                        
                        # 清空临时输出列表
                        temp_outputs.clear()
            
            # 移除钩子
            if hook_handle:
                hook_handle.remove()
                
            # 如果没有收集到特征，返回空张量
            if not all_features:
                print(f"Warning: Client {self.id} - 没有收集到任何特征")
                return torch.tensor([], device=self.device)
            
            # 拼接所有特征并计算平均特征表征
            stacked_features = torch.cat(all_features, dim=0)  # [total_samples, feature_dim]
            mean_features = stacked_features.mean(dim=0)  # [feature_dim]
            
            return mean_features.to(self.device)
            
        except Exception as e:
            print(f"Error in get_rep_parameters_vector for client {self.id}: {str(e)}")
            
            # 移除钩子
            if 'hook_handle' in locals() and hook_handle:
                hook_handle.remove()
                
            return torch.tensor([], device=self.device)

    def get_clf_parameters(self):
        """获取分类器 (head) 的参数 state_dict。"""
        if hasattr(self.model, 'head') and self.model.head is not None:
            return {k: v.clone().detach() for k, v in self.model.head.state_dict().items()}
        else:
            # print(f"Client {self.id}: Model has no 'head' attribute for classifier parameters.")
            # Fallback or error handling: if clf_keys are defined, use them
            if self.clf_keys:
                clf_state_dict = {}
                full_state_dict = self.model.state_dict()
                for key_pattern in self.clf_keys:
                    for name, param in full_state_dict.items():
                        if name.startswith(key_pattern):
                            # Adjust key for the standalone head dictionary if needed (e.g., remove 'head.' prefix)
                            adjusted_key = name[len(key_pattern):] if name.startswith(key_pattern) and key_pattern.endswith('.') else name
                            # Simplified: assume key_pattern is like 'head.', so remove it.
                            # This might need adjustment based on how clf_keys are defined and how BaseHeadSplit names parameters.
                            # If clf_keys are direct names from head.named_parameters(), no prefix stripping is needed if we build a new dict.
                            # The current main.py logic for clf_keys adds 'head.', so we should strip it here.
                            if key_pattern == "head." and name.startswith("head."):
                                adjusted_key = name[len("head."):]
                            else:
                                adjusted_key = name  # Or handle other patterns
                            clf_state_dict[adjusted_key] = param.clone().detach()
                if not clf_state_dict:
                    # print(f"Client {self.id}: No classifier parameters found using clf_keys.")
                    return None
                return clf_state_dict
            else:
                # print(f"Client {self.id}: Model has no 'head' and clf_keys are not set. Cannot determine classifier parameters.")
                return None  # Or an empty dict {} if that's preferred for missing classifiers

    def receive_cluster_model(self, cluster_model_head_state_dict):
        """用集群的分类器头参数初始化本地模型的头部。表示层 (body) 不变。"""
        if cluster_model_head_state_dict is None:
            # print(f"Client {self.id} received None for cluster model head state_dict. Retaining current head.")
            return

        if not hasattr(self.model, 'head') or self.model.head is None:
            # print(f"Client {self.id}: Local model has no 'head' to load cluster model into.")
            # This case should ideally not happen if models are structured with BaseHeadSplit
            # Potentially, one could try to re-construct the head if args.head is available
            # but that adds complexity. For now, we assume head exists.
            return
        
        try:
            # Ensure all keys in cluster_model_head_state_dict exist in self.model.head.state_dict()
            # and have matching shapes.
            current_head_state_dict = self.model.head.state_dict()
            sanitized_state_dict = {}
            for key, param_val in cluster_model_head_state_dict.items():
                if key in current_head_state_dict:
                    if current_head_state_dict[key].shape == param_val.shape:
                        sanitized_state_dict[key] = param_val.clone().to(self.device)
                    else:
                        # print(f"Client {self.id}: Shape mismatch for key '{key}' in head. Expected {current_head_state_dict[key].shape}, got {param_val.shape}. Skipping this key.")
                        sanitized_state_dict[key] = current_head_state_dict[key]  # Keep original param
                else:
                    pass  # print(f"Client {self.id}: Key '{key}' from cluster model not found in local model head. Skipping this key.")
            
            # Load only the matching and shape-compatible keys
            self.model.head.load_state_dict(sanitized_state_dict, strict=False)  # strict=False to allow partial loads
            # print(f"Client {self.id}: Loaded cluster model (head) parameters.")

        except (RuntimeError, ValueError) as e:
            print(f"Client {self.id}: Error loading cluster model (head) parameters: {e}. Retaining current head.")
            # Fallback: if loading fails, retain the existing head parameters
            # This can happen if the cluster model head has a different architecture unexpectedly.

    def receive_global_model_body(self, global_model_body_params_vector):
        """用全局聚合的表示层参数更新本地模型的 body。"""
        if global_model_body_params_vector is None or global_model_body_params_vector.numel() == 0:
            # print(f"Client {self.id} received no global model body parameters. Retaining current body.")
            return

        if not hasattr(self.model, 'body') or self.model.body is None:
            # print(f"Client {self.id}: Local model has no 'body' to load global representation into.")
            return
        
        try:
            body_params_template = [param for param in self.model.body.parameters()]
            if not body_params_template:
                # print(f"Client {self.id}: Local model body has no parameters to update.")
                return
            vector_to_parameters(global_model_body_params_vector.to(self.device), body_params_template)
            # print(f"Client {self.id}: Updated local model (body) with global representation parameters.")
        except Exception as e:
            print(f"Client {self.id}: Error updating local model body: {e}")

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

    def update_feature_distributions(self, current_round):
        """
        更新特征分布跟踪
        在每轮通信前调用，用于跟踪特征分布的变化
        """
        try:
            # 将当前轮次的特征存储为上一轮的特征
            self.previous_round_features = copy.deepcopy(self.current_round_features)
            
            # 获取当前轮次的特征分布
            self.current_round_features = self.get_intermediate_outputs_with_labels()
            
            # 存储到历史记录中
            if current_round not in self.feature_history:
                self.feature_history[current_round] = {}
                
            self.feature_history[current_round] = copy.deepcopy(self.current_round_features)
            
            # 可选：限制历史记录的长度以节省内存
            max_history_length = 10  # 保持最近10轮的特征分布
            if len(self.feature_history) > max_history_length:
                oldest_round = min(self.feature_history.keys())
                del self.feature_history[oldest_round]
                
        except Exception as e:
            print(f"Error updating feature distributions for client {self.id}: {str(e)}")

    def get_feature_distribution_stats(self):
        """
        获取特征分布统计信息
        返回当前轮次和上一轮次的特征分布对比
        """
        stats = {}
        
        try:
            if self.current_round_features:
                stats['current_round'] = {}
                for label, features in self.current_round_features.items():
                    if len(features) > 0:
                        stats['current_round'][label] = {
                            'count': len(features),
                            'mean': features.mean(dim=0),
                            'std': features.std(dim=0) if len(features) > 1 else torch.zeros_like(features.mean(dim=0))
                        }
            
            if self.previous_round_features:
                stats['previous_round'] = {}
                for label, features in self.previous_round_features.items():
                    if len(features) > 0:
                        stats['previous_round'][label] = {
                            'count': len(features),
                            'mean': features.mean(dim=0),
                            'std': features.std(dim=0) if len(features) > 1 else torch.zeros_like(features.mean(dim=0))
                        }
        
        except Exception as e:
            print(f"Error getting feature distribution stats for client {self.id}: {str(e)}")
            
        return stats

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
