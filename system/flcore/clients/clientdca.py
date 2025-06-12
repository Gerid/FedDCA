import copy
import torch
import torch.nn as nn
import torch.nn.functional as F # Import F
import numpy as np
import copy
import time
import traceback
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score # Import metrics

from torch.utils.data import DataLoader, TensorDataset
from flcore.clients.clientbase import Client
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
        
        # GMM related parameters (may become less relevant with new profile generation)
        self.gmm_n_components = args.gmm_n_components if hasattr(args, 'gmm_n_components') else 1
        # Ensure gmm_min_samples_for_fitting is at least 1, even if gmm_n_components is 0 or 1
        self.gmm_min_samples_for_fitting = max(1, args.gmm_min_samples_for_fitting if hasattr(args, 'gmm_min_samples_for_fitting') else self.gmm_n_components * 5)

        # Parameters for new label profile generation
        self.num_profile_samples = getattr(args, 'num_profile_samples', 30)
        self.add_noise_to_profiles = getattr(args, 'add_noise_to_profiles', False)
        self.profile_noise_stddev = getattr(args, 'profile_noise_stddev', 0.01)

        # Ablation study flags from args
        self.ablation_no_lp = getattr(args, 'ablation_no_lp', False)
        self.ablation_lp_type = getattr(args, 'ablation_lp_type', 'feature_based') # 'feature_based' or 'class_counts'

        # Loss function for collecting per-sample losses (typically CrossEntropy for classification)
        self.loss_fn_per_sample = nn.CrossEntropyLoss(reduction='none').to(self.device)
        
        self.current_round_training_outputs = [] 
        self.hook_handle = None # To manage the forward hook

        try:
            self.alpha = args.alpha
            self.beta = args.beta
            if args.model is None:
                # This case should be handled: either raise error or have a default model
                raise ValueError("args.model cannot be None during client initialization.") # Added error
            self.model = copy.deepcopy(args.model)
            self.global_model = copy.deepcopy(args.model) # Assuming global_model is still used for some base logic
            
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
            
            # 添加特征分布跟踪属性 (current use might change with new profile generation)
            self.current_round_features = {}
            self.previous_round_features = {}
            self.feature_history = {}

            # Concept ID for ground truth clustering evaluation
            self.current_concept_id = 0 # Default to 0, will be updated by drift simulation

        except AttributeError as e:
            print(f"Error initializing client {id}: Missing required attribute - {str(e)}")
            raise
        except ValueError as e:
            print(f"Error initializing client {id}: {str(e)}")
            raise
        except RuntimeError as e:
            print(f"Runtime error initializing client {id}: {str(e)}")
            raise

    def _get_feature_extractor_module(self):
        """Helper to get the feature extractor module from the model."""
        if hasattr(self.model, 'base') and self.model.base is not None:
            return self.model.base
        elif hasattr(self.model, 'body') and self.model.body is not None: # Common in some architectures
            return self.model.body
        elif hasattr(self.model, 'features') and self.model.features is not None:
            return self.model.features
        elif hasattr(self.model, 'encoder') and self.model.encoder is not None:
            return self.model.encoder
        # print(f"Client {self.id}: Could not identify feature extractor module (tried base, body, features, encoder).")
        return None

    def set_clf_keys(self, clf_keys):
        """Sets the classifier keys, typically received from the server."""
        self.clf_keys = clf_keys

    def get_label_profiles(self):
        """
        Generates Label Profiles based on ablation settings.
        - If ablation_no_lp is True, returns None.
        - If ablation_lp_type is 'class_counts', returns class count based profiles.
        - Otherwise, returns feature-based profiles (N lowest-loss samples per label).
        Returns:
            label_profiles: Dict[label, Tuple[np.ndarray_samples, np.ndarray_losses]] or Dict[label, float] or None
        """
        if self.ablation_no_lp:
            # print(f"Client {self.id}: Ablation: No LPs requested.")
            return None # Return None if LPs are disabled

        if self.ablation_lp_type == 'class_counts':
            # print(f"Client {self.id}: Ablation: Generating class count-based LPs.")
            trainloader = self.load_train_data() # Or use self.train_data if already loaded
            class_counts = {}
            total_samples = 0
            for _, y_batch in trainloader:
                for y_sample in y_batch:
                    label = y_sample.item()
                    class_counts[label] = class_counts.get(label, 0) + 1
                    total_samples += 1
            
            if total_samples == 0:
                return {} # Return empty dict if no samples
            
            # Normalize counts to get a distribution (profile)
            # The server-side distance metric needs to be appropriate for this type of profile.
            # For now, sending raw counts, normalization can be done server-side if needed or here.
            # Or, send as a structure that mimics the feature-based one if server expects (samples, losses)
            # For simplicity, let's send a dict of {label: normalized_count} or {label: count}
            # The server's _estimate_profile_size and distance functions will need to handle this.
            # Let's send normalized counts for now, as it represents a distribution.
            normalized_class_counts = {label: count / total_samples for label, count in class_counts.items()}
            # To fit the expected Tuple[np.ndarray_samples, np.ndarray_losses] structure somewhat,
            # we can store the count as a single-element array in the 'samples' part and None for losses.
            # This is a bit of a hack and might need refinement based on how server processes it.
            class_count_profiles = {}
            for label, norm_count in normalized_class_counts.items():
                # Storing the normalized count as a 1x1 array in the first element of the tuple.
                # The second element (losses) is None.
                class_count_profiles[label] = (np.array([[norm_count]]), None) 
            return class_count_profiles

        # Original feature-based label profile generation
        if not hasattr(self, 'current_round_training_outputs') or not self.current_round_training_outputs:
            # print(f"Client {self.id}: No training outputs collected for feature-based label profiles. Returning empty dict.")
            return {}

        # Group collected (features, loss) by label
        data_by_label = {}
        for item in self.current_round_training_outputs:
            label = item['label']
            if label not in data_by_label:
                data_by_label[label] = []
            data_by_label[label].append({'features': item['features'], 'loss': item['loss']})

        label_profiles = {}

        for label, items in data_by_label.items():
            if not items:
                continue

            items.sort(key=lambda x: x['loss'])
            selected_items = items[:self.num_profile_samples]

            if not selected_items:
                continue

            selected_features_list = [item['features'] for item in selected_items]
            selected_losses_list = [item['loss'] for item in selected_items]
            
            if not selected_features_list:
                continue

            try:
                profile_samples_np = np.array(selected_features_list)
                if profile_samples_np.ndim == 1 and len(selected_features_list) == 1:
                    profile_samples_np = profile_samples_np.reshape(1, -1)
                
                profile_losses_np = np.array(selected_losses_list)

            except ValueError as e:
                # print(f"Client {self.id}, Label {label}: Error stacking features/losses: {e}. Skipping label.")
                continue

            if self.add_noise_to_profiles and profile_samples_np.size > 0:
                noise = np.random.normal(0, self.profile_noise_stddev, profile_samples_np.shape)
                profile_samples_np = profile_samples_np + noise
            
            if profile_samples_np.size > 0:
                 label_profiles[label] = (profile_samples_np, profile_losses_np)
            
        return label_profiles

    def get_clf_parameters(self):
        """获取分类器 (head) 的参数 state_dict, ensuring parameters are on CPU."""
        if hasattr(self.model, 'head') and self.model.head is not None:
            return {k: v.clone().detach().cpu() for k, v in self.model.head.state_dict().items()}
        elif self.clf_keys:
            clf_params = {}
            full_model_state_dict = self.model.state_dict()
            for key in self.clf_keys:
                if key in full_model_state_dict:
                    clf_params[key] = full_model_state_dict[key].clone().detach().cpu()
            return clf_params
        else:
            return {}

    def forward_hook(self, module, input_val, output_val):
        """注册前向传播钩子函数. Appends output_val to self.intermediate_outputs."""
        self.intermediate_outputs.append(output_val)

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()
        
        # Remove any pre-existing hook from previous calls or errors
        if self.hook_handle:
            try:
                self.hook_handle.remove()
            except Exception: # Handle cases where hook might already be invalid
                pass
            self.hook_handle = None
        
        # Register hook for capturing intermediate features if catch_intermediate_output is True
        if self.catch_intermediate_output:
            feature_extractor_module = self._get_feature_extractor_module()
            if feature_extractor_module:
                try:
                    self.hook_handle = feature_extractor_module.register_forward_hook(self.forward_hook)
                except Exception as e:
                    # print(f"Client {self.id}: Error registering hook in train(): {e}")
                    self.hook_handle = None # Ensure it's None if registration failed
            # else:
                # print(f"Client {self.id}: Feature extractor not found in train(), cannot capture intermediate outputs.")
                pass # hook_handle remains None

        start_time = time.time()
        max_local_epochs = self.local_epochs
        
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        # Initialize list to store (features, per-sample loss, label) for this training round
        self.current_round_training_outputs = [] 

        try:
            for epoch in range(max_local_epochs):
                for i, (x, y) in enumerate(trainloader):
                    if self.train_slow: # Simulate heterogeneous clients
                        time.sleep(0.1 * np.abs(np.random.rand()))

                    # Data to device
                    if isinstance(x, list): # Handles models with multiple input tensors
                        x = [item.to(self.device) for item in x]
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)

                    # Clear intermediate_outputs list before this batch's forward pass
                    # self.intermediate_outputs is populated by the hook
                    self.intermediate_outputs.clear() 

                    output = self.model(x) # Forward pass, triggers hook if registered
                    
                    # Calculate per-sample loss using self.loss_fn_per_sample
                    per_sample_loss = self.loss_fn_per_sample(output, y)
                    
                    # Calculate reduced loss for backpropagation (e.g., mean)
                    # The main self.loss attribute might be configured with reduction='mean'
                    # For consistency, if self.loss is defined and suitable, use it. Otherwise, mean().
                    if hasattr(self, 'loss') and callable(self.loss):
                         reduced_loss = self.loss(output, y) # Assumes self.loss handles reduction
                    else:
                         reduced_loss = per_sample_loss.mean()


                    self.optimizer.zero_grad()
                    reduced_loss.backward()
                    # Optional: Gradient clipping or other optimizer steps
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) 
                    self.optimizer.step()

                    # Store features, per-sample loss, and label for profile generation
                    if self.catch_intermediate_output and self.hook_handle and self.intermediate_outputs:
                        # Assuming hook appends one tensor: (batch_size, feature_dim)
                        # And it's the first element in self.intermediate_outputs
                        if self.intermediate_outputs[0] is not None:
                            batch_features_tensor = self.intermediate_outputs[0].detach().clone()
                            
                            # Ensure batch_features_tensor and per_sample_loss have same batch dimension
                            if batch_features_tensor.size(0) == per_sample_loss.size(0):
                                for j in range(batch_features_tensor.size(0)):
                                    self.current_round_training_outputs.append({
                                        'features': batch_features_tensor[j].cpu().numpy(), # Store as NumPy array on CPU
                                        'loss': per_sample_loss[j].item(),       # Store as Python float
                                        'label': y[j].item()                     # Store as Python int
                                    })
                            # else:
                                # print(f"Client {self.id}: Mismatch in batch size between features ({batch_features_tensor.size(0)}) and losses ({per_sample_loss.size(0)}). Skipping data collection for this batch.")
                        # else:
                            # print(f"Client {self.id}: Hook produced None in intermediate_outputs. Skipping data collection for this batch.")
                    # self.intermediate_outputs.clear() # Cleared at the start of the next iteration's feature collection

            # Optional: Learning rate scheduler step (if used per training call rather than per epoch)
            # if hasattr(self, 'learning_rate_scheduler') and self.learning_rate_scheduler:
            #    self.learning_rate_scheduler.step()

        except Exception as e_train: # Catch specific training errors if needed
            print(f"Error during training for client {self.id}: {str(e_train)}")
            print(traceback.format_exc())
            # Re-raise the exception so it can be handled by the caller or system
            raise e_train
        finally:
            # Ensure hook is always removed after training finishes or if an error occurs
            if self.hook_handle:
                try:
                    self.hook_handle.remove()
                except Exception: # Handle cases where hook might already be invalid
                    pass
                self.hook_handle = None
        
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
