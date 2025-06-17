import copy
import torch
import torch.nn as nn
import numpy as np
import os
import json # Added for loading superclass map if not already present
import random # Added for choosing fixed concept mapping
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
# Ensure the new drift utility is imported
from utils.concept_drift_utils import drift_dataset, load_superclass_maps, apply_complex_drift 


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
        
        self.train_data_original = read_client_data(self.dataset, self.id, is_train=True)
        self.test_data_original = read_client_data(self.dataset, self.id, is_train=False)
        self.train_data = copy.deepcopy(self.train_data_original) 
        self.test_data = copy.deepcopy(self.test_data_original)

        self.global_test_id = 0

        # --- Complex Concept Drift Attributes ---
        self.complex_drift_config = getattr(args, 'drift_config', None)
        self.superclass_map_path = getattr(args, 'superclass_map_path', None)
        self.superclass_maps = None

        self.fixed_concept_mappings = getattr(args, 'fixed_concept_mappings', None) # --- Fixed Concept Mappings Attributes ---
        self.client_chosen_concept_index = None
        if self.fixed_concept_mappings and len(self.fixed_concept_mappings) > 0: # Check if list is not empty
            # Each client randomly chooses one of the K fixed concept patterns for the entire experiment duration
            self.client_chosen_concept_index = random.randrange(len(self.fixed_concept_mappings))
            print(f"Client {self.id}: Will use fixed concept mapping index {self.client_chosen_concept_index}.") # --- End Fixed Concept Mappings Attributes ---

        if self.superclass_map_path and self.complex_drift_config:
            # Resolve path if relative, similar to main.py
            resolved_path = self.superclass_map_path
            if not os.path.isabs(resolved_path):
                script_dir = os.path.dirname(os.path.realpath(__file__))
                project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..")) 
                resolved_path = os.path.join(project_root, self.superclass_map_path)
            
            print(f"Client {self.id}: Loading superclass map from: {resolved_path}")
            self.superclass_maps = load_superclass_maps(resolved_path)
            if not self.superclass_maps or not self.superclass_maps[0]: 
                print(f"Client {self.id}: Warning - Failed to load valid superclass maps from {resolved_path}. Complex drift may not function as expected.")
                self.superclass_maps = None 
        elif self.complex_drift_config:
            print(f"Client {self.id}: Warning - Superclass map path not provided, or complex_drift_config missing. Complex drift features needing it might be limited.")
        # --- End Complex Concept Drift Attributes ---

        # 添加对概念漂移数据集的支持 (original simpler drift)
        self.current_iteration = 0
        self.drift_data_dir = args.drift_data_dir if hasattr(args, 'drift_data_dir') else None
        self.use_drift_dataset = args.use_drift_dataset if hasattr(args, 'use_drift_dataset') else False
        self.max_iterations = args.max_iterations if hasattr(args, 'max_iterations') else 200
          # 模拟漂移相关参数
        self.simulate_drift = args.simulate_drift if hasattr(args, 'simulate_drift') else False
        self.increment_iteration = args.increment_iteration if hasattr(args, 'increment_iteration') else True
        self.shared_concepts = []
        self.client_concepts = []
        self.current_concept_id = 0 # Initialize current_concept_id to 0 (initial concept)
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
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory

    def update_iteration(self, new_iteration):
        """更新当前迭代计数器，并根据配置应用概念漂移。"""
        if hasattr(self, 'current_iteration'):
            self.current_iteration = new_iteration
        
        if self.complex_drift_config:

            # 统一由 apply_complex_drift 处理所有复杂漂移和概念ID
            empty_superclass_maps = ({}, {}, [], [])
            current_superclass_maps_to_use = self.superclass_maps if self.superclass_maps and self.superclass_maps[0] else empty_superclass_maps

            def process_dataset_for_drift(dataset, dataset_name):
                if not isinstance(dataset, list) or not all(isinstance(item, tuple) and len(item) == 2 for item in dataset):
                    print(f"Client {self.id}: {dataset_name} is not in the expected format (list of (feature, label) tuples). Skipping drift application.")
                    return False, None
                if not dataset:
                    print(f"Client {self.id}: {dataset_name} is empty. Skipping drift application.")
                    return False, None
                try:
                    original_labels_list = [item[1].item() if isinstance(item[1], torch.Tensor) else item[1] for item in dataset]
                except Exception as e:
                    print(f"Client {self.id}: Error extracting labels from {dataset_name}: {e}. Skipping drift application.")
                    return False, None
                modified_labels_list = list(original_labels_list)
                drift_applied, current_concept_id = apply_complex_drift(
                    modified_labels_list,
                    str(self.id),
                    new_iteration,
                    self.complex_drift_config,
                    current_superclass_maps_to_use,
                    fixed_concept_mappings=self.fixed_concept_mappings,
                    client_chosen_concept_index=self.client_chosen_concept_index
                )
                if drift_applied:
                    for i in range(len(dataset)):
                        new_label_tensor = torch.tensor(modified_labels_list[i], dtype=torch.int64)
                        dataset[i] = (dataset[i][0], new_label_tensor)
                    print(f"Client {self.id}: Complex drift applied to {len(modified_labels_list)} samples in {dataset_name} at iteration {new_iteration}.")
                return drift_applied, current_concept_id

            # 只需处理一次，取 train_data 的 concept_id 作为全局当前概念
            _, current_concept_id = process_dataset_for_drift(self.train_data, "train_data")
            self.current_concept_id = current_concept_id if current_concept_id is not None else self.current_concept_id
            process_dataset_for_drift(self.test_data, "test_data")
        # ...existing code...
    def update_concept(self, concept):
        """更新当前使用的概念"""
        self.current_concept = concept
        if concept is not None and 'id' in concept: # This is for general concept objects, might not be used if fixed_concept_mappings is the primary source of truth for ID
            # self.current_concept_id = concept['id'] # This line might conflict with the update in update_iteration
            # Decide the source of truth for current_concept_id. 
            # If update_iteration handles it for fixed_concept_switch, this might be redundant or for other types of concept updates.
            # For now, let update_iteration be the primary place to set current_concept_id during drift.
            pass # Avoid overwriting if update_iteration already set it based on fixed_concept_switch
            
    def load_train_data(self, batch_size=None):
        """加载训练数据，支持概念漂移数据集"""
        if batch_size == None:
            batch_size = self.batch_size
        # return DataLoader(self.train_data, batch_size, drop_last=True, shuffle=True)
        return DataLoader(self.train_data, batch_size, drop_last=True, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def load_test_data(self, batch_size=None):
        """加载测试数据，支持概念漂移数据集"""
        if batch_size == None:
            batch_size = self.batch_size
        # return DataLoader(self.test_data, batch_size, drop_last=False, shuffle=True)
        return DataLoader(self.test_data, batch_size, drop_last=False, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def set_parameters(self, params_dict, part=None):
        """Sets the parameters of the model or parts of it.

        Args:
            params_dict: A state_dict containing parameters to load.
            part (str, optional): Specifies which part of the model to update. 
                                  Can be 'feature_extractor', 'classifier', or None.
                                  If None, loads into the entire model.
        """
        # Convert model object to state_dict if necessary
        if isinstance(params_dict, nn.Module):
            params_dict = params_dict.state_dict()

        if part == 'feature_extractor':
            target_model_part = None
            if hasattr(self.model, 'base') and self.model.base is not None:
                target_model_part = self.model.base
            elif hasattr(self.model, 'body') and self.model.body is not None: # For models like ResNet
                target_model_part = self.model.body
            elif hasattr(self.model, 'features') and self.model.features is not None: # For models like VGG
                target_model_part = self.model.features
            elif hasattr(self.model, 'encoder') and self.model.encoder is not None:
                target_model_part = self.model.encoder
            
            if target_model_part:
                try:
                    target_model_part.load_state_dict(params_dict, strict=True)
                except RuntimeError as e:
                    # print(f"Client {self.id}: Runtime error loading feature_extractor params (strict=True): {e}. Trying with strict=False.")
                    try:
                        target_model_part.load_state_dict(params_dict, strict=False)
                    except Exception as e_non_strict:
                        print(f"Client {self.id}: Error loading feature_extractor params (strict=False): {e_non_strict}")
                        pass # Or handle more gracefully
            # else:
                # print(f"Client {self.id}: Could not identify feature extractor part to set parameters.")
                # Fallback: try to load into the whole model if part is not found, or raise error
                # For now, we'll just skip if the specific part isn't clearly identifiable.
                # Consider adding a more robust way to identify feature extractors if this becomes an issue.
                pass

        elif part == 'classifier':
            if hasattr(self.model, 'head') and self.model.head is not None:
                try:
                    self.model.head.load_state_dict(params_dict, strict=True)
                except RuntimeError as e:
                    # print(f"Client {self.id}: Runtime error loading classifier params (strict=True): {e}. Trying with strict=False.")
                    try:
                        self.model.head.load_state_dict(params_dict, strict=False)
                    except Exception as e_non_strict:
                        # print(f"Client {self.id}: Error loading classifier params (strict=False): {e_non_strict}")
                        pass
            elif hasattr(self, 'clf_keys') and self.clf_keys: # If head attribute doesn't exist, but we know the keys
                # This is more complex as it requires loading parts of the main model's state_dict.
                # For simplicity, this case might require the server to send the full model state_dict
                # if the client doesn't have a distinct 'head' module.
                # Current implementation assumes 'head' exists for 'classifier' part.
                # print(f"Client {self.id}: Model has no 'head' attribute. Classifier part update might be incomplete if relying on clf_keys directly here.")
                # A more robust solution would be to iterate through clf_keys and update self.model.state_dict()
                # This is tricky because params_dict would only contain classifier keys.
                # A simple self.model.load_state_dict(params_dict, strict=False) might work if keys are unique.
                try:
                    self.model.load_state_dict(params_dict, strict=False) # Try loading, hoping keys match
                except Exception as e:
                    # print(f"Client {self.id}: Error loading classifier params using clf_keys and load_state_dict (strict=False): {e}")
                    pass
            # else:
                # print(f"Client {self.id}: Model has no 'head' attribute or clf_keys for classifier parameters.")
                pass
        
        elif part is None: # Load into the entire model
            # The input `params_dict` here is expected to be a full model state_dict
            # The original `set_parameters` took a model object, not a state_dict.
            # This needs to be reconciled. Assuming `params_dict` is a state_dict for consistency.
            try:
                self.model.load_state_dict(params_dict, strict=True)
            except RuntimeError as e:
                # print(f"Client {self.id}: Runtime error loading full model params (strict=True): {e}. Trying with strict=False.")
                try:
                    self.model.load_state_dict(params_dict, strict=False)
                except Exception as e_non_strict:
                    # print(f"Client {self.id}: Error loading full model params (strict=False): {e_non_strict}")
                    pass
        # else:
            # print(f"Client {self.id}: Unknown part '{part}' specified for set_parameters.")
            pass

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
        y_true_auc = [] # Renamed for clarity, this is for AUC
        
        y_true_raw = [] # For F1 and TPR
        y_pred_raw = [] # For F1 and TPR

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

                # For AUC
                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1 # Adjusted for binary case if necessary by original code
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2 and lb.shape[1] > 1: # Ensure correct shape for binary
                    lb = lb[:, :self.num_classes-1] if self.num_classes > 1 else lb # Adapted from potential original logic
                y_true_auc.append(lb)

                # For F1 and TPR
                y_true_raw.append(y.cpu().numpy())
                predictions = torch.argmax(output, dim=1)
                y_pred_raw.append(predictions.cpu().numpy())

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob_flat = np.concatenate(y_prob, axis=0)
        y_true_auc_flat = np.concatenate(y_true_auc, axis=0)
        
        # Calculate AUC
        # Handle potential ValueError if only one class present in y_true_auc_flat for a client
        auc = 0.0
        try:
            if len(np.unique(y_true_auc_flat)) > 1 or (y_true_auc_flat.ndim > 1 and y_true_auc_flat.shape[1] > 1 and np.any(np.sum(y_true_auc_flat, axis=0) > 0)):
                 auc = metrics.roc_auc_score(y_true_auc_flat, y_prob_flat, average="micro")
            else:
                # print(f"Client {self.id}: AUC not computed due to single class in y_true_auc_flat.")
                pass # auc remains 0.0
        except ValueError as e:
            # print(f"Client {self.id}: Could not calculate AUC: {e}")
            pass # auc remains 0.0


        # Calculate F1 and TPR (Weighted)
        y_true_flat_raw = np.concatenate(y_true_raw)
        y_pred_flat_raw = np.concatenate(y_pred_raw)

        f1_weighted = metrics.f1_score(y_true_flat_raw, y_pred_flat_raw, average='weighted', zero_division=0)
        tpr_weighted = metrics.recall_score(y_true_flat_raw, y_pred_flat_raw, average='weighted', zero_division=0) # recall_score is TPR

        return test_acc, test_num, auc, f1_weighted, tpr_weighted

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()

        train_num = 0
        losses = 0.0  # Initialize losses
        y_prob_auc = [] # Initialize for AUC
        y_true_auc = [] # Initialize for AUC

        with torch.no_grad():
            for x, y in trainloader:
                if isinstance(x, list): # Handle cases where x might be a list (e.g. for text data)
                    x = [item.to(self.device) for item in x]
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                output = self.model(x)
                loss = self.loss(output, y)
                
                losses += loss.item() * y.size(0) # Accumulate weighted loss
                train_num += y.shape[0]

                # For AUC calculation on training data (optional)
                # Ensure output is suitable for softmax/sigmoid and label_binarize
                if output.ndim == 2 and output.shape[1] == self.num_classes:
                    y_prob_auc.append(F.softmax(output, dim=1).detach().cpu().numpy())
                    y_true_auc.append(label_binarize(y.detach().cpu().numpy(), classes=range(self.num_classes)))
                elif output.ndim == 2 and output.shape[1] == 1 and self.num_classes == 2: # Binary case with single logit output
                    probs = torch.sigmoid(output).detach().cpu().numpy()
                    y_prob_auc.append(np.hstack((1-probs, probs))) # Convert to [prob_class_0, prob_class_1]
                    y_true_auc.append(label_binarize(y.detach().cpu().numpy(), classes=range(self.num_classes))) # Should be [0, 1]
                # else: # Log a warning or handle other cases if necessary
                    # print(f"Client {self.id}: Output shape {output.shape} or num_classes {self.num_classes} not directly handled for train AUC.")

        if train_num == 0:
            # print(f"Client {self.id}: Warning - train_num is 0 in train_metrics(). Returning zero loss and AUC.")
            return 0.0, 0.0 # loss, auc
            
        avg_loss = losses / train_num
        train_auc = 0.0 # Initialize train_auc
        if len(y_prob_auc) > 0 and len(y_true_auc) > 0:
            try:
                y_prob_all_auc = np.concatenate(y_prob_auc, axis=0)
                y_true_all_auc = np.concatenate(y_true_auc, axis=0)
                if y_prob_all_auc.size > 0 and y_true_all_auc.size > 0:
                    if self.num_classes > 2:
                        # Ensure y_true_all_auc is 2D for multi_class='ovr'
                        if y_true_all_auc.ndim == 1:
                            # This might happen if label_binarize wasn't called or returned 1D array unexpectedly
                            # Attempt to reshape or handle, though ideally label_binarize handles this.
                            # For now, we rely on label_binarize to produce correct shape.
                            pass 
                        if y_true_all_auc.ndim == 2 and y_true_all_auc.shape[1] == self.num_classes and y_prob_all_auc.shape[1] == self.num_classes:
                            train_auc = metrics.roc_auc_score(y_true_all_auc, y_prob_all_auc, average='weighted', multi_class='ovr')
                        # else: # Log shape mismatch
                            # print(f"Client {self.id}: Shape mismatch for multi-class AUC. y_true: {y_true_all_auc.shape}, y_prob: {y_prob_all_auc.shape}")
                    elif self.num_classes == 2:
                        # For binary, ensure y_true_all_auc is 1D (or select the positive class column if binarized)
                        # and y_prob_all_auc corresponds to the positive class probability.
                        true_labels_for_auc = y_true_all_auc
                        if y_true_all_auc.ndim == 2 and y_true_all_auc.shape[1] == 2:
                            true_labels_for_auc = y_true_all_auc[:, 1] # Use the positive class column
                        elif y_true_all_auc.ndim == 2 and y_true_all_auc.shape[1] == 1: # if binarize gave [[0],[1]]
                             true_labels_for_auc = y_true_all_auc.flatten()
                        
                        prob_positive_class = y_prob_all_auc
                        if y_prob_all_auc.ndim == 2 and y_prob_all_auc.shape[1] == 2:
                            prob_positive_class = y_prob_all_auc[:, 1]
                        elif y_prob_all_auc.ndim == 2 and y_prob_all_auc.shape[1] == 1: # if model outputs single probability
                            prob_positive_class = y_prob_all_auc.flatten()

                        if true_labels_for_auc.ndim == 1 and prob_positive_class.ndim == 1:
                           train_auc = metrics.roc_auc_score(true_labels_for_auc, prob_positive_class)
                        # else: # Log shape mismatch
                            # print(f"Client {self.id}: Shape mismatch for binary AUC. y_true: {true_labels_for_auc.shape}, y_prob: {prob_positive_class.shape}")
            except ValueError as e:
                # print(f"Client {self.id}: Could not calculate train AUC: {e}")
                train_auc = 0.0 # Set to 0.0 on error
            except Exception as e: # Catch any other unexpected errors during AUC calculation
                # print(f"Client {self.id}: Unexpected error calculating train AUC: {e}")
                train_auc = 0.0
        
        return avg_loss, train_auc

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

    def apply_drift_transformation(self):
        if self.id % 10 < 3:
            drift_dataset(self.train_data, 1, 2)
            drift_dataset(self.test_data, 1, 2)
            self.global_test_id = 1
        elif self.id % 10 < 6:
            drift_dataset(self.train_data, 3, 4)
            drift_dataset(self.test_data, 3, 4)
            self.global_test_id = 2
        else:
            drift_dataset(self.train_data, 5, 6)
            drift_dataset(self.test_data, 5, 6)
            self.global_test_id = 3

