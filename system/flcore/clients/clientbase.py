import copy
import torch
import torch.nn as nn
import numpy as np
import os
import json # Added for loading superclass map if not already present
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
        
        # Original data loading
        self.train_data_original = read_client_data(self.dataset, self.id, is_train=True)
        self.test_data_original = read_client_data(self.dataset, self.id, is_train=False)
        # Make copies that can be modified by drift
        self.train_data = copy.deepcopy(self.train_data_original) 
        self.test_data = copy.deepcopy(self.test_data_original)

        self.global_test_id = 0 # This seems related to a simpler drift, might need review

        # --- Complex Concept Drift Attributes ---
        self.complex_drift_config = getattr(args, 'complex_drift_config', None)
        self.superclass_map_path = getattr(args, 'superclass_map_path', None)
        self.superclass_maps = None
        if self.superclass_map_path and self.complex_drift_config:
            print(f"Client {self.id}: Loading superclass map from: {self.superclass_map_path}")
            self.superclass_maps = load_superclass_maps(self.superclass_map_path)
            if not self.superclass_maps[0]: # Check if fine_to_coarse is empty
                print(f"Client {self.id}: Warning - Failed to load valid superclass maps. Complex drift may not function.")
                self.superclass_maps = None # Ensure it's None if loading failed
        else:
            if self.complex_drift_config:
                print(f"Client {self.id}: Warning - Superclass map path not provided, complex drift disabled.")
            # else: not using complex drift, so no message needed
            pass
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
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory

    def update_iteration(self, new_iteration):
        """更新当前迭代计数器，用于概念漂移数据集"""
        if hasattr(self, 'current_iteration'):
            self.current_iteration = new_iteration
        
        # --- Apply Complex Drift at the start of a new iteration/round ---
        if self.complex_drift_config and self.superclass_maps:
            # Apply to training data
            if hasattr(self.train_data, 'targets') and isinstance(self.train_data.targets, list):
                # Ensure targets are modifiable (e.g. list, not tuple)
                train_targets_list = list(self.train_data.targets)
                drift_applied_train = apply_complex_drift(
                    train_targets_list, 
                    self.id, 
                    new_iteration, # Assuming new_iteration is the current global round/epoch
                    self.complex_drift_config, 
                    self.superclass_maps
                )
                if drift_applied_train:
                    self.train_data.targets = train_targets_list # Update with modified targets
                    # print(f"Client {self.id}: Complex drift applied to training data at iteration {new_iteration}.")
            # else:
                # print(f"Client {self.id}: Warning - train_data.targets not found or not a list, cannot apply complex drift.")

            # Apply to testing data
            if hasattr(self.test_data, 'targets') and isinstance(self.test_data.targets, list):
                test_targets_list = list(self.test_data.targets)
                drift_applied_test = apply_complex_drift(
                    test_targets_list, 
                    self.id, 
                    new_iteration, 
                    self.complex_drift_config, 
                    self.superclass_maps
                )
                if drift_applied_test:
                    self.test_data.targets = test_targets_list
                    # print(f"Client {self.id}: Complex drift applied to testing data at iteration {new_iteration}.")
            # else:
                # print(f"Client {self.id}: Warning - test_data.targets not found or not a list, cannot apply complex drift.")
        # --- End Apply Complex Drift ---
            
    def update_concept(self, concept):
        """更新当前使用的概念"""
        self.current_concept = concept
        if concept is not None and 'id' in concept:
            self.current_concept_id = concept['id']
            
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
                        # print(f"Client {self.id}: Error loading feature_extractor params (strict=False): {e_non_strict}")
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

