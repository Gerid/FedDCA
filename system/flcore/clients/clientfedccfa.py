import copy
import torch
import time
import numpy as np
from collections import Counter
from typing import Iterator
from torch.nn import Parameter, MSELoss, CosineSimilarity, CrossEntropyLoss
from torch.utils.data import DataLoader, Subset

from ..clients.clientbase import Client
# from utils.data_utils import read_client_data # Not used directly in this file based on FedCCFA repo

class clientFedCCFA(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.args = args # Retain for easy access to all args
        # self.learning_rate = args.local_learning_rate # learning_rate is set in optimizer directly or per-method
        self.model = copy.deepcopy(args.model) # This should be the wrapped model if feature extraction is needed
        
        self.clf_keys = []  # Classifier layer keys, to be set by server
        
        # Proto criterion based on penalize type from args
        if self.args.penalize == "L2": # Use args.penalize as defined in main.py
            self.proto_criterion = MSELoss().to(self.device)
        else: # Default to contrastive (CrossEntropyLoss)
            self.proto_criterion = CrossEntropyLoss().to(self.device)
        
        self.local_protos = {}
        self.global_protos = [] # Will be populated by server or from local protos
        self.p_clf_params = [] # Personalized classifier parameters
        self.label_distribution = torch.zeros(args.num_classes, dtype=torch.float32) # Use float for probabilities
        self.proto_weight = 0.0
        
        # Optimizer: Base optimizer, specific LRs will be applied in training methods
        # Re-initialize optimizer in specific training methods if LR changes or parts of model are frozen/unfrozen
        
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def set_clf_keys(self, clf_keys):
        """Sets the classifier keys for the client."""
        self.clf_keys = clf_keys
        # print(f"Client {self.id} clf_keys set to: {self.clf_keys}")

    def update_label_distribution(self):
        train_data_loader = self.load_train_data() # Returns DataLoader
        # Access the underlying dataset
        if hasattr(train_data_loader, 'dataset'):
            dataset = train_data_loader.dataset
            if isinstance(dataset, Subset): # Handle Subset case if data was split
                labels = np.array([dataset.dataset.targets[i] for i in dataset.indices])
            elif hasattr(dataset, 'targets'):
                labels = np.array(dataset.targets)
            else: # Fallback for datasets without .targets (e.g. list of tuples)
                labels = np.array([y for _, y in dataset])
        else: # Should not happen if load_train_data is consistent
            print(f"Client {self.id}: Could not determine labels from train_data_loader.")
            return

        if len(labels) == 0:
            print(f"Client {self.id}: No labels found in training data.")
            self.label_distribution.zero_() # Ensure it's zero if no labels
            self.proto_weight = 0.0
            return

        distribution = Counter(labels)
        current_label_distribution = torch.zeros(self.args.num_classes, dtype=torch.float32)
        total_samples = len(labels)
        
        for label, count in distribution.items():
            if 0 <= label < self.args.num_classes:
                current_label_distribution[label] = count
        
        self.label_distribution = current_label_distribution # Store counts

        if total_samples > 0:
            prob = self.label_distribution / total_samples
            # Filter out zero probabilities before log to avoid nan
            non_zero_probs = prob[prob > 0]
            if len(non_zero_probs) > 0:
                entropy = -torch.sum(non_zero_probs * torch.log(non_zero_probs)).item()
            else:
                entropy = 0.0
        else:
            entropy = 0.0
            
        if self.args.gamma != 0:
            self.proto_weight = entropy / self.args.gamma
        else:
            # Use lambda_proto from args, which corresponds to "lambda" in FedCCFA.yaml
            self.proto_weight = self.args.lambda_proto 
    
    def set_rep_params(self, new_params: Iterator[Parameter]):
        rep_params = [param for name, param in self.model.named_parameters() if name not in self.clf_keys]
        for new_param, local_param in zip(new_params, rep_params):
            local_param.data = new_param.data.clone()
    
    def set_clf_params(self, new_params: Iterator[Parameter]):
        clf_params = [param for name, param in self.model.named_parameters() if name in self.clf_keys]
        for new_param, local_param in zip(new_params, clf_params):
            local_param.data = new_param.data.clone()
    
    def set_label_params(self, label, new_params_vector: torch.Tensor): # Expect a vector for a specific label
        # This method needs to correctly map a flat vector of parameters for a *single label*
        # back to the corresponding parts of the classifier's weights and biases for that label.
        # This is complex if the classifier is not a simple nn.Linear layer or if it has multiple parameter tensors.
        # The original FedCCFA code seems to assume clf_keys point to parameters where the 0-th dim is num_classes.
        
        # Assuming clf_keys correspond to parameters like (out_features, in_features) and (out_features)
        # And 'label' selects the row for that class.
        current_idx = 0
        for name, param in self.model.named_parameters():
            if name in self.clf_keys:
                if param.dim() > 0 and param.shape[0] == self.args.num_classes: # e.g., weight/bias of final layer
                    num_elements = param.data[label].numel()
                    param.data[label] = new_params_vector[current_idx : current_idx + num_elements].view_as(param.data[label]).clone()
                    current_idx += num_elements
                # else:
                #     print(f"Warning: Param {name} in clf_keys but not structured per-label as expected.")
        if current_idx != new_params_vector.numel():
            print(f"Warning: Mismatch in number of elements for set_label_params for label {label}.")

    def get_clf_parameters(self):
        clf_params = [param.clone() for name, param in self.model.named_parameters() if name in self.clf_keys]
        return clf_params

    def train_with_protos(self, current_round): # Renamed _round to current_round
        if self.p_clf_params: # Load personalized classifier if available
            self.set_clf_params(self.p_clf_params)
            
        self.model.train()
        trainloader = self.load_train_data()

        # Learning rates from args (as defined in main.py and FedCCFA.yaml)
        clf_lr = self.args.local_learning_rate # Default to local_learning_rate if specific clf_lr not set
        rep_lr = self.args.local_learning_rate # Default to local_learning_rate if specific rep_lr not set
        
        # Check for specific LR for FedCCFA stages if defined (e.g. args.clf_lr, args.rep_lr from a config)
        # The provided FedCCFA.yaml has clf_lr and rep_lr, but main.py doesn't add them as general FedCCFA args yet.
        # Assuming they might be added or are part of a more complex args setup.
        # For now, using local_learning_rate as a base.
        # The original FedCCFA code uses args["clf_lr"] and args["rep_lr"]
        # We should ensure these are available in self.args if used.
        # Let's assume they are part of self.args for now, matching FedCCFA repo.
        if hasattr(self.args, 'clf_lr'):
             clf_lr = self.args.clf_lr
        if hasattr(self.args, 'rep_lr'):
             rep_lr = self.args.rep_lr

        # LR decay (optional, from base args)
        if self.args.learning_rate_decay and hasattr(self.args, 'learning_rate_decay_gamma'):
             # Simple decay based on global rounds, can be more sophisticated
            decay_factor = self.args.learning_rate_decay_gamma ** current_round
            clf_lr *= decay_factor
            rep_lr *= decay_factor
        
        # --- Train Classifier ---
        # print(f"Client {self.id} entering train_with_protos - classifier training phase.")
        # print(f"Client {self.id} self.clf_keys: {self.clf_keys}")
        
        model_param_names_clf_phase = {name: param.requires_grad for name, param in self.model.named_parameters()}
        # print(f"Client {self.id} model parameter names and initial requires_grad (clf phase): {model_param_names_clf_phase}")

        grad_params_to_be_set_clf = []
        # print(f"Client {self.id} Comparing model param names with self.clf_keys ({self.clf_keys}):")
        for name, param in self.model.named_parameters():
            # Detailed check
            is_in_clf_keys = name in self.clf_keys
            # print(f"  Checking param: '{name}' -> In clf_keys? {is_in_clf_keys}")
            if is_in_clf_keys:
                param.requires_grad = True
                grad_params_to_be_set_clf.append(name)
            else:
                param.requires_grad = False
        # print(f"Client {self.id} parameters set to requires_grad=True (clf phase): {grad_params_to_be_set_clf}")

        params_for_optimizer_clf = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        if not params_for_optimizer_clf:
            print(f"Client {self.id} - ERROR: No parameters found for optimizer_clf in train_with_protos.")
            for name, param in self.model.named_parameters():
                print(f"  Param (clf phase): {name}, requires_grad: {param.requires_grad}")
            # Potentially return or raise an error if this is critical and unrecoverable
            # For now, let it proceed to hit the ValueError to keep behavior consistent with current error

        optimizer_clf = torch.optim.SGD(
            params_for_optimizer_clf, # Use the pre-filtered list
            lr=clf_lr,
            weight_decay=self.args.weight_decay if hasattr(self.args, 'weight_decay') else 0,
            momentum=self.args.momentum if hasattr(self.args, 'momentum') else 0.9
        )
        
        for epoch in range(self.args.clf_epochs): # clf_epochs from args
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer_clf.zero_grad()
                outputs = self.model(x) # Assumes model returns logits directly
                loss = self.criterion(outputs, y.long())
                loss.backward()
                optimizer_clf.step()

        # --- Train Representation ---
        # print(f"Client {self.id} entering train_with_protos - representation training phase.")
        # No need to print clf_keys again, but let's see param names and requires_grad status before this phase changes them
        model_param_names_rep_phase_before = {name: param.requires_grad for name, param in self.model.named_parameters()}
        # print(f"Client {self.id} model parameter requires_grad before rep phase changes: {model_param_names_rep_phase_before}")

        grad_params_to_be_set_rep = []
        # print(f"Client {self.id} Comparing model param names with self.clf_keys for rep phase (NOT in clf_keys means True):")
        for name, param in self.model.named_parameters():
            is_NOT_in_clf_keys = name not in self.clf_keys
            # print(f"  Checking param: '{name}' -> NOT in clf_keys? {is_NOT_in_clf_keys}")
            if is_NOT_in_clf_keys:
                param.requires_grad = True
                grad_params_to_be_set_rep.append(name)
            else:
                param.requires_grad = False
        # print(f"Client {self.id} parameters set to requires_grad=True (rep phase): {grad_params_to_be_set_rep}")
        
        params_for_optimizer_rep = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        # if not params_for_optimizer_rep:
            # print(f"Client {self.id} - ERROR: No parameters found for optimizer_rep in train_with_protos.")
            # for name, param in self.model.named_parameters():
                # print(f"  Param (rep phase): {name}, requires_grad: {param.requires_grad}")

        optimizer_rep = torch.optim.SGD(
            params_for_optimizer_rep, # Use the pre-filtered list
            lr=rep_lr,
            weight_decay=self.args.weight_decay if hasattr(self.args, 'weight_decay') else 0,
            momentum=self.args.momentum if hasattr(self.args, 'momentum') else 0.9
        )
        
        cos_sim = CosineSimilarity(dim=1).to(self.device) # Adjusted dim for [N, D] vs [N, D]

        for epoch in range(self.args.rep_epochs): # rep_epochs from args
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer_rep.zero_grad()
                
                # Correctly call the WrappedFeatureExtractor model to get outputs and features
                outputs, features = self.model(x, True)


                loss_sup = self.criterion(outputs, y.long())
                loss_proto = torch.tensor(0.0, device=self.device)

                # Proto loss calculation, aligned with FedCCFA repo logic
                # Start proto loss after some rounds (e.g. round 10 or 20 in FedCCFA repo)
                proto_start_round = 10 # Make this configurable if needed
                if len(self.global_protos) > 0 and current_round >= proto_start_round and self.proto_weight > 0:
                    # Filter out None protos before stacking
                    valid_global_protos_tensors = [p.to(self.device) for p in self.global_protos if p is not None]
                    
                    if not valid_global_protos_tensors: # No valid global protos
                        pass # loss_proto remains 0
                    elif self.args.penalize == "L2":
                        # L2 alignment
                        # Normalize features
                        features_norm = torch.nn.functional.normalize(features, p=2, dim=1)
                        
                        # For each sample, find its corresponding global proto (normalized)
                        target_protos_norm_list = []
                        for i in range(len(y)):
                            label = y[i].item()
                            if 0 <= label < len(self.global_protos) and self.global_protos[label] is not None:
                                target_protos_norm_list.append(torch.nn.functional.normalize(self.global_protos[label].to(self.device), p=2, dim=0))
                            else:
                                # If no proto for this label, use its own feature as target (effectively zero loss for this sample)
                                target_protos_norm_list.append(features_norm[i].detach()) 
                        
                        if target_protos_norm_list:
                            target_protos_norm = torch.stack(target_protos_norm_list)
                            loss_proto = self.proto_criterion(features_norm, target_protos_norm)
                        
                    else: # Contrastive alignment
                        temperature = self.args.temperature # temperature from args
                        
                        # Normalize features
                        features_norm = torch.nn.functional.normalize(features, p=2, dim=1) # (N, D)
                        
                        # Stack and normalize valid global protos
                        stacked_global_protos = torch.stack(valid_global_protos_tensors) # (Num_valid_protos, D)
                        stacked_global_protos_norm = torch.nn.functional.normalize(stacked_global_protos, p=2, dim=1)

                        # Create a mapping from original label index to valid_global_protos_tensors index
                        label_to_valid_idx = {original_idx: valid_idx 
                                              for valid_idx, original_idx in enumerate(
                                                  [i for i, p in enumerate(self.global_protos) if p is not None]
                                              )}
                        
                        # Prepare target labels for contrastive loss (indices in stacked_global_protos_norm)
                        contrastive_target_labels = []
                        valid_sample_indices = [] # Keep track of samples that have a corresponding valid global proto

                        for i in range(len(y)):
                            label = y[i].item()
                            if label in label_to_valid_idx:
                                contrastive_target_labels.append(label_to_valid_idx[label])
                                valid_sample_indices.append(i)
                        
                        if contrastive_target_labels: # If any samples have valid protos
                            contrastive_target_labels = torch.tensor(contrastive_target_labels, dtype=torch.long, device=self.device)
                            
                            # Select features for which we have a target proto
                            features_for_contrastive = features_norm[valid_sample_indices] # (N_valid_samples, D)

                            # Calculate logits: (N_valid_samples, D) @ (D, Num_valid_protos) -> (N_valid_samples, Num_valid_protos)
                            logits = torch.matmul(features_for_contrastive, stacked_global_protos_norm.t())
                            loss_proto = self.proto_criterion(logits / temperature, contrastive_target_labels)
                
                loss = loss_sup + self.proto_weight * loss_proto
                loss.backward()
                optimizer_rep.step()
        
        with torch.no_grad():
            self.model.eval() # Set to eval mode for proto extraction
            self.local_protos = self.get_local_protos()
            # Client should not directly overwrite its global_protos view from local_protos here.
            # Server will aggregate local_protos from clients to form the new global_protos.
            # However, for the very first round or if disconnected, it might initialize its view.
            # FedCCFA paper implies client sends local_protos, server aggregates into global_protos,
            # then server sends global_protos back to clients for next round's proto loss.
            # So, client.global_protos should be what it received from server.
            # The line `self.global_protos = [self.local_protos[label] ...]` from FedCCFA repo's client
            # might be for a scenario where client immediately uses its own protos if no global ones yet.
            # Let's stick to: client computes local_protos, server aggregates them. Client uses server-sent global_protos.

    def set_label_parameters(self, label_idx: int, params_vector: torch.Tensor):
        """ 
        Sets the parameters for a specific label in the classifier layers.
        Assumes params_vector is a 1D tensor containing all parameters for that specific label_idx
        (e.g., a row of weights from the final linear layer and the corresponding bias element).
        This method needs to be called by the server after it aggregates label-specific parameters.
        """
        if not self.clf_keys:
            # print(f"Client {self.id}: clf_keys not set. Cannot set label parameters.")
            return

        current_offset = 0
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.clf_keys:
                    if param.dim() > 0 and param.shape[0] == self.args.num_classes: # e.g., weight/bias of final layer
                        # This assumes the first dimension of the parameter tensor is num_classes
                        # For a weight tensor (e.g., [num_classes, in_features]):
                        #   param.data[label_idx] is a slice of shape [in_features]
                        # For a bias tensor (e.g., [num_classes]):
                        #   param.data[label_idx] is a scalar
                        
                        target_shape = param.data[label_idx].shape
                        num_elements_for_slice = param.data[label_idx].numel()
                        
                        if current_offset + num_elements_for_slice > params_vector.numel():
                            # print(f"Error in client {self.id} set_label_parameters: Not enough elements in params_vector for param {name}, label {label_idx}.")
                            # print(f"  Vector size: {params_vector.numel()}, current_offset: {current_offset}, needed: {num_elements_for_slice}")
                            break # Avoid out-of-bounds error
                            
                        slice_data = params_vector[current_offset : current_offset + num_elements_for_slice]
                        param.data[label_idx] = slice_data.view(target_shape).clone()
                        current_offset += num_elements_for_slice
                    # else:
                        # print(f"Warning: Client {self.id}, Param {name} in clf_keys but not structured per-label as expected or label_idx out of bounds.")
        
        if current_offset != params_vector.numel():
            # print(f"Warning in client {self.id} set_label_parameters: Mismatch in total elements processed for label {label_idx}.")
            # print(f"  Vector size: {params_vector.numel()}, elements processed: {current_offset}")
            pass # This could indicate an issue with how params_vector was constructed or how clf_keys are defined

    def get_local_protos(self):
        """
        Extract local prototypes from the model for each class in the client's training data.
        Returns a dictionary mapping class labels to prototype vectors.
        """
        protos = {}
        trainloader = self.load_train_data()
        
        # Store features for each class
        class_features = {}
        
        self.model.eval() # Ensure model is in eval mode for feature extraction
        with torch.no_grad(): # No gradients needed for feature extraction
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                
                # Correctly call the WrappedFeatureExtractor model to get features
                _, features = self.model(x, return_features=True)
                
                # Group features by class label
                for i, label in enumerate(y):
                    label_idx = label.item()
                    if label_idx not in class_features:
                        class_features[label_idx] = []
                    class_features[label_idx].append(features[i].detach())
        
        # Compute mean of features for each class
        for label, feature_list in class_features.items():
            if feature_list:
                protos[label] = torch.stack(feature_list).mean(dim=0)
        
        return protos
    
    def class_balance_sample(self):
        """
        Creates a class-balanced subset of the client's training data.
        Samples at most a certain number of samples (e.g., 5, as in reference) per class,
        or fewer if a class has less than that number.
        """
        indices = []
        
        try:
            train_loader = self.load_train_data() # Load DataLoader
            if not hasattr(train_loader, 'dataset'):
                print(f"Client {self.id} - ERROR: DataLoader has no 'dataset' attribute in class_balance_sample.")
                return Subset(torch.utils.data.Dataset(), []) # Return empty subset with a dummy dataset

            dataset = train_loader.dataset # Get the underlying dataset

            if isinstance(dataset, Subset):
                # If dataset is a Subset, access targets from the original dataset using subset indices
                if not hasattr(dataset.dataset, 'targets'):
                    # print(f"Client {self.id} - WARNING: Original dataset in Subset has no 'targets' attribute. Trying iteration.")
                    # This case might be slow or problematic if original dataset is large & not easily iterable for labels
                    original_labels = np.array([dataset.dataset[i][1] for i in range(len(dataset.dataset))])
                    labels = original_labels[dataset.indices]
                else:
                    labels = np.array([dataset.dataset.targets[i] for i in dataset.indices])
            elif hasattr(dataset, 'targets'):
                labels = np.array(dataset.targets)
            else:
                # print(f"Client {self.id} - WARNING: Dataset has no 'targets' attribute. Extracting labels by iteration for class_balance_sample.")
                labels = np.array([dataset[i][1] for i in range(len(dataset))]) # Iterate to get labels

        except Exception as e:
            # print(f"Client {self.id} - ERROR accessing labels in class_balance_sample: {e}")
            # Try to return an empty subset with the dataset if available, otherwise a dummy one
            base_dataset_for_empty_subset = dataset if 'dataset' in locals() else torch.utils.data.Dataset()
            return Subset(base_dataset_for_empty_subset, [])

        if len(labels) == 0:
            # print(f"Client {self.id} has an empty training set (or no labels found) for class_balance_sample.")
            return Subset(dataset, []) # dataset should be defined here

        num_per_class = Counter(labels)
        if not num_per_class:
            # print(f"Client {self.id} has no class counts in class_balance_sample.")
            return Subset(dataset, []) # Use the fetched dataset object

        # Determine the number of samples to take per class
        # Reference FedCCFA samples min(smallest_class_size, 5)
        smallest_class_size = num_per_class.most_common()[-1][1]
        sample_count_per_class = min(smallest_class_size, self.args.balanced_samples_per_class if hasattr(self.args, 'balanced_samples_per_class') else 5)


        unique_classes = sorted(num_per_class.keys())

        for class_label in unique_classes:
            class_indices = np.where(labels == class_label)[0]
            
            num_to_sample = min(len(class_indices), sample_count_per_class)

            if num_to_sample > 0:
                chosen_indices = np.random.choice(class_indices, num_to_sample, replace=False)
                indices.extend(chosen_indices)
        
        np.random.shuffle(indices)
        indices = [int(i) for i in indices] # Ensure indices are python int
        return Subset(dataset, indices) # Use the fetched dataset object

    def balance_train(self):
        self.model.train()
        
        # Get balanced dataset and create DataLoader
        balanced_dataset = self.class_balance_sample()

        # if len(balanced_dataset) == 0:
        #     print(f"Client {self.id} - WARNING: No data in balanced_dataset for balance_train. Skipping.")
        #     return

        # Use self.args.batch_size or a specific one for balanced training if available
        current_batch_size = self.args.batch_size 
        balanced_trainloader = DataLoader(
            balanced_dataset,
            batch_size=current_batch_size,
            shuffle=True,
            num_workers=self.args.num_workers if hasattr(self.args, 'num_workers') else 0,
            # drop_last can be important if batch size is larger than dataset size
            drop_last=True if len(balanced_dataset) > current_batch_size else False 
        )
        
        # The rest of the prints for debugging can remain
        # print(f"Client {self.id} entering balance_train with {len(balanced_dataset)} samples.")
        # print(f"Client {self.id} self.clf_keys: {self.clf_keys}")
        
        model_param_names = [name for name, _ in self.model.named_parameters()]
        # print(f"Client {self.id} model parameter names: {model_param_names}") # Can be verbose

        grad_params_to_be_set = []
        for name, param in self.model.named_parameters():
            if name in self.clf_keys:
                param.requires_grad = True
                grad_params_to_be_set.append(name)
            else:
                param.requires_grad = False
        # print(f"Client {self.id} parameters set to requires_grad=True: {grad_params_to_be_set}")
        
        params_for_optimizer = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        # if not params_for_optimizer:
        #     # print(f"Client {self.id} - ERROR: No parameters found for optimizer in balance_train.")
        #     for name, param in self.model.named_parameters():
        #         # print(f"  Param: {name}, requires_grad: {param.requires_grad}, is_leaf: {param.is_leaf}")
        #     return 

        optimizer_balanced_clf = torch.optim.SGD(
            params_for_optimizer,
            lr=self.args.balanced_clf_lr, # Use specific LR for balanced training
            weight_decay=self.args.weight_decay if hasattr(self.args, 'weight_decay') else 0,
            momentum=self.args.momentum if hasattr(self.args, 'momentum') else 0.9
        )
        
        # Use balanced_epochs from args
        epochs_to_run = self.args.balanced_epochs if hasattr(self.args, 'balanced_epochs') else 1 # Default to 1 if not set
        
        for epoch in range(epochs_to_run):
            epoch_loss = 0.0
            num_batches = 0
            for x, y in balanced_trainloader:
                if x.size(0) == 0: continue # Skip if batch is empty (e.g. due to drop_last=True and small dataset)
                x, y = x.to(self.device), y.to(self.device)
                optimizer_balanced_clf.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y.long())
                loss.backward()
                # Gradient clipping (optional, but good practice)

                optimizer_balanced_clf.step()
                epoch_loss += loss.item()
                num_batches += 1
            # if num_batches > 0:
            #     print(f"Client {self.id} Balance Train Epoch {epoch+1}/{epochs_to_run}, Avg Loss: {epoch_loss/num_batches:.4f}")
            # else:
            #     print(f"Client {self.id} Balance Train Epoch {epoch+1}/{epochs_to_run}, No batches processed.")

