import copy
import torch
import time
import numpy as np
import os
from flcore.servers.serverbase import Server
from flcore.clients.clientfedccfa import clientFedCCFA # Ensure this client is compatible
from threading import Thread
from torch.nn.utils import parameters_to_vector, vector_to_parameters
# import matplotlib.pyplot as plt # Not used
from sklearn.cluster import DBSCAN
# import traceback # Not used
import wandb


class FedCCFA(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        
        # Initialize client type
        self.set_slow_clients()
        self.set_clients(clientFedCCFA) # Corrected: was set_clients(clientFedCCFA)
        # self.set_slow_clients() # This might be handled by ServerBase or not needed if join_ratio controls participation

        # Classifier layer keys - will be set after model is loaded, typically in main.py
        self.clf_keys = [] 
        
        # Global prototypes and performance tracking (from reference)
        self.global_protos = [None] * args.num_classes # Initialize with Nones
        # self.prev_rep_norm = 0 # Not in reference FedCCFAServer
        # self.prev_clf_norm = 0 # Not in reference FedCCFAServer
        # self.rep_norm_scale = 0 # Not in reference FedCCFAServer
        # self.clf_norm_scale = 0 # Not in reference FedCCFAServer
        self.start_time = time.time() # Start time for the server
        
        self.Budget = [] # Kept from original
        self.client_data_size = {}  # For weighted aggregation

        # Ensure essential args are present (defaults from reference FedCCFA.yaml if not)
        if not hasattr(args, 'eps'):
            args.eps = 0.5 
        if not hasattr(args, 'clustered_protos'): # This arg is used on client side in reference
            args.clustered_protos = False 
        if not hasattr(args, 'oracle'):
            args.oracle = False
        if not hasattr(args, 'weights'): # For aggregation weighting
            args.weights = "label" # "uniform" or "label"
        if not hasattr(args, 'gamma'): # For client-side proto_weight calculation
             args.gamma = 0.0 # Default to disable adaptive proto_weight if not set
        if not hasattr(args, 'lambda_proto'): # Renamed from 'lambda' to avoid keyword clash
             args.lambda_proto = 1.0 # Default proto loss weight if gamma is 0


        print(f"\\nFedCCFA Server initialized.")
        print(f"Participation ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"Aggregation weights type: {self.args.weights}")
        print(f"Oracle merging: {self.args.oracle}")
        print(f"DBSCAN eps: {self.args.eps}")

    def get_client_data_size(self, clients):
        """Records the training data size for each client."""
        for client in clients:
            self.client_data_size[client.id] = client.train_samples

    # send_params is inherited from ServerBase, sends the whole model.

    def send_rep_params(self, clients_to_send_to):
        """Sends representation layer parameters to specified clients."""
        if not self.clf_keys:
            print("Warning: clf_keys not set in server. Cannot send rep_params.")
            return

        with torch.no_grad():
            rep_params = [param.detach().clone() for name, param in self.global_model.named_parameters()
                          if name not in self.clf_keys]
        
        for client in clients_to_send_to:
            client.set_rep_params(rep_params)

    def aggregate_rep(self, selected_clients):
        """Aggregates representation layer parameters from selected clients."""
        if not self.clf_keys:
            print("Warning: clf_keys not set in server. Cannot aggregate rep_params.")
            return

        total_data_size = 0
        base_params = [param for name, param in self.global_model.named_parameters() if name not in self.clf_keys]
        aggregated_rep_vector = torch.zeros_like(parameters_to_vector(base_params), device=self.device)

        for client in selected_clients:
            client_data_size = self.client_data_size.get(client.id, 0)
            if client_data_size == 0:
                print(f"Warning: Client {client.id} has 0 data size. Skipping for rep aggregation.")
                continue
            
            total_data_size += client_data_size
            client_rep_params = [param for name, param in client.model.named_parameters() 
                                 if name not in self.clf_keys]
            aggregated_rep_vector += parameters_to_vector(client_rep_params).to(self.device) * client_data_size
        
        if total_data_size > 0:
            aggregated_rep_vector /= total_data_size
        else:
            print("Warning: Total data size is 0. Rep aggregation resulted in zero vector.")
            # Keep original global_model rep_params if no client contributed
            return 

        # Load aggregated vector back to global model's representation layers
        vector_to_parameters(aggregated_rep_vector, base_params)

    def aggregate_protos(self, selected_clients, current_round=None):
        """Aggregates global prototypes based on clients' local prototypes and label distributions."""
        # Ensure self.global_protos is initialized correctly
        if not self.global_protos or len(self.global_protos) != self.args.num_classes:
            # Assuming prototypes are feature vectors, get dim from a client or model
            # This is a fallback, ideally proto dim is known.
            # For now, let's assume client.local_protos gives correctly shaped tensors.
            # If a client has no proto for a class, it might be an issue.
            # The reference initializes self.global_protos = [None] * num_classes
            # and then fills it.
            pass # Rely on client.local_protos to provide shape.

        aggregate_proto_dict = {} # Stores sum of weighted protos
        label_total_weight = {} # Stores sum of weights for each label

        for client in selected_clients:
            if not hasattr(client, 'local_protos') or not client.local_protos:
                # print(f"Client {client.id} has no local_protos. Skipping for proto aggregation.")
                continue
            
            client_label_dist = client.label_distribution # Should be a list/tensor of counts or proportions

            for label_idx, local_proto in client.local_protos.items(): # client.local_protos is a dict {label: proto_tensor}
                if local_proto is None:
                    continue

                current_weight = 0
                if self.args.weights == "uniform":
                    current_weight = 1.0
                elif self.args.weights == "label":
                    # Ensure client_label_dist is accessible and correct
                    if label_idx < len(client_label_dist):
                        current_weight = client_label_dist[label_idx].item() if torch.is_tensor(client_label_dist[label_idx]) else client_label_dist[label_idx]
                    else:
                        # print(f"Warning: Label index {label_idx} out of bounds for client {client.id} label distribution. Using 0 weight.")
                        current_weight = 0
                else: # Default to uniform if not specified
                    current_weight = 1.0

                if current_weight == 0: # Do not include if weight is zero
                    continue

                if label_idx not in aggregate_proto_dict:
                    aggregate_proto_dict[label_idx] = local_proto.clone().to(self.device) * current_weight
                    label_total_weight[label_idx] = current_weight
                else:
                    aggregate_proto_dict[label_idx] += local_proto.to(self.device) * current_weight
                    label_total_weight[label_idx] += current_weight
        
        # Update global prototypes
        new_global_protos = [None] * self.args.num_classes
        for label_idx in range(self.args.num_classes):
            if label_idx in aggregate_proto_dict and label_total_weight[label_idx] > 0:
                new_global_protos[label_idx] = (aggregate_proto_dict[label_idx] / label_total_weight[label_idx])
            elif self.global_protos and self.global_protos[label_idx] is not None: # Keep old if no update
                new_global_protos[label_idx] = self.global_protos[label_idx] 
                # print(f"Proto for label {label_idx} kept from previous round.")
            # else:
                # print(f"No prototype aggregated for label {label_idx} in round {current_round}.")


        self.global_protos = new_global_protos

        # Wandb logging for prototypes
        if self.args.use_wandb and wandb.run is not None and current_round is not None:
            # Check if there's anything to log
            valid_protos_to_log = [p for p in self.global_protos if p is not None]
            if not valid_protos_to_log:
                # print(f"No valid global prototypes to log for round {current_round}.")
                return

            try:
                # Handle None for saving, e.g., by using a zero tensor of expected shape if possible
                # Or, filter out None values before saving, though this changes the structure.
                # For now, let's assume we need to know the proto dimension.
                # If all are None, we can't infer dim.
                proto_dim = None
                for p in self.global_protos:
                    if p is not None:
                        proto_dim = p.shape[0] # Assuming 1D feature vector
                        break
                
                if proto_dim is None and valid_protos_to_log: # Should not happen if valid_protos_to_log is not empty
                     proto_dim = valid_protos_to_log[0].shape[0]


                protos_to_save_list = []
                for p_idx, p in enumerate(self.global_protos):
                    if p is not None:
                        protos_to_save_list.append(p.cpu())
                    elif proto_dim is not None : # If we know the dim, save a placeholder
                        # print(f"Warning: Global proto for label {p_idx} is None in round {current_round}. Saving zeros.")
                        protos_to_save_list.append(torch.zeros(proto_dim, device='cpu'))
                    # If proto_dim is None and p is None, we skip (should not happen if list is not empty)

                if not protos_to_save_list: # If after processing, list is empty
                    # print(f"No prototypes to save to wandb for round {current_round} after handling Nones.")
                    return

                # Save as a list of tensors or a stacked tensor if all have same shape and are not None
                # For simplicity, saving as a list of tensors (which torch.save handles)
                
                protos_dir = os.path.join(self.save_folder_name, "protos")
                if not os.path.exists(protos_dir):
                    os.makedirs(protos_dir, exist_ok=True)
                
                protos_filename = f"global_protos_round_{current_round}.pt"
                protos_filepath = os.path.join(protos_dir, protos_filename)
                torch.save(protos_to_save_list, protos_filepath)

                artifact_name = f'{self.args.wandb_run_name_prefix}_global_protos'
                protos_artifact = wandb.Artifact(
                    artifact_name,
                    type='global-prototypes',
                    description=f'Global prototypes for FedCCFA at round {current_round}',
                    metadata={'round': current_round, 'algorithm': self.algorithm, 'num_classes': self.args.num_classes, 'num_protos_saved': len(protos_to_save_list)}
                )
                protos_artifact.add_file(protos_filepath, name=protos_filename)
                wandb.log_artifact(protos_artifact, aliases=[f'protos_round_{current_round}', 'latest_protos'])
                # print(f"Global prototypes for round {current_round} saved to wandb.")
            except Exception as e:
                print(f"Error saving global prototypes to wandb: {e}")


    @staticmethod
    def madd(vecs):
        """Computes MADD distance matrix based on cosine similarity differences."""
        def cos_sim(a, b):
            # Add epsilon for numerical stability, as in reference client
            return a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8) 
        
        num = len(vecs)
        if num <= 2: # Cannot compute MADD for less than 3 vectors as per formula (div by num-2)
            # Return a zero matrix or handle as an edge case appropriate for the caller
            # print("MADD calculation: Not enough vectors (<=2) to compute meaningful MADD. Returning zero matrix.")
            return np.zeros((num, num))

        res = np.zeros((num, num))
        for i in range(num):
            for j in range(i + 1, num):
                dist = 0.0
                for z in range(num):
                    if z == i or z == j:
                        continue
                    dist += np.abs(cos_sim(vecs[i], vecs[z]) - cos_sim(vecs[j], vecs[z]))
                
                # Original PFL-Non-IID code had a check `if num > 2`, which is implicitly true here
                # due to the check at the beginning of the function.
                res[i][j] = res[j][i] = dist / (num - 2) 
        return res

    def merge_classifiers(self, clf_params_dict_from_clients):
        """
        Merges classifiers for each label based on parameter distance using DBSCAN.
        Args:
            clf_params_dict_from_clients: Dict {client_id: [list of classifier layer (Param) tensors]}
        Returns:
            Dict {label_idx: [[list of client_ids in cluster1], [list of client_ids in cluster2], ...]}
        """
        if not self.clf_keys:
            print("Warning: clf_keys not set in server. Cannot merge classifiers.")
            return {label: [] for label in range(self.args.num_classes)}

        client_ids_list = list(clf_params_dict_from_clients.keys())
        # Ensure client_clf_params are correctly extracted (vectorized per label)
        # The input clf_params_dict_from_clients should contain the actual Parameter objects or their data.
        # The reference FedCCFA.py calls client.get_clf_parameters() which returns a list of Parameter objects.

        label_merged_dict = {}
        for label_idx in range(self.args.num_classes):
            params_for_label_list = [] # List of numpy vectors, one per client for this label_idx
            valid_client_indices_for_label = [] # Store original indices of clients who have params for this label

            for i, client_id in enumerate(client_ids_list):
                client_clf_layers = clf_params_dict_from_clients[client_id] # List of [weight_tensor, bias_tensor]
                
                # Extract parameters for the current label_idx from each layer
                # Assumes classifier layers have output units equal to num_classes (e.g., Linear layer)
                # Weight shape: [num_classes, feature_dim], Bias shape: [num_classes]
                # We need the parameters corresponding to *this* specific label_idx
                
                per_client_label_params_parts = []
                try:
                    for layer_param in client_clf_layers: # e.g., layer_param is weight or bias tensor
                        if layer_param.dim() > 1: # Typically weights
                             # Ensure label_idx is within bounds
                            if label_idx < layer_param.shape[0]:
                                per_client_label_params_parts.append(layer_param[label_idx].detach().cpu().numpy().flatten())
                            else:
                                # This client's classifier doesn't have this label_idx (e.g. smaller output layer than num_classes)
                                # This should ideally not happen if models are consistent.
                                # print(f"Warning: Label index {label_idx} out of bounds for client {client_id}'s classifier layer (shape {layer_param.shape}). Skipping this layer for this client.")
                                raise IndexError # To be caught below
                        elif layer_param.dim() == 1: # Typically biases
                            if label_idx < layer_param.shape[0]:
                                per_client_label_params_parts.append(layer_param[label_idx].detach().cpu().numpy().flatten()) # flatten just in case
                            else:
                                # print(f"Warning: Label index {label_idx} out of bounds for client {client_id}'s classifier bias (shape {layer_param.shape}). Skipping this bias for this client.")
                                raise IndexError # To be caught below
                    
                    if not per_client_label_params_parts:
                        # print(f"Client {client_id} had no classifier parameters for label {label_idx}")
                        continue # Skip this client for this label if no params found

                    params_for_label_list.append(np.hstack(per_client_label_params_parts))
                    valid_client_indices_for_label.append(i) # Store original index from client_ids_list

                except IndexError:
                    # print(f"Skipping client {client_id} for label {label_idx} due to parameter dimension mismatch.")
                    continue # Skip this client for this label if params are not as expected
            
            if len(params_for_label_list) < 2 : # Need at least 2 clients to form a cluster or compare
                label_merged_dict[label_idx] = [] # No merges possible
                # print(f"Label {label_idx}: Not enough clients ({len(params_for_label_list)}) with valid params to perform clustering.")
                continue

            params_matrix = np.array(params_for_label_list)
            dist_matrix = self.madd(params_matrix)
            
            # DBSCAN clustering
            # print(f"Label {label_idx} dist matrix for DBSCAN:\n{np.round(dist_matrix, 3)}")
            clustering = DBSCAN(eps=self.args.eps, min_samples=1, metric="precomputed") # min_samples=1 means every point can be a core point
            try:
                cluster_labels = clustering.fit_predict(dist_matrix)
            except Exception as e:
                print(f"Error during DBSCAN fitting for label {label_idx}: {e}")
                label_merged_dict[label_idx] = []
                continue

            # print(f"Label {label_idx} DBSCAN cluster labels: {cluster_labels}")

            # Process clustering results
            merged_client_id_groups = []
            unique_cluster_labels = set(cluster_labels)
            
            original_client_ids_array = np.array(client_ids_list)[valid_client_indices_for_label]


            for cl_label in unique_cluster_labels:
                if cl_label == -1: # Noise points by DBSCAN, treat as singleton clusters if min_samples > 1
                                  # With min_samples=1, -1 should not typically occur unless eps is very small.
                    # print(f"Label {label_idx}: Noise points found by DBSCAN. Treating as singletons.")
                    # noise_indices = np.where(cluster_labels == -1)[0]
                    # for idx in noise_indices:
                    #    merged_client_id_groups.append([original_client_ids_array[idx]]) # Each noise point is its own group
                    continue # Or ignore noise points if they shouldn't form groups

                current_cluster_indices = np.where(cluster_labels == cl_label)[0]
                
                # Only consider actual merges (groups of size > 1)
                # The reference FedCCFA.py (methods/FedCCFA.py, line 76) includes `if len(indices) > 1:`
                # However, the subsequent logic iterates through these groups to aggregate.
                # If a "group" has only one client, it means its classifier for that label is unique.
                # The aggregation for a single-client group would just be its own parameters.
                # Let's include all groups, even singletons, as the aggregation logic will handle it.
                # The reference `FedCCFA.py` (line 79) `merged_identities` seems to be a list of lists of client_ids.
                
                # if len(current_cluster_indices) > 0: # Always true if cl_label is in unique_cluster_labels and not -1
                ids_in_cluster = original_client_ids_array[current_cluster_indices].tolist()
                ids_in_cluster.sort() # For consistency
                merged_client_id_groups.append(ids_in_cluster)
            
            label_merged_dict[label_idx] = merged_client_id_groups
            # print(f"Label {label_idx} merged groups: {merged_client_id_groups}")
        
        return label_merged_dict

    def oracle_merging(self, current_round, client_ids_this_round):
        """Oracle merging strategy based on reference FedCCFA.py (hardcoded for 10 classes)."""
        num_classes = self.args.num_classes
        merged_dict = {}

        if num_classes == 10: # Specific logic from reference for num_classes=10
            if current_round < 100:
                for label in range(num_classes):
                    merged_dict[label] = [client_ids_this_round] # All clients in one group for each label
            else: # Drift scenario from reference
                merged_dict[0] = [client_ids_this_round]
                merged_dict[1] = [[_id for _id in client_ids_this_round if 0 <= _id % 10 < 3], [_id for _id in client_ids_this_round if _id % 10 >= 3]]
                merged_dict[2] = [[_id for _id in client_ids_this_round if 0 <= _id % 10 < 3], [_id for _id in client_ids_this_round if _id % 10 >= 3]]
                merged_dict[3] = [[_id for _id in client_ids_this_round if 3 <= _id % 10 < 6], [_id for _id in client_ids_this_round if not (3 <= _id % 10 < 6)]]
                merged_dict[4] = [[_id for _id in client_ids_this_round if 3 <= _id % 10 < 6], [_id for _id in client_ids_this_round if not (3 <= _id % 10 < 6)]]
                merged_dict[5] = [[_id for _id in client_ids_this_round if _id % 10 >= 6], [_id for _id in client_ids_this_round if _id % 10 < 6]]
                merged_dict[6] = [[_id for _id in client_ids_this_round if _id % 10 >= 6], [_id for _id in client_ids_this_round if _id % 10 < 6]]
                merged_dict[7] = [client_ids_this_round]
                merged_dict[8] = [client_ids_this_round]
                merged_dict[9] = [client_ids_this_round]
                # Filter out empty groups that might result from client selection
                for label in range(num_classes):
                    merged_dict[label] = [group for group in merged_dict[label] if group]

        else: # Generic oracle: all clients in one group if no specific logic
            # print(f"Oracle merging: Using generic strategy (all clients in one group) for num_classes={num_classes}")
            for label in range(num_classes):
                merged_dict[label] = [client_ids_this_round]
        
        return merged_dict

    def aggregate_label_params(self, label_idx, clients_in_group):
        """
        Aggregates classifier parameters for a specific label from a group of clients.
        Args:
            label_idx: The specific label index.
            clients_in_group: List of client objects in the same cluster for this label.
        Returns:
            A 1D tensor representing the aggregated parameters for this label (e.g., concatenated weight row and bias element).
        """
        if not self.clf_keys:
            print("Warning: clf_keys not set. Cannot aggregate label_params.")
            return None
        if not clients_in_group:
            # print(f"Warning: No clients in group for label {label_idx} to aggregate params.")
            return None

        # Determine the shape of label-specific parameters from the first client
        # This assumes all clients in the group have compatible classifier structures.
        first_client = clients_in_group[0]
        param_template_parts = []
        try:
            for name, param_layer in first_client.model.named_parameters():
                if name in self.clf_keys:
                    if param_layer.dim() > 1 and label_idx < param_layer.shape[0]: # Weights
                        param_template_parts.append(param_layer[label_idx].detach().clone())
                    elif param_layer.dim() == 1 and label_idx < param_layer.shape[0]: # Biases
                        param_template_parts.append(param_layer[label_idx].detach().clone())
                    # else: param doesn't have this label_idx, skip (should be consistent across group)
            
            if not param_template_parts:
                # print(f"Could not determine parameter template for label {label_idx} from client {first_client.id}")
                return None
            aggregated_params_vector = torch.zeros_like(parameters_to_vector(param_template_parts), device=self.device)
        except IndexError:
            # print(f"Error: Client {first_client.id} classifier params incompatible with label {label_idx}")
            return None


        total_weight_for_label = 0

        for client in clients_in_group:
            current_client_label_params_parts = []
            valid_client_for_label = True
            for name, param_layer in client.model.named_parameters():
                if name in self.clf_keys:
                    try:
                        if param_layer.dim() > 1: # Weights
                            current_client_label_params_parts.append(param_layer[label_idx].detach().clone())
                        elif param_layer.dim() == 1: # Biases
                            current_client_label_params_parts.append(param_layer[label_idx].detach().clone())
                    except IndexError:
                        # print(f"Warning: Client {client.id} params incompatible for label {label_idx}. Skipping this client for this label param aggregation.")
                        valid_client_for_label = False
                        break 
            
            if not valid_client_for_label or not current_client_label_params_parts:
                continue # Skip this client if params are missing or incompatible

            client_params_vector = parameters_to_vector(current_client_label_params_parts).to(self.device)
            
            current_weight = 0
            if self.args.weights == "uniform":
                current_weight = 1.0
            elif self.args.weights == "label":
                if hasattr(client, 'label_distribution') and label_idx < len(client.label_distribution):
                    current_weight = client.label_distribution[label_idx].item() if torch.is_tensor(client.label_distribution[label_idx]) else client.label_distribution[label_idx]
                else:
                    # print(f"Warning: Client {client.id} missing label_distribution for label {label_idx}. Using 0 weight for param agg.")
                    current_weight = 0
            else: # Default to uniform
                current_weight = 1.0
            
            if current_weight > 0:
                aggregated_params_vector += client_params_vector * current_weight
                total_weight_for_label += current_weight
        
        if total_weight_for_label > 0:
            aggregated_params_vector /= total_weight_for_label
            return aggregated_params_vector
        else:
            # print(f"Warning: Total weight for label {label_idx} is 0. Returning None for aggregated_params_vector.")
            # Fallback: could return the average of the first client's params or zeros of correct shape if template is known
            return None # Or torch.zeros_like(parameters_to_vector(param_template_parts))

    def aggregate_label_protos(self, label_idx, clients_in_group):
        """
        Aggregates prototypes for a specific label from a group of clients.
        Args:
            label_idx: The specific label index.
            clients_in_group: List of client objects in the same cluster for this label.
        Returns:
            A tensor for the aggregated prototype for this label, or None.
        """
        if not clients_in_group:
            # print(f"Warning: No clients in group for label {label_idx} to aggregate protos.")
            return None

        aggregated_proto = None
        total_weight_for_label = 0
        proto_initialized = False

        for client in clients_in_group:
            if hasattr(client, 'local_protos') and label_idx in client.local_protos and client.local_protos[label_idx] is not None:
                client_proto = client.local_protos[label_idx].to(self.device)
                
                current_weight = 0
                if self.args.weights == "uniform":
                    current_weight = 1.0
                elif self.args.weights == "label":
                    if hasattr(client, 'label_distribution') and label_idx < len(client.label_distribution):
                         current_weight = client.label_distribution[label_idx].item() if torch.is_tensor(client.label_distribution[label_idx]) else client.label_distribution[label_idx]
                    else:
                        # print(f"Warning: Client {client.id} missing label_distribution for label {label_idx}. Using 0 weight for proto agg.")
                        current_weight = 0
                else: # Default to uniform
                    current_weight = 1.0

                if current_weight > 0:
                    if not proto_initialized:
                        aggregated_proto = torch.zeros_like(client_proto, device=self.device)
                        proto_initialized = True
                    
                    aggregated_proto += client_proto * current_weight
                    total_weight_for_label += current_weight
            # else:
                # print(f"Client {client.id} does not have a local prototype for label {label_idx}. Skipping for proto aggregation.")
        
        if proto_initialized and total_weight_for_label > 0:
            aggregated_proto /= total_weight_for_label
            return aggregated_proto
        else:
            # print(f"Warning: Could not aggregate proto for label {label_idx}. Total weight {total_weight_for_label}, Initialized: {proto_initialized}")
            return None


    def train(self):
        for i in range(self.global_rounds + 1):
            self.current_round = i

            s_t = time.time()

            # Potentially apply concept drift (logic from reference FedCCFA.py main script)
            # This would require access to the full client list and global_test_sets,
            # and drift functions like sudden_drift, incremental_drift.
            # For now, this is omitted from server.train() but could be added if drift simulation is needed here.

            selected_clients = self.select_clients() # From ServerBase
            self.get_client_data_size(selected_clients) # Update data sizes for selected clients

            # Initial send of full model (includes rep and classifier)
            # Clients will use this to initialize or reset their models.
            # The reference FedCCFA.py sends full params, then clients might get p_clf_params.
            # Let's send full model, then rep_params after first rep aggregation.
            if i == 0: # Initial round, send full model
                 self.send_models() # beta=1 for full model
                 # Initialize client's p_clf_params with the current global classifier
                 for client in selected_clients:
                    client.p_clf_params = [p.detach().clone() for name, p in self.global_model.named_parameters() if name in self.clf_keys]            # Client-side operations
            client_timings = []
            balanced_clf_params_from_clients = {} # To store clf params after balanced_train or train_with_protos            # Apply concept drift transformation if needed
            self.apply_drift_transformation()

            for client in selected_clients:
                client_start_time = time.time()

                # 1. Client updates its label distribution
                client.update_label_distribution()
                self.evaluate(self.current_round, is_global=True)
                # 2. Client performs balanced training (updates its classifier)
                if self.args.balanced_epochs > 0:
                    client.balance_train() 
                    # Store classifier params after balanced training
                    balanced_clf_params_from_clients[client.id] = [p.detach().clone() for p in client.get_clf_parameters()]
                
                # 3. Client receives global prototypes (if not clustered_protos on client)
                # The client FedCCFA's train_with_protos uses self.global_protos.
                # This needs to be set on the client.
                if not self.args.clustered_protos: # clustered_protos is an arg for client behavior
                    # Send current server's global_protos to client
                    # Client's train_with_protos will use its self.global_protos
                    client.global_protos = [p.clone().to(client.device) if p is not None else None for p in self.global_protos]

                # 4. Client trains representation and classifier with prototypes
                # This updates client.model (rep and clf) and client.local_protos
                client.train_with_protos(current_round=i) 

                # 5. Store classifier params (if not already stored after balanced_train)
                if self.args.balanced_epochs == 0:
                    balanced_clf_params_from_clients[client.id] = [p.detach().clone() for p in client.get_clf_parameters()]
                
                client_timings.append(time.time() - client_start_time)

            # if self.args.use_wandb:
            #     wandb.log({"Client Training Time (avg)": np.mean(client_timings) if client_timings else 0}, step=i)

            # Server-side aggregations and updates
            server_aggregation_start_time = time.time()

            # 6. Aggregate representation layers from clients
            self.aggregate_rep(selected_clients)

            # 7. Aggregate global prototypes from clients' local_protos
            self.aggregate_protos(selected_clients, current_round=i)
            
            # 8. Send updated representation parameters to clients
            self.send_rep_params(selected_clients) # Clients now have G_rep, G_protos

            # 9. Merge classifiers
            if self.args.oracle:
                # print(f"Round {i}: Using Oracle Merging.")
                client_ids_this_round = [c.id for c in selected_clients]
                label_merged_client_groups = self.oracle_merging(i, client_ids_this_round)
            else:
                # print(f"Round {i}: Using DBSCAN Classifier Merging.")
                # Ensure balanced_clf_params_from_clients is populated correctly
                if not balanced_clf_params_from_clients:
                    print(f"Warning Round {i}: balanced_clf_params_from_clients is empty. Skipping classifier merging.")
                    # Create empty groups so loop below doesn't fail
                    label_merged_client_groups = {label_idx: [] for label_idx in range(self.args.num_classes)}
                else:
                    label_merged_client_groups = self.merge_classifiers(balanced_clf_params_from_clients)
            
            if self.args.use_wandb:
                # Log number of groups per label, or average number of groups
                num_groups_per_label = [len(groups) for groups in label_merged_client_groups.values()]
                wandb.log({
                    "Avg Groups per Label": np.mean(num_groups_per_label) if num_groups_per_label else 0,
                    "Total Groups": np.sum(num_groups_per_label)
                }, step=i)


            # 10. Aggregate label-specific parameters and prototypes for each group and send to clients
            for label_idx, client_id_groups_for_label in label_merged_client_groups.items():
                if not client_id_groups_for_label: # No groups for this label
                    # print(f"Round {i}, Label {label_idx}: No client groups from merging. Skipping label param/proto aggregation.")
                    continue

                for group_of_client_ids in client_id_groups_for_label:
                    if not group_of_client_ids: # Empty group
                        continue
                        
                    # Map client IDs to client objects
                    clients_in_current_group = [client for client in selected_clients if client.id in group_of_client_ids]
                    if not clients_in_current_group:
                        # print(f"Round {i}, Label {label_idx}: Client ID group {group_of_client_ids} resulted in no client objects. Skipping.")
                        continue

                    # Aggregate label-specific classifier parameters for this group
                    aggregated_label_specific_params_vector = self.aggregate_label_params(label_idx, clients_in_current_group)
                    
                    # Aggregate label-specific prototypes for this group
                    aggregated_label_specific_proto = self.aggregate_label_protos(label_idx, clients_in_current_group)

                    # Send to clients in this group
                    for client in clients_in_current_group:
                        # Client needs to set these aggregated label-specific params into its classifier
                        if aggregated_label_specific_params_vector is not None:
                            # Client needs a method like set_label_specific_parameters(label_idx, params_vector)
                            # This method would convert vector to param list and update model[clf_keys][label_idx]
                            client.set_label_parameters(label_idx, aggregated_label_specific_params_vector) # Assumes client method takes vector

                        # Client updates its global_protos[label_idx] with this group's aggregated proto
                        if aggregated_label_specific_proto is not None:
                            if not client.global_protos or len(client.global_protos) != self.args.num_classes:
                                client.global_protos = [None] * self.args.num_classes # Initialize if needed
                            client.global_protos[label_idx] = aggregated_label_specific_proto.clone().to(client.device)
            
            # 11. Clients store their final personalized classifier parameters for the next round's initialization
            for client in selected_clients:
                client.p_clf_params = [p.detach().clone() for p in client.get_clf_parameters()]

            if self.args.use_wandb:
                wandb.log({"Server Aggregation Time": time.time() - server_aggregation_start_time}, step=i)

            # Evaluation
            # if i % self.eval_gap == 0:

            if i % 10 == 0:
                print(f"\\n------------- Round {i} Evaluation -------------")
                

                # Test on clients using their updated local models
                self.evaluate(current_round=self.current_round, is_global=False)# Uses client.test_metrics()

                # Optionally, test global model on a global test set if available
            

        print("\\nFedCCFA Training finished.")

        self.save_results()
        self.save_global_model(self.current_round)

    def evaluate_global_model(self, current_round, final_eval=False):
        if self.global_test_set is None:
            return

        stats = self.test_metrics(self.global_model, self.global_test_set, self.device)
        
        if self.args.use_wandb:
            log_data = {
                f"Global Model/Average Accuracy": stats['accuracy'],
                f"Global Model/Average Loss": stats['loss']
            }
            if final_eval:
                 wandb.summary[f"Final Global Model Accuracy"] = stats['accuracy']
                 wandb.summary[f"Final Global Model Loss"] = stats['loss']

            wandb.log(log_data, step=current_round)
        print(f"Global Model --- Round {current_round} --- Acc: {stats['accuracy']:.4f}, Loss: {stats['loss']:.4f}")



    def set_clf_keys(self, clf_keys):
        """Sets the classifier keys for the server."""
        self.clf_keys = clf_keys
        print(f"Server clf_keys set to: {self.clf_keys}")
