import copy
import torch
import torch.nn as nn
import numpy as np
import random
import time
import traceback # For detailed error logging
# Removed: from sklearn.mixture import GaussianMixture
# Removed: from sklearn.cluster import DBSCAN 
import ot # Python Optimal Transport library
# from scipy.stats import multivariate_normal # For GMM sampling if needed, or use np.random.multivariate_normal

from flcore.servers.serverbase import Server
from flcore.clients.clientdca import clientDCA # Ensure clientDCA is imported
# ... other necessary imports ...

# Removed GMM-specific sampling. This function now directly computes Sinkhorn distance between two sets of samples.
def compute_sinkhorn_distance_samples(samples1, samples2, reg=0.1):
    if samples1 is None or samples2 is None or samples1.size == 0 or samples2.size == 0 or samples1.shape[0] < 1 or samples2.shape[0] < 1:
        return float('inf')
    
    s1 = samples1
    s2 = samples2

    if s1.ndim == 1: s1 = s1.reshape(-1, 1)
    if s2.ndim == 1: s2 = s2.reshape(-1, 1)

    if s1.shape[1] != s2.shape[1]:
        # print(f"Warning: Shape mismatch in compute_sinkhorn_distance_samples. s1: {s1.shape}, s2: {s2.shape}")
        return float('inf')
    if s1.shape[1] == 0 : # No features
        return float('inf')

    try:
        # Assuming samples are weighted uniformly.
        # Calculate the cost matrix M (e.g., squared Euclidean distance)
        M = ot.dist(s1, s2, metric='sqeuclidean')

        # Define uniform marginal distributions
        a = np.ones((s1.shape[0],), dtype=np.float64) / s1.shape[0]
        b = np.ones((s2.shape[0],), dtype=np.float64) / s2.shape[0]
        
        # Use ot.sinkhorn to get the transport plan gamma
        # Then compute the squared Sinkhorn distance as sum(gamma * M)
        gamma = ot.sinkhorn(a, b, M, reg=reg, numItermax=200, stopThr=1e-7)
        
        # The squared Sinkhorn distance (transport cost)
        dist_sq_val = np.sum(gamma * M)

        # No longer need to check if dist_sq_val is a tuple, as np.sum returns a scalar.

        if np.isnan(dist_sq_val) or np.isinf(dist_sq_val):
            return float('inf')
        
        # dist_sq_val is now the squared Sinkhorn distance.
        # For comparison purposes, the squared distance is often sufficient and avoids sqrt.
        # If the actual distance is needed, uncomment the next lines:
        # if dist_sq_val < 0: # Should not happen with sqeuclidean cost and proper convergence
        #     print(f"Warning: Negative squared Sinkhorn distance ({dist_sq_val}) before sqrt. Returning inf.")
        #     return float('inf')
        # return np.sqrt(dist_sq_val)
        
        return dist_sq_val # Returning squared distance for now
    except Exception as e:
        print(f"Error in compute_sinkhorn_distance_samples: {e}")
        return float('inf')

# Removed GMM-specific sampling and final GMM fitting.
# This function now computes Wasserstein barycenter from a list of sample sets.
# It returns the barycentric samples directly as an np.ndarray.
def compute_wasserstein_barycenter_samples(client_sample_sets_list, client_weights, reg=0.1):
    if not client_sample_sets_list or len(client_sample_sets_list) != len(client_weights):
        # print("Warning: Invalid input for compute_wasserstein_barycenter_samples.")
        return None

    active_samples_list = []
    active_weights = []

    for i, s_set in enumerate(client_sample_sets_list):
        if s_set is None or s_set.size == 0:
            continue
        
        current_samples = s_set
        if current_samples.ndim == 1: current_samples = current_samples.reshape(-1, 1) # Ensure 2D
        if current_samples.ndim == 0: continue

        active_samples_list.append(current_samples)
        active_weights.append(client_weights[i])

    if not active_samples_list or not active_weights:
        # print("Warning: No active samples or weights for barycenter computation.")
        return None
        
    # Check dimensionality consistency
    dim = active_samples_list[0].shape[1]
    if not all(s.shape[1] == dim for s in active_samples_list):
        # print("Warning: Dimensionality mismatch in samples for barycenter.")
        return None
    if dim == 0: 
        # print("Warning: Zero dimensionality samples for barycenter.")
        return None

    normalized_weights = np.array(active_weights, dtype=np.float64)
    sum_weights = np.sum(normalized_weights)
    if sum_weights <= 0:
        # print("Warning: Sum of weights is not positive for barycenter.")
        return None 
    normalized_weights /= sum_weights

    try:
        # ot.bregman.barycenter expects a list of sample arrays
        barycentric_samples = ot.bregman.barycenter(
            active_samples_list, 
            reg=reg, 
            weights=normalized_weights,
            stopThr=1e-5, 
            numItermax=150, # Consider making these configurable
            verbose=False, 
            log=False
        )

        if barycentric_samples is None or barycentric_samples.size == 0:
            # print("Warning: Barycenter computation resulted in no samples.")
            return None
        
        return barycentric_samples # Return the raw barycentric samples
    except Exception as e:
        # print(f"Error in compute_wasserstein_barycenter_samples: {e}")
        # print(traceback.format_exc())
        return None

def _get_feature_extractor_module_from_model(model_instance):
    if hasattr(model_instance, 'base') and model_instance.base is not None:
        return model_instance.base
    elif hasattr(model_instance, 'body') and model_instance.body is not None: # For models like ResNet
        return model_instance.body
    elif hasattr(model_instance, 'features') and model_instance.features is not None: # For models like VGG
        return model_instance.features
    elif hasattr(model_instance, 'encoder') and model_instance.encoder is not None: # For autoencoder-like models
        return model_instance.encoder
    # print(f"Warning: Could not identify feature extractor module from model.")
    return None

class FedDCA(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientDCA) 
        
        self.cluster_inited = False
        self.args = args
        self.args.load_pretrain = False 

        self.client_label_profiles = {} 
        self.client_label_profiles_history = {} 
        self.drift_threshold_wasserstein = args.drift_threshold_wasserstein if hasattr(args, 'drift_threshold_wasserstein') else 0.5 
        self.reduce_drifted_influence_factor = args.reduce_drifted_influence_factor if hasattr(args, 'reduce_drifted_influence_factor') else 1.0
          # VWC specific attributes
        self.vwc_reg = args.vwc_reg if hasattr(args, 'vwc_reg') else 0.1
        self.vwc_num_samples_dist = args.vwc_num_samples_dist if hasattr(args, 'vwc_num_samples_dist') else 500
        self.vwc_num_samples_bary = args.vwc_num_samples_bary if hasattr(args, 'vwc_num_samples_bary') else 500
        self.vwc_num_samples_concept = args.vwc_num_samples_concept if hasattr(args, 'vwc_num_samples_concept') else 30 
        self.vwc_num_centroid_samples = args.vwc_num_centroid_samples if hasattr(args, 'vwc_num_centroid_samples') else 30
        self.vwc_max_iter = args.vwc_max_iter if hasattr(args, 'vwc_max_iter') else 10
        self.vwc_K_t = args.vwc_K_t if hasattr(args, 'vwc_K_t') else 3 
        self.vwc_drift_penalty_factor = args.vwc_drift_penalty_factor if hasattr(args, 'vwc_drift_penalty_factor') else 0.5
        self.vwc_potential_update_factor = args.vwc_potential_update_factor if hasattr(args, 'vwc_potential_update_factor') else self.vwc_reg 
        self.gmm_n_components_concept = args.gmm_n_components_concept if hasattr(args, 'gmm_n_components_concept') else 1
        self.gmm_n_components_cluster = args.gmm_n_components_cluster if hasattr(args, 'gmm_n_components_cluster') else 1
        self.gmm_cov_type_barycenter = args.gmm_cov_type_barycenter if hasattr(args, 'gmm_cov_type_barycenter') else 'full'
        self.gmm_reg_covar_barycenter = args.gmm_reg_covar_barycenter if hasattr(args, 'gmm_reg_covar_barycenter') else 1e-6

        self.cluster_classifiers = {}  
        self.client_cluster_assignments = {}  
        
        self.clf_keys = [] 
        if hasattr(args.model, 'head') and args.model.head is not None:
            self.clf_keys = list(args.model.head.state_dict().keys())
        
        self.Budget = []
        self.current_round = 0 

    def train(self):
        for i in range(self.global_rounds + 1):
            self.current_round = i
            self.selected_clients = self.select_clients()
            self.send_models() 

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}--------------")
                print("Evaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train() 
            
            self.receive_client_updates()
            self.perform_drift_analysis_and_adaptation()
            
            current_K_t = self.vwc_K_t 
            if len(self.client_label_profiles) > 0 :
                 self.run_vwc_clustering(self.client_label_profiles, self.client_drift_status, current_K_t)
            else:
                print("No client profiles received, skipping VWC clustering.")
                for client_obj in self.selected_clients: 
                    self.client_cluster_assignments[client_obj.id] = 0

            self.aggregate_parameters()
            
            # if hasattr(self, 'learning_rate_scheduler') and self.learning_rate_scheduler is not None:
            #     self.learning_rate_scheduler.step()

        print("\nFinal Global Trainning accuracy:")
        self.print_(max(self.rs_train_acc), max(self.rs_train_loss))
        print("\nFinal Global Test accuracy:")
        self.print_(max(self.rs_test_acc), max(self.rs_global_test_acc), max(self.rs_test_auc))

        self.save_results()
        self.save_global_model()

    def send_models(self):
        if not self.selected_clients:
            return

        feature_extractor_params = None
        if hasattr(self.global_model, 'base') and self.global_model.base is not None:
            feature_extractor_params = {k: v.cpu() for k, v in self.global_model.base.state_dict().items()}
        elif hasattr(self.global_model, 'body') and self.global_model.body is not None:
             feature_extractor_params = {k: v.cpu() for k, v in self.global_model.body.state_dict().items()}
        else: 
            print("Warning: Could not determine feature extractor from global_model for sending.")
            feature_extractor_params = {k: v.cpu() for k, v in self.global_model.state_dict().items()} 

        for client in self.selected_clients:
            client.set_parameters(feature_extractor_params, part='feature_extractor')
            
            cluster_id = self.client_cluster_assignments.get(client.id)
            if cluster_id is not None and cluster_id in self.cluster_classifiers:
                classifier_params = {k: v.cpu() for k, v in self.cluster_classifiers[cluster_id].state_dict().items()}
                client.set_parameters(classifier_params, part='classifier')
            else:
                if hasattr(self.global_model, 'head') and self.global_model.head is not None:
                    default_classifier_params = {k: v.cpu() for k, v in self.global_model.head.state_dict().items()}
                    client.set_parameters(default_classifier_params, part='classifier')

    def receive_client_updates(self):
        self.client_label_profiles.clear()
        self.client_feature_extractors_params = {} # Assuming this might be used elsewhere or for future features
        self.client_classifier_params = {} # To store received classifier heads

        for client in self.selected_clients:
            # lp_c_t is now expected to be: Dict[label, Tuple[np.ndarray_samples, np.ndarray_losses]]
            # Ensure client.get_label_profiles() in clientdca.py returns this structure.
            lp_c_t = client.get_label_profiles() 
            self.client_label_profiles[client.id] = lp_c_t
            
            # Store history of label profiles (samples and losses)
            if client.id not in self.client_label_profiles_history:
                self.client_label_profiles_history[client.id] = {}
            # Use copy.deepcopy for mutable structures like dicts containing numpy arrays
            self.client_label_profiles_history[client.id][self.current_round] = copy.deepcopy(lp_c_t)

            # Get classifier parameters from client
            if hasattr(client, 'get_clf_parameters'): # Corrected method name
                clf_params = client.get_clf_parameters() # Corrected method name
                if clf_params:
                    self.client_classifier_params[client.id] = clf_params            # else:
            #     print(f"Warning: Client {client.id} does not have get_clf_parameters method.")

    def perform_drift_analysis_and_adaptation(self):
        self.client_drift_status = {} 
        for client_id, profile_history in self.client_label_profiles_history.items():
            if self.current_round > 0 and (self.current_round - 1) in profile_history and self.current_round in profile_history:
                # profile_history[round] is Dict[label, Tuple[samples, losses]]
                label_data_t = profile_history[self.current_round] 
                label_data_t_minus_1 = profile_history[self.current_round - 1]
                
                total_wasserstein_dist = 0
                num_common_labels = 0
                
                for label, data_t_for_label in label_data_t.items():
                    if label in label_data_t_minus_1:
                        data_t_minus_1_for_label = label_data_t_minus_1[label]
                        
                        current_label_samples = data_t_for_label[0] # Extract samples
                        previous_label_samples = data_t_minus_1_for_label[0] # Extract samples
                        
                        # Corrected multi-line if condition using parentheses
                        if (current_label_samples is not None and previous_label_samples is not None and
                            current_label_samples.size > 0 and previous_label_samples.size > 0):
                            
                            dist = compute_sinkhorn_distance_samples(
                                current_label_samples, 
                                previous_label_samples,
                                reg=self.vwc_reg
                            )
                            if dist != float('inf'):
                                total_wasserstein_dist += dist
                                num_common_labels += 1
                
                avg_wasserstein_dist = total_wasserstein_dist / num_common_labels if num_common_labels > 0 else 0

                if avg_wasserstein_dist > self.drift_threshold_wasserstein:
                    self.client_drift_status[client_id] = True
                else:
                    self.client_drift_status[client_id] = False
            else:
                self.client_drift_status[client_id] = False
    
    def run_vwc_clustering(self, all_client_label_data, drift_status, K_t):
        """
        Performs label-wise Variational Wasserstein Clustering (VWC).
        
        Key changes:
        1. Cluster centroids are represented as Dict[label, np.ndarray_samples] per cluster
        2. Client-to-cluster distance is computed as average of per-label Sinkhorn distances
        3. Centroid updates use N lowest-loss samples per label from member clients
        
        Args:
            all_client_label_data: Dict[client_id, Dict[label, Tuple[samples, losses]]]
            drift_status: Dict[client_id, bool] indicating drift status
            K_t: Number of clusters
        """
        current_seed = getattr(self.args, 'seed', None)
        num_centroid_samples = getattr(self.args, 'vwc_num_centroid_samples', 30)

        client_ids = list(all_client_label_data.keys())
        if not client_ids:
            print("VWC: No clients to cluster.")
            self.client_cluster_assignments.clear()
            return

        if K_t <= 0:
            print(f"VWC: Invalid number of clusters K_t = {K_t}. Defaulting all to cluster 0.")
            for cid in client_ids:
                self.client_cluster_assignments[cid] = 0
            return

        # Filter clients with valid label data
        active_client_ids = []
        for client_id in client_ids:
            client_label_profiles = all_client_label_data.get(client_id, {})
            if client_label_profiles:  # Client has at least some label data
                active_client_ids.append(client_id)

        if not active_client_ids:
            print("VWC: No clients with valid label profiles. Assigning all to cluster 0.")
            for cid in client_ids:
                self.client_cluster_assignments[cid] = 0
            return

        # Adjust K_t based on available clients
        actual_K_t = min(K_t, len(active_client_ids))
        if actual_K_t <= 0:
            actual_K_t = 1
        if K_t != actual_K_t:
            print(f"VWC: Adjusted K_t from {K_t} to {actual_K_t} due to limited active clients.")
        K_t = actual_K_t

        # Initialize cluster centroids - each centroid is a Dict[label, samples]
        cluster_centroids_vk_dicts = []
        
        # Select K_t clients to initialize centroids
        if current_seed is not None:
            random.seed(current_seed)
        
        num_to_sample_init = min(K_t, len(active_client_ids))
        initial_client_indices = random.sample(range(len(active_client_ids)), num_to_sample_init)
        
        for i in range(num_to_sample_init):
            client_idx = initial_client_indices[i]
            client_id = active_client_ids[client_idx]
            client_label_profiles = all_client_label_data[client_id]
            
            # Create initial centroid from this client's label profiles
            initial_centroid = {}
            for label, (samples, losses) in client_label_profiles.items():
                if samples is not None and samples.size > 0 and losses is not None and losses.size > 0:
                    # Select num_centroid_samples lowest-loss samples for this label
                    if samples.shape[0] == losses.shape[0]:
                        sorted_indices = np.argsort(losses)
                        num_samples_to_take = min(num_centroid_samples, samples.shape[0])
                        selected_samples = samples[sorted_indices[:num_samples_to_take]]
                        
                        # Pad samples if fewer than num_centroid_samples
                        current_num_samples = selected_samples.shape[0]
                        if current_num_samples > 0 and current_num_samples < num_centroid_samples:
                            num_repeats = num_centroid_samples // current_num_samples
                            remainder = num_centroid_samples % current_num_samples
                            
                            padded_samples_list = [selected_samples] * num_repeats
                            if remainder > 0:
                                padded_samples_list.append(selected_samples[:remainder])
                            
                            if padded_samples_list: # Ensure list is not empty before vstack
                                selected_samples = np.vstack(padded_samples_list)

                        initial_centroid[label] = copy.deepcopy(selected_samples)
            
            if initial_centroid:  # Only add if centroid has at least one label
                cluster_centroids_vk_dicts.append(initial_centroid)

        # Handle case where we need more centroids (duplicate existing ones)
        while len(cluster_centroids_vk_dicts) < K_t:
            if cluster_centroids_vk_dicts:
                idx_to_duplicate = len(cluster_centroids_vk_dicts) % len(cluster_centroids_vk_dicts)
                cluster_centroids_vk_dicts.append(copy.deepcopy(cluster_centroids_vk_dicts[idx_to_duplicate]))
            else:
                # Fallback: create empty centroid
                cluster_centroids_vk_dicts.append({})

        # Update K_t to actual number of initialized centroids
        K_t = len(cluster_centroids_vk_dicts)
        if K_t == 0:
            print("VWC: Could not initialize any cluster centroids. Assigning all to cluster 0.")
            for cid in client_ids:
                self.client_cluster_assignments[cid] = 0
            return

        # VWC iteration
        current_assignments = {cid: -1 for cid in active_client_ids}
        
        for vwc_iter in range(self.vwc_max_iter):
            new_assignments = {}
            
            # Assignment Step: Assign each client to closest cluster
            for client_id in active_client_ids:
                client_label_profiles = all_client_label_data[client_id]
                min_avg_distance = float('inf')
                assigned_cluster_idx = -1
                
                for k_idx in range(K_t):
                    cluster_centroid = cluster_centroids_vk_dicts[k_idx]
                    
                    # Compute distance as average of per-label Sinkhorn distances
                    total_distance = 0.0
                    num_common_labels = 0
                    
                    for label in client_label_profiles.keys():
                        if label in cluster_centroid:
                            client_samples, _ = client_label_profiles[label]
                            centroid_samples = cluster_centroid[label]
                            
                            if (client_samples is not None and client_samples.size > 0 and
                                centroid_samples is not None and centroid_samples.size > 0):
                                
                                distance = compute_sinkhorn_distance_samples(
                                    client_samples, centroid_samples, reg=self.vwc_reg
                                )
                                
                                if distance != float('inf'):
                                    total_distance += distance
                                    num_common_labels += 1
                    
                    # Average distance over common labels
                    if num_common_labels > 0:
                        avg_distance = total_distance / num_common_labels
                        
                        # Apply drift penalty if client is drifted
                        if drift_status.get(client_id, False):
                            avg_distance += self.vwc_drift_penalty_factor * avg_distance
                        
                        if avg_distance < min_avg_distance:
                            min_avg_distance = avg_distance
                            assigned_cluster_idx = k_idx
                
                if assigned_cluster_idx != -1:
                    new_assignments[client_id] = assigned_cluster_idx
                else:
                    # Fallback: assign to cluster 0 if no valid assignment found
                    new_assignments[client_id] = 0

            # Check for convergence
            converged = True
            if vwc_iter > 0:
                for cid, assigned_k in current_assignments.items():
                    if assigned_k != new_assignments.get(cid, -1):
                        converged = False
                        break
                
                if converged:
                    break
            
            current_assignments = new_assignments

            # Update Step: Update cluster centroids based on member clients
            new_cluster_centroids = []
            
            for k_idx in range(K_t):
                # Find clients assigned to this cluster
                clients_in_cluster = [cid for cid, assigned_k in new_assignments.items() if assigned_k == k_idx]
                
                if not clients_in_cluster:
                    # Keep old centroid if cluster is empty
                    if k_idx < len(cluster_centroids_vk_dicts):
                        new_cluster_centroids.append(copy.deepcopy(cluster_centroids_vk_dicts[k_idx]))
                    else:
                        new_cluster_centroids.append({})
                    continue

                # Collect all labels present in member clients
                all_labels_in_cluster = set()
                for cid in clients_in_cluster:
                    all_labels_in_cluster.update(all_client_label_data[cid].keys())

                # For each label, pool samples from all member clients and select lowest-loss ones
                new_centroid = {}
                for label in all_labels_in_cluster:
                    pooled_samples_list = []
                    pooled_losses_list = []
                    
                    for cid in clients_in_cluster:
                        if label in all_client_label_data[cid]:
                            samples, losses = all_client_label_data[cid][label]
                            if (samples is not None and samples.size > 0 and
                                losses is not None and losses.size > 0 and
                                samples.shape[0] == losses.shape[0]):
                                pooled_samples_list.append(samples)
                                pooled_losses_list.append(losses)
                    
                    if pooled_samples_list:
                        # Combine all samples and losses for this label
                        combined_samples = np.vstack(pooled_samples_list)
                        combined_losses = np.concatenate(pooled_losses_list)
                        
                        # Select num_centroid_samples lowest-loss samples
                        sorted_indices = np.argsort(combined_losses)
                        num_samples_to_take = min(num_centroid_samples, combined_samples.shape[0])
                        centroid_samples_for_label = combined_samples[sorted_indices[:num_samples_to_take]]
                        new_centroid[label] = centroid_samples_for_label

                new_cluster_centroids.append(new_centroid)

            # Update centroids
            cluster_centroids_vk_dicts = new_cluster_centroids

        # Final assignment for all clients (including inactive ones)
        final_client_assignments = {}
        for client_id in client_ids:
            if client_id in current_assignments:
                final_client_assignments[client_id] = current_assignments[client_id]
            else:
                # Assign inactive clients to cluster 0
                final_client_assignments[client_id] = 0

        self.client_cluster_assignments = final_client_assignments

        # Aggregate classifier heads for each cluster
        self.cluster_classifiers.clear()
        client_id_to_datasize_map = {client.id: client.train_samples for client in self.clients if hasattr(client, 'train_samples')}

        for k_idx in range(K_t):
            clients_assigned_to_k = [cid for cid, assigned_k in self.client_cluster_assignments.items() if assigned_k == k_idx]
            
            if not clients_assigned_to_k:
                # Empty cluster, use global head
                if hasattr(self.global_model, 'head') and self.global_model.head is not None:
                    self.cluster_classifiers[k_idx] = copy.deepcopy(self.global_model.head)
                continue

            # Collect classifier parameters from cluster members
            classifier_heads_params_list = []
            data_sizes_list = []
            
            for cid in clients_assigned_to_k:
                if cid in self.client_classifier_params:
                    classifier_heads_params_list.append(self.client_classifier_params[cid])
                    data_sizes_list.append(client_id_to_datasize_map.get(cid, 1.0))
            
            if classifier_heads_params_list:
                # Weighted average of classifier parameters
                aggregated_params = {}
                first_head_keys = classifier_heads_params_list[0].keys()

                for key in first_head_keys:
                    weighted_sum = torch.zeros_like(classifier_heads_params_list[0][key], dtype=torch.float32)
                    total_weight = 0.0

                    for i, params in enumerate(classifier_heads_params_list):
                        if key in params:
                            weight = data_sizes_list[i]
                            weighted_sum += params[key].float() * weight
                            total_weight += weight
                    
                    if total_weight > 0:
                        aggregated_params[key] = weighted_sum / total_weight
                    else:
                        aggregated_params[key] = classifier_heads_params_list[0][key]

                # Create cluster classifier
                if hasattr(self.global_model, 'head') and self.global_model.head is not None:
                    cluster_head = copy.deepcopy(self.global_model.head)
                    try:
                        cluster_head.load_state_dict(aggregated_params)
                        self.cluster_classifiers[k_idx] = cluster_head
                    except RuntimeError as e:
                        print(f"VWC: Error loading aggregated state_dict for cluster {k_idx}: {e}. Using global head.")
                        self.cluster_classifiers[k_idx] = copy.deepcopy(self.global_model.head)
                else:
                    self.cluster_classifiers[k_idx] = copy.deepcopy(self.global_model.head)
            else:
                # No classifier heads found, use global head
                if hasattr(self.global_model, 'head') and self.global_model.head is not None:
                    self.cluster_classifiers[k_idx] = copy.deepcopy(self.global_model.head)

        print(f"VWC: Label-wise clustering complete. Assignments: {self.client_cluster_assignments}")
        print(f"VWC: {len(cluster_centroids_vk_dicts)} clusters created with {len(self.cluster_classifiers)} cluster classifiers.")

    def aggregate_parameters(self):
        """
        Aggregates feature extractors from selected clients to update the global model's feature extractor.
        Also aggregates a global classifier head from received client classifier parameters.
        Cluster-specific classifiers are handled in run_vwc_clustering.
        """
        if not self.selected_clients:
            # print("FedDCA: No selected clients for parameter aggregation.")
            return

        # 1. Aggregate Feature Extractors
        uploaded_feature_extractor_state_dicts = []
        feature_extractor_weights = []

        for client in self.selected_clients:
            client_model_fe_module = _get_feature_extractor_module_from_model(client.model)
            if client_model_fe_module:
                uploaded_feature_extractor_state_dicts.append(copy.deepcopy(client_model_fe_module.state_dict()))
                feature_extractor_weights.append(client.train_samples)
            # else:
                # print(f"FedDCA Warning: Could not get feature extractor from client {client.id} for aggregation.")

        if uploaded_feature_extractor_state_dicts:
            total_fe_weight = sum(feature_extractor_weights)
            if total_fe_weight > 0:
                norm_fe_weights = [w / total_fe_weight for w in feature_extractor_weights]
                
                # Initialize aggregated_fe_params from the first one
                aggregated_fe_params = copy.deepcopy(uploaded_feature_extractor_state_dicts[0])
                for key in aggregated_fe_params:
                    aggregated_fe_params[key].zero_()

                for fe_state_dict, weight in zip(uploaded_feature_extractor_state_dicts, norm_fe_weights):
                    for key in aggregated_fe_params:
                        if key in fe_state_dict:
                            aggregated_fe_params[key] += fe_state_dict[key] * weight
                
                global_model_fe_module = _get_feature_extractor_module_from_model(self.global_model)
                if global_model_fe_module:
                    try:
                        global_model_fe_module.load_state_dict(aggregated_fe_params)
                    except RuntimeError as e:
                        # print(f"FedDCA: Error loading aggregated feature extractor params (strict=True): {e}. Trying strict=False.")
                        try:
                            global_model_fe_module.load_state_dict(aggregated_fe_params, strict=False)
                        except Exception as e_false:
                            # print(f"FedDCA: Error loading aggregated feature extractor params (strict=False): {e_false}")
                            pass # Or handle more gracefully
                # else:
                    # print("FedDCA Warning: Could not identify global model's feature extractor to load aggregated params.")
            # else:
                # print("FedDCA: Total weight for feature extractor aggregation is zero. Skipping FE update.")
        # else:
            # print("FedDCA: No feature extractors collected for aggregation. Global feature extractor not updated.")

        # 2. Aggregate a Global Classifier Head for self.global_model.head
        if not self.client_classifier_params:
            # print("FedDCA: No client classifier parameters received for global head aggregation.")
            return # Return if no classifier params, FE might have been updated.

        if hasattr(self.global_model, 'head') and self.global_model.head is not None:
            aggregated_clf_params_state_dict = None
            first_valid_head_state_dict = None
            
            # Find the first valid head to determine keys and initialize structure
            for cid_temp in self.client_classifier_params:
                if self.client_classifier_params[cid_temp]: # Check if state_dict is not empty
                    first_valid_head_state_dict = self.client_classifier_params[cid_temp]
                    break
            
            if first_valid_head_state_dict:
                # Initialize with zeros, same keys and device as the first valid head
                aggregated_clf_params_state_dict = {
                    k: torch.zeros_like(v, device=v.device) 
                    for k, v in first_valid_head_state_dict.items()
                }
                total_clf_weight = 0.0
                
                # Build a map for client train_samples lookup
                client_train_samples_map = {c.id: c.train_samples for c in self.clients}

                for client_id, clf_state_dict in self.client_classifier_params.items():
                    if clf_state_dict: # if state_dict is not empty
                        weight = client_train_samples_map.get(client_id, 1.0) # Default weight 1.0
                        
                        for key in aggregated_clf_params_state_dict:
                            if key in clf_state_dict: # Ensure key exists in current client's head
                                # Ensure tensors are on the same device before aggregation
                                # self.client_classifier_params should store CPU tensors as per clientDCA
                                aggregated_clf_params_state_dict[key] += clf_state_dict[key].to(aggregated_clf_params_state_dict[key].device) * weight
                        total_clf_weight += weight
                
                if total_clf_weight > 0:
                    for key in aggregated_clf_params_state_dict:
                        aggregated_clf_params_state_dict[key] /= total_clf_weight
                    
                    try:
                        self.global_model.head.load_state_dict(aggregated_clf_params_state_dict)
                    except RuntimeError as e:
                        # print(f"FedDCA: Error loading aggregated global classifier head params (strict=True): {e}. Trying strict=False.")
                        try:
                            self.global_model.head.load_state_dict(aggregated_clf_params_state_dict, strict=False)
                        except Exception as e_false:
                            # print(f"FedDCA: Error loading aggregated global classifier head params (strict=False): {e_false}")
                            pass
                # else:
                    # print("FedDCA: Total weight for global classifier head aggregation is zero. Global head not updated.")
            # else:
                # print("FedDCA: No valid client classifier params found to initialize global head aggregation. Global head not updated.")
        # else:
            # print("FedDCA: Global model has no 'head' attribute, or it's None. Global head not updated.")


    def set_clf_keys(self, clf_keys):
        """Sets the classifier keys for the server."""
