import time
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import ot
import copy # Added for deepcopying profiles if needed
import random # Added to resolve NameError
from sklearn.cluster import KMeans # Added for run_vwc_clustering fallback
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.mixture import GaussianMixture
from scipy.stats import wasserstein_distance
import itertools # For combinations in splitting

# Added imports for comprehensive evaluation
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from collections import Counter # For purity calculation
import json # For saving evaluation report

import wandb # Added for global access, complemented by try-except in __init__

from flcore.servers.serverbase import Server
from flcore.clients.clientdca import clientDCA # Ensure clientDCA is imported

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
        # Normalize the cost matrix
        if M.max() > 0:
            M = M / M.max()

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
        self.client_label_profiles_history = {} # User's existing attribute
        self.drift_threshold_wasserstein = args.drift_threshold_wasserstein if hasattr(args, 'drift_threshold_wasserstein') else 0.5 
        self.reduce_drifted_influence_factor = args.reduce_drifted_influence_factor if hasattr(args, 'reduce_drifted_influence_factor') else 1.0
        
        # VWC specific attributes from user's code
        self.vwc_reg = args.vwc_reg if hasattr(args, 'vwc_reg') else 0.1
        self.vwc_num_samples_dist = args.vwc_num_samples_dist if hasattr(args, 'vwc_num_samples_dist') else 500
        self.vwc_num_samples_bary = args.vwc_num_samples_bary if hasattr(args, 'vwc_num_samples_bary') else 500
        self.vwc_num_samples_concept = args.vwc_num_samples_concept if hasattr(args, 'vwc_num_samples_concept') else 30 
        self.vwc_num_centroid_samples = args.vwc_num_centroid_samples if hasattr(args, 'vwc_num_centroid_samples') else 30 # User had this
        self.vwc_max_iter = args.vwc_max_iter if hasattr(args, 'vwc_max_iter') else 10
        self.vwc_K_t = args.vwc_K_t if hasattr(args, 'vwc_K_t') else 3 # User's default for K_t
        self.vwc_drift_penalty_factor = args.vwc_drift_penalty_factor if hasattr(args, 'vwc_drift_penalty_factor') else 0.5
        self.vwc_potential_update_factor = args.vwc_potential_update_factor if hasattr(args, 'vwc_potential_update_factor') else self.vwc_reg 
        
        # Dynamic K configuration
        self.dynamic_k_enabled = getattr(args, 'dynamic_k_enabled', False)
        self.dynamic_k_min = getattr(args, 'dynamic_k_min', 2)
        self.dynamic_k_max = getattr(args, 'dynamic_k_max', 10)
        self.dynamic_k_start_round = getattr(args, 'dynamic_k_start_round', 0) # Round from which to start dynamic K adjustments
        self.dynamic_k_silhouette_method = getattr(args, 'dynamic_k_silhouette_method', 'avg_client_silhouette') # e.g., 'avg_client_silhouette' or 'standard'
        self.dynamic_k_silhouette_range_step = getattr(args, 'dynamic_k_silhouette_range_step', 1) # Step for K range in silhouette analysis
        # self.dynamic_k_initial_estimation_iter = getattr(args, 'dynamic_k_initial_estimation_iter', 3) # Iterations for VWC during K estimation (if VWC itself is iterative) - currently GMM is direct
        self.dynamic_k_refinement_iterations = getattr(args, 'dynamic_k_refinement_iterations', 5) # Max iterations for merge/split refinement
        self.dynamic_k_merge_wasserstein_threshold = getattr(args, 'dynamic_k_merge_wasserstein_threshold', 0.2) # Threshold for merging clusters
        self.dynamic_k_split_dispersion_threshold = getattr(args, 'dynamic_k_split_dispersion_threshold', 0.5) # Threshold for splitting clusters
        self.dynamic_k_split_min_clients = getattr(args, 'dynamic_k_split_min_clients', 5) # Min clients in a cluster to consider splitting
        self.dynamic_k_split_subcluster_k = getattr(args, 'dynamic_k_split_subcluster_k', 2) # Number of sub-clusters when splitting
        self.dynamic_k_min_silhouette_improvement = getattr(args, 'dynamic_k_min_silhouette_improvement', 0.01) # Min improvement for merge/split (optional)

        self.gmm_n_components_concept = args.gmm_n_components_concept if hasattr(args, 'gmm_n_components_concept') else 1
        self.gmm_n_components_cluster = args.gmm_n_components_cluster if hasattr(args, 'gmm_n_components_cluster') else 1
        self.gmm_cov_type_barycenter = args.gmm_cov_type_barycenter if hasattr(args, 'gmm_cov_type_barycenter') else 'full'
        self.gmm_reg_covar_barycenter = args.gmm_reg_covar_barycenter if hasattr(args, 'gmm_reg_covar_barycenter') else 1e-6

        self.cluster_classifiers = {}  
        self.client_cluster_assignments = {}  
        self.client_drift_status = {} # From user's perform_drift_analysis_and_adaptation
        self.client_classifier_params = {} # From user's receive_client_updates

        self.clf_keys = [] 
        if hasattr(args.model, 'head') and args.model.head is not None:
            self.clf_keys = list(args.model.head.state_dict().keys()) # Completed this line
        
        self.Budget = []
        # self.current_round = 0 # Initialized in ServerBase

        # Comprehensive evaluation system initialization
        self.evaluation_metrics = {
            'clustering_quality': {
                'ari_history': [], 'nmi_history': [], 'purity_history': [],
            },
            'communication_efficiency': {
                'upload_bytes_per_round': [], 'download_bytes_per_round': [],
                'avg_client_compute_time_per_round': [], 'server_compute_time_per_round': []
            },
            'fairness_metrics': {
                'accuracy_std_per_round': [], 'worst_client_accuracy_per_round': [],
                'client_accuracies_history': {} # Dict[round, List[acc]]
            },
            'drift_adaptation_capability': {
                'detected_drifts_per_round': [],             # Count of drifted clients each round
                'accuracy_drifted_clients_post_adaptation': [], # Avg accuracy of clients flagged as drifted, after adaptation
                'accuracy_non_drifted_clients': [],          # Avg accuracy of non-drifted clients for comparison
            }
        }
        self.true_client_concepts = {}  # Dict[client_id, concept_id] for clustering eval
        
        # For current round's detailed communication tracking
        self.bytes_tracker = { 
            'upload_per_client': {}, 'download_per_client': {},
            'total_upload_this_round': 0, 'total_download_this_round': 0
        }
        
        # Wandb setup
        if self.args.use_wandb:
            try:
                import wandb # Ensure wandb is imported here
            except ImportError:
                print("wandb not installed, proceeding without it.")
                self.args.use_wandb = False

    def train(self):
        for i in range(self.global_rounds + 1):
            self.current_round = i
            self._init_round_tracking()  # Initialize round-specific trackers
            self._load_ground_truth_concepts()  # Load true client concepts

            server_compute_time_this_round = 0

            self.selected_clients = self.select_clients()
            
            send_models_start_time = time.time()
            self.send_models()
            server_compute_time_this_round += (time.time() - send_models_start_time)
            if self.current_round % self.args.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global models")
                self.evaluate()  # ServerBase evaluate method

            client_compute_times_this_round = []
            for client in self.selected_clients:
                client_train_start_time = time.time()
                client.train()
                client_compute_times_this_round.append(time.time() - client_train_start_time)
            
            if client_compute_times_this_round:
                avg_client_compute_time = sum(client_compute_times_this_round) / len(client_compute_times_this_round)
                self.evaluation_metrics['communication_efficiency']['avg_client_compute_time_per_round'].append(avg_client_compute_time)
            else:
                self.evaluation_metrics['communication_efficiency']['avg_client_compute_time_per_round'].append(0)

            receive_updates_start_time = time.time()
            self.receive_client_updates()
            server_compute_time_this_round += (time.time() - receive_updates_start_time)
            
            op_start_time = time.time()
            self.perform_drift_analysis_and_adaptation()
            server_compute_time_this_round += (time.time() - op_start_time)
            
            op_start_time = time.time()
            current_K_t = self.vwc_K_t
            if len(self.client_label_profiles) > 0:
                self.run_vwc_clustering(self.client_label_profiles, self.client_drift_status, current_K_t)
            else:
                print("No client profiles received, skipping VWC clustering.")
                for client_obj in self.selected_clients:
                    self.client_cluster_assignments[client_obj.id] = 0  # Default assignment
            server_compute_time_this_round += (time.time() - op_start_time)

            op_start_time = time.time()
            self.aggregate_parameters()
            server_compute_time_this_round += (time.time() - op_start_time)
            
            self.evaluation_metrics['communication_efficiency']['server_compute_time_per_round'].append(server_compute_time_this_round)

            # Comprehensive evaluation
            # if i % self.eval_gap == 0 and i > 0: # Avoid evaluation at round 0 if not meaningful
            if i % self.eval_gap == 0: # Avoid evaluation at round 0 if not meaningful
                print("\nEvaluate personalized models")
                self.evaluate(is_global=False)
                self._comprehensive_evaluation()

            # Original evaluate call (can be removed if _comprehensive_evaluation covers it)
            # if i % self.eval_gap == 0:
            #     print(f"\\n-------------Round number: {i}--------------")
            #     print("Evaluate global model") # This is ServerBase.evaluate()
            #     self.evaluate() 


            # if hasattr(self, 'learning_rate_scheduler') and self.learning_rate_scheduler is not None:
            #     self.learning_rate_scheduler.step()

        # End of training loop
        self._generate_final_evaluation_report()

        # Original final print statements (can be removed if covered by the report)
        # print("\\nFinal Global Trainning accuracy:")
        # if self.rs_train_acc: self.print_(max(self.rs_train_acc), max(self.rs_train_loss))
        # print("\\nFinal Global Test accuracy:")
        # if self.rs_test_acc: self.print_(max(self.rs_test_acc), max(self.rs_global_test_acc), max(self.rs_test_auc))

        self.save_results() # ServerBase method
        self.save_global_model()

    def send_models(self):
        if not self.selected_clients:
            return

        # Reset download bytes for this round for selected clients
        for client in self.selected_clients:
            self.bytes_tracker['download_per_client'][client.id] = 0
        self.bytes_tracker['total_download_this_round'] = 0

        feature_extractor_params = None
        if hasattr(self.global_model, 'base') and self.global_model.base is not None:
            feature_extractor_params = {k: v.cpu() for k, v in self.global_model.base.state_dict().items()}
        elif hasattr(self.global_model, 'body') and self.global_model.body is not None:
             feature_extractor_params = {k: v.cpu() for k, v in self.global_model.body.state_dict().items()}
        else: 
            print("Warning: Could not determine feature extractor from global_model for sending.")
            feature_extractor_params = {k: v.cpu() for k, v in self.global_model.state_dict().items()}

        fe_size_bytes = self._calculate_model_size_from_state_dict(feature_extractor_params)

        for client in self.selected_clients:
            client.set_parameters(feature_extractor_params, part='feature_extractor')
            self._track_communication_bytes(client.id, fe_size_bytes, 'download')
            
            cluster_id = self.client_cluster_assignments.get(client.id)
            classifier_params_to_send = None
            if cluster_id is not None and cluster_id in self.cluster_classifiers:
                classifier_params_to_send = {k: v.cpu() for k, v in self.cluster_classifiers[cluster_id].state_dict().items()}
            elif hasattr(self.global_model, 'head') and self.global_model.head is not None:
                classifier_params_to_send = {k: v.cpu() for k, v in self.global_model.head.state_dict().items()}
            
            if classifier_params_to_send:
                client.set_parameters(classifier_params_to_send, part='classifier')
                clf_size_bytes = self._calculate_model_size_from_state_dict(classifier_params_to_send)
                self._track_communication_bytes(client.id, clf_size_bytes, 'download')
            # else: client receives only FE if no relevant classifier found

    def receive_client_updates(self):
        self.client_label_profiles.clear()
        # self.client_feature_extractors_params.clear() # Clearing this if it was populated by clients
        self.client_classifier_params.clear()

        # Reset upload bytes for this round for selected clients
        for client in self.selected_clients:
            self.bytes_tracker['upload_per_client'][client.id] = 0
        self.bytes_tracker['total_upload_this_round'] = 0

        for client in self.selected_clients:
            lp_c_t = client.get_label_profiles()
            self.client_label_profiles[client.id] = lp_c_t
            
            # Estimate and track upload size for label profiles
            profile_size_bytes = self._estimate_profile_size(lp_c_t)
            self._track_communication_bytes(client.id, profile_size_bytes, 'upload')

            if client.id not in self.client_label_profiles_history:
                self.client_label_profiles_history[client.id] = {}
            self.client_label_profiles_history[client.id][self.current_round] = copy.deepcopy(lp_c_t)

            if hasattr(client, 'get_clf_parameters'):
                clf_params = client.get_clf_parameters()
                if clf_params:
                    self.client_classifier_params[client.id] = clf_params
                    # Estimate and track upload size for classifier parameters
                    # This assumes get_clf_parameters returns a state_dict or similar
                    clf_param_size_bytes = self._calculate_model_size_from_state_dict(clf_params)
                    self._track_communication_bytes(client.id, clf_param_size_bytes, 'upload')
            
            # If clients also send feature extractors (not typical in this FedDCA structure but for completeness):
            # if hasattr(client, 'get_feature_extractor_parameters'):
            #     fe_params = client.get_feature_extractor_parameters()
            #     if fe_params:
            #         # self.client_feature_extractors_params[client.id] = fe_params # If storing them
            #         fe_param_size_bytes = self._calculate_model_size_from_state_dict(fe_params)
            #         self._track_communication_bytes(client.id, fe_param_size_bytes, 'upload')

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
                
                # Build a map for client_train_samples lookup
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
        self.clf_keys = clf_keys

    # Helper and Evaluation methods for FedDCA
    def _init_round_tracking(self):
        """Initializes/resets trackers for the current round (e.g., communication bytes)."""
        self.bytes_tracker = { 
            'upload_per_client': {client.id: 0 for client in self.selected_clients},
            'download_per_client': {client.id: 0 for client in self.selected_clients},
            'total_upload_this_round': 0,
            'total_download_this_round': 0
        }

    def _load_ground_truth_concepts(self):
        """Loads or updates the true concept IDs for each client.
        This is a placeholder. In a real scenario, this might involve reading from a file 
        or using information from a drift simulation environment.
        For now, it assumes clients have a 'current_concept_id' attribute.
        """
        self.true_client_concepts.clear()
        for client in self.clients: # Iterate over all clients, not just selected ones for this round,
            if hasattr(client, 'current_concept_id'):
                self.true_client_concepts[client.id] = client.current_concept_id
            # else:
            #     print(f"Warning: Client {client.id} does not have current_concept_id for ground truth.")

    def _calculate_model_size_from_state_dict(self, state_dict):
        """Calculates the size of a model's state_dict in bytes."""
        if not state_dict: return 0
        size_bytes = 0
        for param_tensor in state_dict.values():
            if isinstance(param_tensor, torch.Tensor):
                size_bytes += param_tensor.element_size() * param_tensor.nelement()
        return size_bytes

    def _estimate_profile_size(self, label_profile):
        """Estimates the size of a client's label profile in bytes.
        label_profile: Dict[label, Tuple[np.ndarray_samples, np.ndarray_losses]]
        """
        if not label_profile: return 0
        size_bytes = 0
        for label, (samples, losses) in label_profile.items():
            if samples is not None:
                size_bytes += samples.nbytes
            if losses is not None:
                size_bytes += losses.nbytes
        # Add overhead for dictionary structure, keys, etc. (rough estimate)
        size_bytes += len(label_profile) * (np.dtype(int).itemsize + 2 * np.dtype(np.intp).itemsize) # For keys and tuple pointers
        return size_bytes

    def _track_communication_bytes(self, client_id, num_bytes, direction):
        """Tracks communication bytes for a client and the round total.
        direction: 'upload' or 'download'
        """
        if direction == 'upload':
            self.bytes_tracker['upload_per_client'][client_id] = self.bytes_tracker['upload_per_client'].get(client_id, 0) + num_bytes
            self.bytes_tracker['total_upload_this_round'] += num_bytes
        elif direction == 'download':
            self.bytes_tracker['download_per_client'][client_id] = self.bytes_tracker['download_per_client'].get(client_id, 0) + num_bytes
            self.bytes_tracker['total_download_this_round'] += num_bytes

    def _calculate_purity(self, y_true, y_pred):
        """Calculates clustering purity."""
        contingency_matrix = Counter()
        for true, pred in zip(y_true, y_pred):
            contingency_matrix[(true, pred)] += 1

        cluster_dominant_counts = {}
        for (true_label, cluster_label), count in contingency_matrix.items():
            if cluster_label not in cluster_dominant_counts:
                cluster_dominant_counts[cluster_label] = {}
            if true_label not in cluster_dominant_counts[cluster_label]:
                cluster_dominant_counts[cluster_label][true_label] = 0
            cluster_dominant_counts[cluster_label][true_label] += count

        correct_assignments = 0
        for cluster_label in cluster_dominant_counts:
            if cluster_dominant_counts[cluster_label]: # Check if dict is not empty
                correct_assignments += max(cluster_dominant_counts[cluster_label].values())
        
        return correct_assignments / len(y_true) if len(y_true) > 0 else 0

    def _evaluate_clustering_quality(self):
        """Evaluates clustering quality using ARI, NMI, and Purity."""
        if not self.client_cluster_assignments or not self.true_client_concepts:
            # print("Evaluation: Not enough data for clustering quality assessment.")
            return None, None, None

        # Align true_labels and pred_labels based on common client IDs present in assignments
        # This is crucial if not all clients participated or have ground truth
        client_ids_in_assignments = list(self.client_cluster_assignments.keys())
        
        true_labels_list = []
        pred_labels_list = []

        for cid in client_ids_in_assignments:
            if cid in self.true_client_concepts:
                true_labels_list.append(self.true_client_concepts[cid])
                pred_labels_list.append(self.client_cluster_assignments[cid])
            # else:
            #     print(f"Warning: Client {cid} in assignments but not in true_client_concepts.")

        if not true_labels_list or len(true_labels_list) < 2: # Metrics require at least 2 samples
            # print("Evaluation: Not enough common clients with ground truth for clustering metrics.")
            return 0, 0, 0 # Return default values if not enough data

        # Ensure all labels are integers for scikit-learn metrics
        try:
            true_labels_np = np.array(true_labels_list, dtype=int)
            pred_labels_np = np.array(pred_labels_list, dtype=int)
        except ValueError as e:
            print(f"Error converting labels to int for clustering metrics: {e}")
            return 0,0,0

        ari = adjusted_rand_score(true_labels_np, pred_labels_np)
        nmi = normalized_mutual_info_score(true_labels_np, pred_labels_np)
        purity = self._calculate_purity(true_labels_np, pred_labels_np)
        
        self.evaluation_metrics['clustering_quality']['ari_history'].append(ari)
        self.evaluation_metrics['clustering_quality']['nmi_history'].append(nmi)
        self.evaluation_metrics['clustering_quality']['purity_history'].append(purity)
        
        return ari, nmi, purity

    def _evaluate_communication_efficiency(self):
        """Stores current round's communication and timing metrics."""
        # Byte tracking is now done per round and stored directly in train loop
        self.evaluation_metrics['communication_efficiency']['upload_bytes_per_round'].append(self.bytes_tracker['total_upload_this_round'])
        self.evaluation_metrics['communication_efficiency']['download_bytes_per_round'].append(self.bytes_tracker['total_download_this_round'])
        # Client and server compute times are appended in the train loop
        return self.bytes_tracker['total_upload_this_round'], self.bytes_tracker['total_download_this_round']

    def _get_current_client_accuracies(self):
        """Collects test accuracies from evaluated clients and returns them as a dictionary."""
        client_accuracies_map = {}
        # self.eval_clients contains the client objects that were evaluated in the last call to self.evaluate()
        # self.rs_per_acc contains the corresponding accuracies in the same order.
        if hasattr(self, 'eval_clients') and hasattr(self, 'rs_per_acc') and self.eval_clients and self.rs_per_acc:
            if len(self.eval_clients) == len(self.rs_per_acc):
                for client_obj, acc in zip(self.eval_clients, self.rs_per_acc):
                    client_accuracies_map[client_obj.id] = acc
            else:
                print(f"Warning: Mismatch in lengths of eval_clients ({len(self.eval_clients)}) and rs_per_acc ({len(self.rs_per_acc)}).")
        else:
            print("Warning: eval_clients or rs_per_acc not populated. Ensure self.evaluate() was called.")
            # Fallback or default behavior if data is missing
            # For example, return empty or query all clients if necessary, though this might be slow.
            # For now, returning empty if not populated by self.evaluate().
            pass
        
        return client_accuracies_map

    def _evaluate_fairness_metrics(self):
        """Evaluates fairness metrics like accuracy standard deviation and worst-client performance."""
        client_accuracies = self._get_current_client_accuracies()
        
        if not client_accuracies:
            # print("Evaluation: No client accuracies available for fairness metrics.")
            # Append default/NaN values if no accuracies
            self.evaluation_metrics['fairness_metrics']['accuracy_std_per_round'].append(float('nan'))
            self.evaluation_metrics['fairness_metrics']['worst_client_accuracy_per_round'].append(float('nan'))
            self.evaluation_metrics['fairness_metrics']['client_accuracies_history'][self.current_round] = []
            return float('nan'), float('nan')

        acc_std = np.std(client_accuracies) if len(client_accuracies) > 1 else 0.0
        worst_acc = np.min(client_accuracies) if client_accuracies else float('nan')
        
        self.evaluation_metrics['fairness_metrics']['accuracy_std_per_round'].append(acc_std)
        self.evaluation_metrics['fairness_metrics']['worst_client_accuracy_per_round'].append(worst_acc)
        self.evaluation_metrics['fairness_metrics']['client_accuracies_history'][self.current_round] = client_accuracies
            
        return acc_std, worst_acc

    def _evaluate_drift_adaptation(self):
        """Evaluates the system's capability to adapt to concept drift."""
        if not hasattr(self, 'client_drift_status') or not self.client_drift_status:
            self.evaluation_metrics['drift_adaptation_capability']['detected_drifts_per_round'].append(0)
            # Append NaN or a placeholder if accuracies cannot be calculated due to no drift / no clients
            self.evaluation_metrics['drift_adaptation_capability']['accuracy_drifted_clients_post_adaptation'].append(float('nan'))
            self.evaluation_metrics['drift_adaptation_capability']['accuracy_non_drifted_clients'].append(float('nan'))
            return

        num_drifted_clients = sum(self.client_drift_status.values())
        self.evaluation_metrics['drift_adaptation_capability']['detected_drifts_per_round'].append(num_drifted_clients)

        current_client_accuracies = self._get_current_client_accuracies() # Assumes this returns a dict {client_id: accuracy}

        drifted_client_accuracies = []
        non_drifted_client_accuracies = []

        for client_id, is_drifted in self.client_drift_status.items():
            acc = current_client_accuracies.get(client_id)
            if acc is not None:
                if is_drifted:
                    drifted_client_accuracies.append(acc)
                else:
                    non_drifted_client_accuracies.append(acc)
        
        avg_acc_drifted = np.mean(drifted_client_accuracies) if drifted_client_accuracies else float('nan')
        avg_acc_non_drifted = np.mean(non_drifted_client_accuracies) if non_drifted_client_accuracies else float('nan')

        self.evaluation_metrics['drift_adaptation_capability']['accuracy_drifted_clients_post_adaptation'].append(avg_acc_drifted)
        self.evaluation_metrics['drift_adaptation_capability']['accuracy_non_drifted_clients'].append(avg_acc_non_drifted)

        if self.args.use_wandb:
            wandb.log({
                f"drift_adaptation/detected_drifts_round_{self.current_round}": num_drifted_clients,
                f"drift_adaptation/avg_acc_drifted_clients_round_{self.current_round}": avg_acc_drifted,
                f"drift_adaptation/avg_acc_non_drifted_clients_round_{self.current_round}": avg_acc_non_drifted,
                "round": self.current_round
            })

    def _comprehensive_evaluation(self):
        """Performs a comprehensive evaluation of the system."""
        print(f"--- Comprehensive Evaluation for Round {self.current_round} ---")
        self._evaluate_clustering_quality()
        self._evaluate_communication_efficiency() # This is mostly data collection, reporting happens in final
        self._evaluate_fairness_metrics()
        self._evaluate_drift_adaptation() # New call
        print("--- End of Comprehensive Evaluation ---")
    
    def _generate_final_evaluation_report(self):
        """Generates and prints/saves a final summary of all evaluations."""
        print("\n=============== Final Evaluation Report ===============")
        
        # Clustering Quality Summary
        print("\n--- Clustering Quality ---")
        if self.evaluation_metrics['clustering_quality']['ari_history']:
            avg_ari = np.mean(self.evaluation_metrics['clustering_quality']['ari_history'])
            avg_nmi = np.mean(self.evaluation_metrics['clustering_quality']['nmi_history'])
            avg_purity = np.mean(self.evaluation_metrics['clustering_quality']['purity_history'])
            print(f"  Average ARI: {avg_ari:.4f}")
            print(f"  Average NMI: {avg_nmi:.4f}")
            print(f"  Average Purity: {avg_purity:.4f}")
        else:
            print("  No clustering quality data recorded.")

        # Communication Efficiency Summary
        print("\n--- Communication & Computational Efficiency ---")
        if self.evaluation_metrics['communication_efficiency']['upload_bytes_per_round']:
            total_upload = np.sum(self.evaluation_metrics['communication_efficiency']['upload_bytes_per_round'])
            total_download = np.sum(self.evaluation_metrics['communication_efficiency']['download_bytes_per_round'])
            avg_round_client_time = np.mean(self.evaluation_metrics['communication_efficiency']['avg_client_compute_time_per_round'])
            avg_round_server_time = np.mean(self.evaluation_metrics['communication_efficiency']['server_compute_time_per_round'])
            print(f"  Total Upload Bytes: {total_upload}")
            print(f"  Total Download Bytes: {total_download}")
            print(f"  Average Client Compute Time per Round: {avg_round_client_time:.2f}s")
            print(f"  Average Server Compute Time per Round: {avg_round_server_time:.2f}s")
        else:
            print("  No communication/computation efficiency data recorded.")

        # Fairness Metrics Summary
        print("\n--- Fairness & Performance Consistency ---")
        if self.evaluation_metrics['fairness_metrics']['accuracy_std_per_round']:
            avg_acc_std = np.nanmean(self.evaluation_metrics['fairness_metrics']['accuracy_std_per_round'])
            avg_worst_acc = np.nanmean(self.evaluation_metrics['fairness_metrics']['worst_client_accuracy_per_round'])
            print(f"  Average Accuracy StdDev: {avg_acc_std:.4f}")
            print(f"  Average Worst-Client Accuracy: {avg_worst_acc:.4f}")

            # Save fairness boxplot
            all_rounds_client_accs = [accs for r, accs in self.evaluation_metrics['fairness_metrics']['client_accuracies_history'].items() if accs]
            if all_rounds_client_accs:
                try:
                    plt.figure(figsize=(10, 6))
                    # We need to decide how to present this. Boxplot of accuracies from LAST round? Or average per client?
                    # For now, let's plot the distribution of accuracies from the final recorded round with accuracies.
                    final_round_accuracies = None
                    for r in sorted(self.evaluation_metrics['fairness_metrics']['client_accuracies_history'].keys(), reverse=True):
                        if self.evaluation_metrics['fairness_metrics']['client_accuracies_history'][r]:
                            final_round_accuracies = self.evaluation_metrics['fairness_metrics']['client_accuracies_history'][r]
                            break
                    
                    if final_round_accuracies:
                        plt.boxplot(final_round_accuracies, vert=False)
                        plt.title(f'Client Accuracy Distribution (Final Round with Data: {self.current_round if not final_round_accuracies else r})')
                        plt.xlabel('Accuracy')
                        plt.yticks([]) # No y-ticks as it's a single box for all clients in that round
                        plot_filename = os.path.join(self.args.results_dir if hasattr(self.args, 'results_dir') else ".", "final_client_accuracy_distribution.png")
                        plt.savefig(plot_filename)
                        plt.close()
                        print(f"  Client accuracy distribution plot saved to {plot_filename}")
                    else:
                        print("  Could not generate fairness boxplot: No per-round client accuracies recorded.")
                except Exception as e:
                    print(f"  Error generating fairness boxplot: {e}")
        else:
            print("  No fairness data recorded.")

        # Drift Adaptation Capability Summary
        print("\n--- Drift Adaptation Capability ---")
        if self.evaluation_metrics['drift_adaptation_capability']['detected_drifts_per_round']:
            avg_detected_drifts = np.mean(self.evaluation_metrics['drift_adaptation_capability']['detected_drifts_per_round'])
            print(f"  Average Detected Drifts per Round: {avg_detected_drifts:.2f}")
            
            # Filter out NaNs before averaging accuracies for the report
            valid_drifted_accs = [x for x in self.evaluation_metrics['drift_adaptation_capability']['accuracy_drifted_clients_post_adaptation'] if not np.isnan(x)]
            avg_acc_drifted = np.mean(valid_drifted_accs) if valid_drifted_accs else float('nan')
            print(f"  Average Accuracy of Drifted Clients (Post-Adaptation): {avg_acc_drifted:.4f}")
            
            valid_non_drifted_accs = [x for x in self.evaluation_metrics['drift_adaptation_capability']['accuracy_non_drifted_clients'] if not np.isnan(x)]
            avg_acc_non_drifted = np.mean(valid_non_drifted_accs) if valid_non_drifted_accs else float('nan')
            print(f"  Average Accuracy of Non-Drifted Clients: {avg_acc_non_drifted:.4f}")
        else:
            print("  No drift adaptation data collected.")

        # Save detailed evaluation metrics to JSON
        report_filename = os.path.join(self.args.results_dir if hasattr(self.args, 'results_dir') else ".", "comprehensive_evaluation_report.json")
        try:
            with open(report_filename, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_metrics = copy.deepcopy(self.evaluation_metrics)
                for main_key, sub_dict in serializable_metrics.items():
                    for metric_key, history_list in sub_dict.items():
                        if isinstance(history_list, dict): # For client_accuracies_history
                            for round_num, acc_list_in_round in history_list.items():
                                if isinstance(acc_list_in_round, np.ndarray):
                                    sub_dict[metric_key][round_num] = acc_list_in_round.tolist()
                                elif isinstance(acc_list_in_round, list):
                                    sub_dict[metric_key][round_num] = [float(x) if isinstance(x, (np.float32, np.float64)) else x for x in acc_list_in_round]
                        elif isinstance(history_list, list):
                            sub_dict[metric_key] = [float(x) if isinstance(x, (np.float32, np.float64, np.int32, np.int64)) else x for x in history_list]
                            # Handle potential NaNs by converting them to None (JSON compatible)
                            sub_dict[metric_key] = [None if isinstance(x, float) and np.isnan(x) else x for x in sub_dict[metric_key]]

                json.dump(serializable_metrics, f, indent=4)
            print(f"\nDetailed evaluation report saved to {report_filename}")
        except Exception as e:
            print(f"Error saving detailed evaluation report: {e}")

        print("=====================================================")
    
    def _calculate_client_distribution_silhouette(self, client_embeddings, labels, metric='euclidean'):
        """Calculates silhouette score for each client's data distribution within its assigned cluster."""
        if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(client_embeddings):
            return -1 # Silhouette score is not defined for < 2 clusters or n_samples <= n_clusters
        
        # Placeholder: This needs a more sophisticated approach.
        # For now, we average silhouette scores of individual clients if they were clustered.
        # A true distribution-based silhouette would compare distributions, not points.
        # This is a complex research problem. Using standard silhouette as a proxy.
        try:
            return silhouette_score(client_embeddings, labels, metric=metric)
        except ValueError:
            return -1 # Handle cases where silhouette score cannot be computed

    def _calculate_avg_silhouette_for_k(self, client_embeddings, k, client_label_profiles_np, client_weights_np):
        """Runs VWC for a given K and returns the average silhouette score."""
        if k < self.dynamic_k_min or k > self.dynamic_k_max or k > len(client_embeddings):
            return -1, None, None # Invalid k

        temp_vwc_K_t = self.vwc_K_t
        self.vwc_K_t = k # Temporarily set K for this run
        
        # Run a simplified VWC for K evaluation (fewer iterations, no full model training)
        # We need cluster assignments (client_concepts) and centroids
        # This part needs to simulate run_vwc_clustering's core logic for assignment
        # For simplicity, let's assume a light version of VWC or use its clustering part directly
        # This is a placeholder for the actual clustering logic call
        
        # Perform GMM-based clustering (simplified from run_vwc_clustering)
        gmm = GaussianMixture(n_components=k, random_state=self.args.seed, covariance_type='diag')
        try:
            client_concepts = gmm.fit_predict(client_embeddings)
            centroids = gmm.means_
        except ValueError as e:
            print(f"Error during GMM fitting for K={k}: {e}")
            self.vwc_K_t = temp_vwc_K_t # Reset K_t
            return -1, None, None

        self.vwc_K_t = temp_vwc_K_t # Reset K_t

        if self.dynamic_k_silhouette_method == 'avg_client_silhouette':
            score = self._calculate_client_distribution_silhouette(client_embeddings, client_concepts)
        else: # Default or other methods
            score = silhouette_score(client_embeddings, client_concepts)
            
        return score, client_concepts, centroids

    def _initial_k_estimation(self, client_embeddings, client_label_profiles_np, client_weights_np):
        """Phase 1: Estimate initial K_t using silhouette analysis."""
        best_k = self.vwc_K_t # Fallback to user-defined K_t
        best_silhouette_score = -1
        best_concepts = None
        best_centroids = None

        print(f"Dynamic K: Initial K estimation phase. Range: [{self.dynamic_k_min}, {self.dynamic_k_max}]")

        for k_candidate in range(self.dynamic_k_min, min(self.dynamic_k_max + 1, len(client_embeddings) +1), self.dynamic_k_silhouette_range_step):
            if k_candidate <= 1: continue # Silhouette needs at least 2 clusters
            
            # Run VWC clustering for k_candidate (simplified or full)
            # For this, we need a version of run_vwc_clustering that can be called with a specific K
            # and returns client_concepts and centroids.
            # The following is a simplified GMM approach for demonstration.
            # In a real scenario, you'd call a modified run_vwc_clustering or its core logic.
            
            current_silhouette_score, concepts, centroids = self._calculate_avg_silhouette_for_k(
                client_embeddings, k_candidate, client_label_profiles_np, client_weights_np
            )
            
            print(f"  K={k_candidate}, Silhouette Score: {current_silhouette_score:.4f}")

            if current_silhouette_score > best_silhouette_score:
                best_silhouette_score = current_silhouette_score
                best_k = k_candidate
                best_concepts = concepts
                best_centroids = centroids
        
        if best_concepts is None or best_centroids is None:
            # Fallback if no valid K was found (e.g., all silhouette scores were -1)
            print(f"Dynamic K: Initial estimation failed to find a suitable K. Falling back to K={self.vwc_K_t}.")
            # Perform a default clustering with self.vwc_K_t to get initial concepts/centroids
            _, best_concepts, best_centroids = self._calculate_avg_silhouette_for_k(
                client_embeddings, self.vwc_K_t, client_label_profiles_np, client_weights_np
            )
            if best_concepts is None:
                 print("CRITICAL: Fallback K also failed in initial_k_estimation. Using K=2 as emergency.")
                 best_k = 2
                 _, best_concepts, best_centroids = self._calculate_avg_silhouette_for_k(
                    client_embeddings, best_k, client_label_profiles_np, client_weights_np
                 )
                 if best_concepts is None:
                    # Ultimate fallback: assign all to one cluster if everything else fails
                    print("CRITICAL: Emergency K=2 also failed. Assigning all clients to cluster 0.")
                    best_k = 1
                    best_concepts = np.zeros(len(client_embeddings), dtype=int)
                    best_centroids = np.mean(client_embeddings, axis=0, keepdims=True)
            else:
                best_k = self.vwc_K_t # Ensure best_k reflects the fallback K used

        print(f"Dynamic K: Initial K estimated to {best_k} with silhouette score: {best_silhouette_score:.4f}")
        return best_k, best_concepts, best_centroids

    def _calculate_inter_centroid_wasserstein_distance(self, centroid1_profile, centroid2_profile):
        """Calculates Wasserstein distance between the label profiles of two centroids."""
        return wasserstein_distance(centroid1_profile, centroid2_profile)

    def _calculate_cluster_dispersion(self, client_embeddings_in_cluster, cluster_centroid_embedding):
        """Calculates dispersion of a cluster (e.g., average distance to centroid)."""
        if len(client_embeddings_in_cluster) == 0:
            return 0
        distances = np.linalg.norm(client_embeddings_in_cluster - cluster_centroid_embedding, axis=1)
        return np.mean(distances)

    def _attempt_cluster_merge(self, current_k, client_embeddings, client_concepts, centroids, client_label_profiles_np):
        """Phase 2: Attempt to merge clusters."""
        if current_k <= self.dynamic_k_min:
            return current_k, client_concepts, centroids, False 

        effective_centroid_profiles = []
        for i in range(current_k):
            cluster_client_indices = np.where(client_concepts == i)[0]
            if len(cluster_client_indices) > 0:
                avg_profile = np.mean(client_label_profiles_np[cluster_client_indices], axis=0)
                effective_centroid_profiles.append(avg_profile)
            else:
                effective_centroid_profiles.append(np.full(client_label_profiles_np.shape[1], np.inf))

        if len(effective_centroid_profiles) < 2: 
            return current_k, client_concepts, centroids, False

        min_dist = float('inf')
        merge_candidates = (-1, -1)

        for i in range(len(effective_centroid_profiles)):
            for j in range(i + 1, len(effective_centroid_profiles)):
                if np.all(np.isinf(effective_centroid_profiles[i])) or np.all(np.isinf(effective_centroid_profiles[j])):
                    continue
                dist = self._calculate_inter_centroid_wasserstein_distance(effective_centroid_profiles[i], effective_centroid_profiles[j])
                if dist < min_dist:
                    min_dist = dist
                    merge_candidates = (i, j)
        
        if merge_candidates != (-1,-1) and min_dist < self.dynamic_k_merge_wasserstein_threshold:
            c1_idx, c2_idx = merge_candidates
            print(f"Dynamic K: Merging clusters {c1_idx} and {c2_idx} (Wasserstein dist: {min_dist:.4f})")
            
            client_concepts[client_concepts == c2_idx] = c1_idx
            client_concepts[client_concepts > c2_idx] = client_concepts[client_concepts > c2_idx] - 1
            current_k -= 1
            
            new_centroids = np.zeros((current_k, client_embeddings.shape[1]))
            for i in range(current_k):
                cluster_client_indices = np.where(client_concepts == i)[0]
                if len(cluster_client_indices) > 0:
                    new_centroids[i] = np.mean(client_embeddings[cluster_client_indices], axis=0)

            return current_k, client_concepts, new_centroids, True 
        
        return current_k, client_concepts, centroids, False 

    def _attempt_cluster_split(self, current_k, client_embeddings, client_concepts, centroids, client_label_profiles_np):
        """Phase 2: Attempt to split a cluster."""
        if current_k >= self.dynamic_k_max:
            return current_k, client_concepts, centroids, False 

        max_dispersion = -1
        split_candidate_idx = -1

        for i in range(current_k):
            cluster_client_indices = np.where(client_concepts == i)[0]
            if len(cluster_client_indices) < self.dynamic_k_split_min_clients:
                continue 
            
            embeddings_in_cluster = client_embeddings[cluster_client_indices]
            current_cluster_centroid_embedding = np.mean(embeddings_in_cluster, axis=0)
            dispersion = self._calculate_cluster_dispersion(embeddings_in_cluster, current_cluster_centroid_embedding)
            if dispersion > max_dispersion:
                max_dispersion = dispersion
                split_candidate_idx = i
        
        if split_candidate_idx != -1 and max_dispersion > self.dynamic_k_split_dispersion_threshold:
            print(f"Dynamic K: Attempting to split cluster {split_candidate_idx} (Dispersion: {max_dispersion:.4f})")
            
            indices_to_split = np.where(client_concepts == split_candidate_idx)[0]
            embeddings_to_split = client_embeddings[indices_to_split]
            
            if len(embeddings_to_split) < self.dynamic_k_split_subcluster_k: 
                 print(f"Dynamic K: Not enough samples in cluster {split_candidate_idx} to split into {self.dynamic_k_split_subcluster_k} sub-clusters.")
                 return current_k, client_concepts, centroids, False

            gmm_split = GaussianMixture(n_components=self.dynamic_k_split_subcluster_k, random_state=self.args.seed, covariance_type='diag')
            try:
                sub_labels = gmm_split.fit_predict(embeddings_to_split)
            except ValueError:
                print(f"Dynamic K: GMM fit failed for splitting cluster {split_candidate_idx}.")
                return current_k, client_concepts, centroids, False

            if len(np.unique(sub_labels)) < self.dynamic_k_split_subcluster_k:
                print(f"Dynamic K: Split of cluster {split_candidate_idx} did not result in {self.dynamic_k_split_subcluster_k} distinct sub-clusters.")
                return current_k, client_concepts, centroids, False

            print(f"Dynamic K: Splitting cluster {split_candidate_idx} into {self.dynamic_k_split_subcluster_k} sub-clusters.")
            
            new_cluster_label_start = current_k
            for sub_cluster_id in range(self.dynamic_k_split_subcluster_k):
                sub_indices_original = indices_to_split[sub_labels == sub_cluster_id]
                if sub_cluster_id == 0: # First sub-cluster reuses original label
                    client_concepts[sub_indices_original] = split_candidate_idx
                else:
                    # Check if we exceed max_k before assigning new label
                    if new_cluster_label_start + (sub_cluster_id -1) >= self.dynamic_k_max:
                        print(f"Dynamic K: Splitting cluster {split_candidate_idx} aborted, would exceed dynamic_k_max.")
                        # This requires a rollback or careful handling. For now, we abort the split.
                        # To properly handle, we'd need to revert any partial relabeling or not proceed.
                        # Simplest is to return False here if the *next* label would exceed max_k.
                        return current_k, client_concepts, centroids, False # Abort split
                    client_concepts[sub_indices_original] = new_cluster_label_start + (sub_cluster_id - 1)
            
            num_new_clusters_added = self.dynamic_k_split_subcluster_k - 1
            current_k += num_new_clusters_added
            
            new_centroids = np.zeros((current_k, client_embeddings.shape[1]))
            for i in range(current_k):
                cluster_client_indices = np.where(client_concepts == i)[0]
                if len(cluster_client_indices) > 0:
                    new_centroids[i] = np.mean(client_embeddings[cluster_client_indices], axis=0)

            return current_k, client_concepts, new_centroids, True 
            
        return current_k, client_concepts, centroids, False 

    def _refine_k_iteratively(self, current_k, client_embeddings, client_concepts, centroids, client_label_profiles_np):
        """Phase 2: Iteratively merge and split clusters."""
        print(f"Dynamic K: Refining K iteratively. Start K: {current_k}")
        for i in range(self.dynamic_k_refinement_iterations):
            made_change_in_iteration = False
            # Attempt merge
            if current_k > self.dynamic_k_min:
                new_k, new_concepts, new_centroids, merged = self._attempt_cluster_merge(
                    current_k, client_embeddings, client_concepts, centroids, client_label_profiles_np
                )
                if merged:
                    current_k, client_concepts, centroids = new_k, new_concepts, new_centroids
                    made_change_in_iteration = True
                    print(f"  Refinement iter {i+1}: Merged. New K: {current_k}")
            
            # Attempt split
            if current_k < self.dynamic_k_max:
                new_k, new_concepts, new_centroids, split = self._attempt_cluster_split(
                    current_k, client_embeddings, client_concepts, centroids, client_label_profiles_np
                )
                if split:
                    current_k, client_concepts, centroids = new_k, new_concepts, new_centroids
                    made_change_in_iteration = True
                    print(f"  Refinement iter {i+1}: Split. New K: {current_k}")
            
            if not made_change_in_iteration:
                print(f"Dynamic K: Refinement converged after {i+1} iterations.")
                break
        
        print(f"Dynamic K: Final K after refinement: {current_k}")
        return current_k, client_concepts, centroids

    def _perform_label_wise_vwc_clustering_core(self, client_ids_to_cluster, all_client_label_data_subset, drift_status_subset, K_t, 
                                                num_centroid_samples, vwc_max_iter, vwc_reg, vwc_drift_penalty_factor, current_seed,
                                                initial_centroids_vwc=None): # Added initial_centroids_vwc
        """
        Core logic for label-wise Variational Wasserstein Clustering (VWC).
        
        Args:
            client_ids_to_cluster: List of client IDs to perform clustering on.
            all_client_label_data_subset: Dict[client_id, Dict[label, Tuple[samples, losses]]] for the subset.
            drift_status_subset: Dict[client_id, bool] for the subset.
            K_t: Number of clusters.
            num_centroid_samples: Number of samples per label in centroids.
            vwc_max_iter: Max iterations for VWC.
            vwc_reg: Regularization for Sinkhorn distance.
            vwc_drift_penalty_factor: Penalty for drifted clients.
            current_seed: Random seed.
            initial_centroids_vwc: Optional pre-initialized centroids (list of dicts).

        Returns:
            Tuple: (final_assignments, final_centroids)
                   final_assignments: Dict[client_id, cluster_idx]
                   final_centroids: List[Dict[label, np.ndarray_samples]]
        """
        if not client_ids_to_cluster:
            print("VWC Core: No clients to cluster.")
            return {}, []

        if K_t <= 0:
            print(f"VWC Core: Invalid number of clusters K_t = {K_t}. Defaulting all to cluster 0.")
            assignments = {cid: 0 for cid in client_ids_to_cluster}
            return assignments, []

        # Filter clients with valid label data from the subset
        active_client_ids_in_subset = []
        for client_id in client_ids_to_cluster:
            if client_id in all_client_label_data_subset and all_client_label_data_subset[client_id]:
                active_client_ids_in_subset.append(client_id)

        if not active_client_ids_in_subset:
            print("VWC Core: No clients in subset with valid label profiles. Assigning all to cluster 0.")
            assignments = {cid: 0 for cid in client_ids_to_cluster}
            return assignments, []
        
        actual_K_t = min(K_t, len(active_client_ids_in_subset))
        if actual_K_t <= 0: actual_K_t = 1
        # No print for adjustment here as this is a core function, caller can log if K_t changed.
        K_t = actual_K_t

        cluster_centroids_vk_dicts = []
        if initial_centroids_vwc is not None and len(initial_centroids_vwc) == K_t:
            cluster_centroids_vk_dicts = copy.deepcopy(initial_centroids_vwc)
            print(f"VWC Core: Using {len(cluster_centroids_vk_dicts)} provided initial centroids.")
        else:
            if current_seed is not None:
                random.seed(current_seed) # Ensure reproducibility for initialization
            
            num_to_sample_init = min(K_t, len(active_client_ids_in_subset))
            # Ensure sampling is from active_client_ids_in_subset
            initial_client_indices_local = random.sample(range(len(active_client_ids_in_subset)), num_to_sample_init)
            
            for i in range(num_to_sample_init):
                client_id = active_client_ids_in_subset[initial_client_indices_local[i]]
                client_label_profiles = all_client_label_data_subset[client_id]
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
                                padded_samples_list = []
                                num_repeats = num_centroid_samples // current_num_samples
                                remainder = num_centroid_samples % current_num_samples
                                if num_repeats > 0:
                                    padded_samples_list.extend([selected_samples] * num_repeats)
                                if remainder > 0:
                                    padded_samples_list.append(selected_samples[:remainder])
                                if padded_samples_list:
                                     selected_samples = np.vstack(padded_samples_list)
                                else: # Handle case where selected_samples was empty initially or after remainder
                                     selected_samples = np.array([]) # Or handle as error/skip

                    if selected_samples.size > 0: # Ensure samples are still valid after padding
                        initial_centroid[label] = copy.deepcopy(selected_samples)
                
                if initial_centroid:
                    cluster_centroids_vk_dicts.append(initial_centroid)

            # Handle case where we need more centroids (duplicate existing ones)
            while len(cluster_centroids_vk_dicts) < K_t:
                if cluster_centroids_vk_dicts: # Check if list is not empty
                    idx_to_duplicate = len(cluster_centroids_vk_dicts) % len(cluster_centroids_vk_dicts) # Safe modulo
                    cluster_centroids_vk_dicts.append(copy.deepcopy(cluster_centroids_vk_dicts[idx_to_duplicate]))
                else: # Fallback if no centroids could be initialized (e.g. K_t=1, no valid clients)
                    cluster_centroids_vk_dicts.append({}) # Add an empty centroid
                    # This case should ideally be prevented by K_t adjustment or active client checks

        K_t = len(cluster_centroids_vk_dicts) # Update K_t to actual number of initialized centroids
        if K_t == 0:
            print("VWC Core: Could not initialize any cluster centroids. Assigning all to cluster 0.")
            assignments = {cid: 0 for cid in client_ids_to_cluster}
            return assignments, []

        current_assignments = {cid: -1 for cid in active_client_ids_in_subset}
        
        for vwc_iter in range(vwc_max_iter):
            new_assignments = {}
            
            # Assignment Step: Assign each client to closest cluster
            for client_id in active_client_ids_in_subset:
                client_label_profiles = all_client_label_data_subset[client_id]
                min_avg_distance = float('inf')
                assigned_cluster_idx = 0 # Default to 0
                
                if not client_label_profiles: # Skip client if they have no profiles
                    new_assignments[client_id] = assigned_cluster_idx 
                    continue

                for k_idx in range(K_t):
                    if k_idx >= len(cluster_centroids_vk_dicts) or not cluster_centroids_vk_dicts[k_idx]: # Check if centroid exists and is not empty
                        continue # Skip empty or non-existent centroid

                    cluster_centroid = cluster_centroids_vk_dicts[k_idx]
                    total_distance = 0.0
                    num_common_labels = 0
                    
                    for label, (client_samples, _) in client_label_profiles.items():
                        if label in cluster_centroid:
                            centroid_samples = cluster_centroid[label]
                            if (client_samples is not None and client_samples.size > 0 and
                                centroid_samples is not None and centroid_samples.size > 0):
                                distance = compute_sinkhorn_distance_samples(
                                    client_samples, centroid_samples, reg=vwc_reg
                                )
                                if distance != float('inf') and not np.isnan(distance):
                                    total_distance += distance
                                    num_common_labels += 1
                    
                    # Average distance over common labels
                    if num_common_labels > 0:
                        avg_distance = total_distance / num_common_labels
                        
                        # Apply drift penalty if client is drifted
                        if drift_status_subset.get(client_id, False):
                            avg_distance += vwc_drift_penalty_factor * avg_distance # Relative penalty
                        
                        if avg_distance < min_avg_distance:
                            min_avg_distance = avg_distance
                            assigned_cluster_idx = k_idx
                
                new_assignments[client_id] = assigned_cluster_idx

            converged = True
            if vwc_iter > 0: # Only check convergence after first iteration
                for cid in active_client_ids_in_subset: # Iterate only over active clients
                    if current_assignments.get(cid, -1) != new_assignments.get(cid, -1): # Check if key exists
                        converged = False
                        break
                if converged:
                    break
            
            current_assignments = new_assignments

            # Update Step
            new_cluster_centroids_dicts = [{} for _ in range(K_t)] # Initialize with empty dicts
            for k_idx in range(K_t):
                clients_in_this_cluster = [cid for cid, assigned_k in new_assignments.items() if assigned_k == k_idx]
                if not clients_in_this_cluster:
                    if k_idx < len(cluster_centroids_vk_dicts): # Keep old centroid if cluster empty
                         new_cluster_centroids_dicts[k_idx] = copy.deepcopy(cluster_centroids_vk_dicts[k_idx])
                    # else: it remains an empty dict, which is fine
                    continue

                all_labels_in_cluster = set()
                for cid in clients_in_this_cluster:
                    if cid in all_client_label_data_subset: # Ensure client data exists
                        all_labels_in_cluster.update(all_client_label_data_subset[cid].keys())
                
                current_centroid_for_k = {}
                for label in all_labels_in_cluster:
                    pooled_samples_list = []
                    pooled_losses_list = []
                    for cid in clients_in_this_cluster:
                        if cid in all_client_label_data_subset and label in all_client_label_data_subset[cid]:
                            samples, losses = all_client_label_data_subset[cid][label]
                            if (samples is not None and samples.size > 0 and
                                losses is not None and losses.size > 0 and
                                samples.shape[0] == losses.shape[0]):
                                pooled_samples_list.append(samples)
                                pooled_losses_list.append(losses)
                    
                    if pooled_samples_list:
                        combined_samples = np.vstack(pooled_samples_list)
                        combined_losses = np.concatenate(pooled_losses_list)
                        sorted_indices = np.argsort(combined_losses)
                        num_samples_to_take = min(num_centroid_samples, combined_samples.shape[0])
                        centroid_samples_for_label = combined_samples[sorted_indices[:num_samples_to_take]]
                        
                        # Pad samples for this label if fewer than num_centroid_samples
                        current_num_label_samples = centroid_samples_for_label.shape[0]
                        if current_num_label_samples > 0 and current_num_label_samples < num_centroid_samples:
                            padded_label_samples_list = []
                            num_repeats_label = num_centroid_samples // current_num_label_samples
                            remainder_label = num_centroid_samples % current_num_label_samples
                            if num_repeats_label > 0:
                                padded_label_samples_list.extend([centroid_samples_for_label] * num_repeats_label)
                            if remainder_label > 0:
                                padded_label_samples_list.append(centroid_samples_for_label[:remainder_label])
                            if padded_label_samples_list:
                                centroid_samples_for_label = np.vstack(padded_label_samples_list)
                            # else: centroid_samples_for_label remains as is if padding fails
                        if centroid_samples_for_label.size > 0:
                             current_centroid_for_k[label] = centroid_samples_for_label

                new_cluster_centroids_dicts[k_idx] = current_centroid_for_k
            cluster_centroids_vk_dicts = new_cluster_centroids_dicts

        # Final assignments for all clients passed in client_ids_to_cluster
        # (including those who might have been filtered as inactive for clustering steps)
        final_assignments_for_all_input_clients = {}
        for client_id in client_ids_to_cluster:
            if client_id in current_assignments: # current_assignments holds results for active_client_ids_in_subset
                final_assignments_for_all_input_clients[client_id] = current_assignments[client_id]
            else: # Assign to cluster 0 if inactive or not in current_assignments map for some reason
                final_assignments_for_all_input_clients[client_id] = 0
        
        return final_assignments_for_all_input_clients, cluster_centroids_vk_dicts

    def run_vwc_clustering(self, client_label_profiles, client_drift_status, K_t):
        """Orchestrator for Variational Wasserstein Clustering (VWC) with dynamic K determination.
        This method coordinates the overall VWC process, including dynamic adjustment of K,
        and calls the core VWC logic.

        Args:
            client_label_profiles (dict): Client label profiles for clustering.
            client_drift_status (dict): Client drift status indicators.
            K_t (int): Initial number of clusters (can be adjusted dynamically).

        Returns:
            None
        """
        current_seed = getattr(self.args, 'seed', None)

        client_ids = list(client_label_profiles.keys())
        if not client_ids:
            print("VWC Orchestrator: No clients to process for clustering.")
            return

        # Extract and prepare client data for clustering
        all_client_label_data = {}
        for client_id in client_ids:
            client_data = client_label_profiles[client_id]
            if client_data is not None:
                all_client_label_data[client_id] = client_data

        # Step 1: Dynamic K determination (if enabled)
        final_centroids = None # Initialize final_centroids

        if self.dynamic_k_enabled and self.round >= getattr(self.args, 'dynamic_k_start_round', 0):
            print(f"\n--- Dynamic K Adjustment for VWC (Round {self.round}) ---")
            estimated_k, estimated_concepts, estimated_centroids = self._initial_k_estimation(
                client_embeddings, client_label_profiles_np, client_weights_np
            )
            
            if estimated_concepts is not None and estimated_centroids is not None:
                final_k, final_client_concepts, final_centroids = self._refine_k_iteratively(
                    estimated_k, client_embeddings, estimated_concepts, estimated_centroids, client_label_profiles_np
                )
                self.vwc_K_t = final_k
                client_concepts_after_dynamic_k = final_client_concepts
            else:
                print("Dynamic K: Critical failure in initial K estimation, using default VWC K_t.")
                client_concepts_after_dynamic_k = None 

            print(f"--- Dynamic K Adjustment Complete. Final K_t for this round: {self.vwc_K_t} ---")
            
        else: 
            client_concepts_after_dynamic_k = None

        # Step 2: Run the core VWC clustering logic
        try:
            client_concepts, cluster_centroids_vk_dicts = self._perform_label_wise_vwc_clustering_core(
                client_ids, all_client_label_data, client_drift_status, self.vwc_K_t, 
                self.vwc_num_centroid_samples, self.vwc_max_iter, self.vwc_reg, 
                self.vwc_drift_penalty_factor, current_seed, final_centroids
            )
        except Exception as e:
            print(f"Error during VWC clustering: {e}")
            return

        self.client_cluster_assignments = client_concepts

        # Aggregate classifier heads for each cluster
        self.cluster_classifiers.clear()
        client_id_to_datasize_map = {client.id: client.train_samples for client in self.clients if hasattr(client, 'train_samples')}

        # Use self.vwc_K_t as the number of clusters
        for k_idx in range(self.vwc_K_t): 
            clients_assigned_to_k = [cid for cid, assigned_k in self.client_cluster_assignments.items() if assigned_k == k_idx]
            
            if not clients_assigned_to_k:
                # Empty cluster, use global head
                if hasattr(self.global_model, 'head') and self.global_model.head is not None:
                    self.cluster_classifiers[k_idx] = copy.deepcopy(self.global_model.head)
                # else: 
                #    print(f"VWC: Warning - Cluster {k_idx} is empty and global_model.head is None.")
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
                    weighted_sum = torch.zeros_like(classifier_heads_params_list[0][key].cpu(), dtype=torch.float32)
                    total_weight = 0.0

                    for i, params_state_dict in enumerate(classifier_heads_params_list):
                        if key in params_state_dict:
                            param_tensor = params_state_dict[key]
                            if isinstance(param_tensor, torch.Tensor):
                                weighted_sum += param_tensor.cpu().float() * data_sizes_list[i]
                            else: 
                                print(f"VWC: Warning - Parameter {key} for client {clients_assigned_to_k[i]} is not a tensor.")
                                weighted_sum += torch.tensor(param_tensor).cpu().float() * data_sizes_list[i]
                            total_weight += data_sizes_list[i]
                    
                    if total_weight > 0:
                        aggregated_params[key] = weighted_sum / total_weight
                    else:
                        aggregated_params[key] = classifier_heads_params_list[0][key].cpu()
                
                # Create cluster classifier
                if hasattr(self.global_model, 'head') and self.global_model.head is not None:
                    cluster_head = copy.deepcopy(self.global_model.head) 
                    try:
                        cluster_head.load_state_dict(aggregated_params)
                        self.cluster_classifiers[k_idx] = cluster_head
                    except RuntimeError as e:
                        print(f"VWC: Error loading state dict for cluster {k_idx} classifier: {e}")
                        self.cluster_classifiers[k_idx] = copy.deepcopy(self.global_model.head)
                else:
                    print(f"VWC: Warning - global_model.head not found or is None. Cannot create classifier for cluster {k_idx}.")
            else:
                print(f"VWC: No classifier parameters from clients in cluster {k_idx}. Using global head if available.")
                if hasattr(self.global_model, 'head') and self.global_model.head is not None:
                    self.cluster_classifiers[k_idx] = copy.deepcopy(self.global_model.head)

        print(f"VWC: Classifier aggregation complete. Assignments: {self.client_cluster_assignments}")
        # cluster_centroids_vk_dicts is available from the _perform_label_wise_vwc_clustering_core call
        print(f"VWC: {len(self.cluster_classifiers)} cluster classifiers created/updated (Target K: {self.vwc_K_t}, Actual Centroids: {len(cluster_centroids_vk_dicts) if 'cluster_centroids_vk_dicts' in locals() else 'N/A'}).")

        # Step 3: (Optional) Post-processing or adjustments can be added here
