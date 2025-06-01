import torch
import numpy as np
import time
import copy
import os
import wandb # Added wandb import
from sklearn.metrics import pairwise_distances # New
from sklearn.cluster import AgglomerativeClustering # New
from flcore.servers.serverbase import Server
from flcore.clients.clientfedrc import clientFedRC # Ensure this client is used

# Helper function to get flattened model parameters as a CPU numpy array.
def get_model_flat_params(model):
    """Helper function to get flattened model parameters as a CPU numpy array."""
    params = []
    # Ensure model parameters are on CPU before converting to numpy
    for param in model.parameters():
        params.append(param.data.cpu().view(-1).numpy())
    if not params:
        return np.array([])
    return np.concatenate(params)

class serverFedRC(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        # FedRC specific parameters
        self.num_clusters = args.num_clusters if hasattr(args, 'num_clusters') else 1
        self.cluster_models = [copy.deepcopy(self.global_model) for _ in range(self.num_clusters)]
        self.client_cluster_assignments = [-1] * self.num_clients # Stores cluster_id for each client_id
        self.cluster_update_frequency = args.cluster_update_frequency if hasattr(args, 'cluster_update_frequency') else 5 # e.g., update clusters every 5 rounds
        self.regularization_lambda = args.fedrc_lambda if hasattr(args, 'fedrc_lambda') else 0.1 # Lambda for regularization term in client training
        
        # self.set_clients should be called before initialize_cluster_assignments
        # if initialize_cluster_assignments needs access to self.clients objects.
        # Serverbase.__init__ calls self.set_slow_clients but not self.set_clients.
        # self.set_clients is called by the algorithm-specific server __init__ usually.

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFedRC) # Use clientFedRC for this server

        # Initialize client cluster assignments (e.g., randomly or based on some initial heuristic)
        self.initialize_cluster_assignments()

    def initialize_cluster_assignments(self):
        # Simple assignment for initialization
        if self.num_clusters <= 0:
            print("Warning: num_clusters is not positive. Skipping initial cluster assignment.")
            return
            
        print("Initializing client cluster assignments...")
        for i in range(self.num_clients):
            assigned_cluster = i % self.num_clusters
            self.client_cluster_assignments[i] = assigned_cluster
            # self.clients is populated by self.set_clients(clientFedRC) call above
            if i < len(self.clients):
                self.clients[i].set_cluster_id(assigned_cluster)
            else:
                # This should ideally not happen if num_clients aligns with actual client objects
                print(f"Warning: Client object with id {i} not found during initial cluster assignment, but assignment stored.")
        # After initial assignment, update cluster models to be averages of their initial members (optional, could also start them all as global_model)
        if self.num_clusters > 0 and len(self.clients) > 0 :
            self.update_all_cluster_models_from_assigned_clients(is_initialization=True)


    def train(self):
        for i in range(self.global_rounds + 1):
            self.current_round = i
            self.selected_clients = self.select_clients()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("Evaluate global model")
                self.evaluate() # Evaluate global model (or cluster models)

            # Update cluster assignments periodically
            if i > 0 and self.num_clusters > 1: # Only cluster if more than 1 cluster
                self.update_client_cluster_assignments()

            # Send cluster-specific models to clients or the global model if no clustering
            self.send_cluster_models()
            
            for client in self.selected_clients:
                if i == 100:  # Condition for drift
                    if hasattr(client, 'use_drift_dataset') and client.use_drift_dataset:
                        if hasattr(client, 'apply_drift_transformation'):
                            print(f"Server: Applying drift for client {client.id} at round {i}")
                            # Apply drift to both training and testing datasets on the client
                            client.apply_drift_transformation()
                        else:
                            print(f"Warning: Client {client.id} is configured to use drift but does not have apply_drift_transformation method.")

                client.train()

            self.receive_models() # Receives models from clients
            
            if self.num_clusters > 0:
                self.aggregate_cluster_parameters() # Aggregate models per cluster
            else:
                self.aggregate_parameters() # Standard FedAvg aggregation if no clusters

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        
        print("\nLastly, evaluate global model")
        self.evaluate()
        self.save_results()
        # self.save_global_model() # Or save cluster models

    def send_cluster_models(self):
        if not self.selected_clients:
            return
            
        for client in self.selected_clients:
            start_time = time.time()
            cluster_id = self.client_cluster_assignments[client.id]
            
            if self.num_clusters > 0 and cluster_id != -1 and cluster_id < len(self.cluster_models):
                model_to_send = self.cluster_models[cluster_id]
                # Client might need to know its regularization strength
                # client.set_regularization_lambda(self.regularization_lambda) 
            else: # Fallback to global model if no clusters or assignment issue
                model_to_send = self.global_model
            
            client.set_parameters(model_to_send)
            
            # The clientfedrc.py might need a method like set_cluster_model_params
            # if it's doing regularization against its cluster's model
            # client.set_cluster_model(model_to_send.parameters())


            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def aggregate_cluster_parameters(self):
        if not self.uploaded_models:
            return

        for cid in range(self.num_clusters):
            cluster_client_models = []
            cluster_client_weights = []
            total_samples_in_cluster = 0

            for client_idx, client_model in enumerate(self.uploaded_models):
                original_client_id = self.uploaded_ids[client_idx]
                if self.client_cluster_assignments[original_client_id] == cid:
                    cluster_client_models.append(client_model)
                    # Use actual train_samples from client object for weighting
                    # This requires uploaded_ids to map back to client objects or storing samples during upload
                    # For simplicity, using uploaded_weights if they represent samples, otherwise adjust
                    # Assuming self.uploaded_weights[client_idx] is proportional to samples
                    
                    # Find the client object to get its train_samples
                    client_obj = next((c for c in self.clients if c.id == original_client_id), None)
                    if client_obj:
                        samples = client_obj.train_samples
                        cluster_client_weights.append(samples)
                        total_samples_in_cluster += samples
                    else: # Fallback if client object not found, though this shouldn't happen
                        cluster_client_weights.append(1) # Equal weight as a fallback
                        total_samples_in_cluster += 1


            if not cluster_client_models: # Skip if no clients in this cluster contributed
                continue

            # Normalize weights within the cluster
            if total_samples_in_cluster > 0:
                normalized_weights = [w / total_samples_in_cluster for w in cluster_client_weights]
            else: # Avoid division by zero if no samples (e.g. all clients dropped)
                normalized_weights = [1.0 / len(cluster_client_weights)] * len(cluster_client_weights) if cluster_client_models else []


            # Aggregate for this cluster
            # Initialize cluster model parameters to zero
            for param in self.cluster_models[cid].parameters():
                param.data.zero_()
            
            for weight, client_model in zip(normalized_weights, cluster_client_models):
                for server_param, client_param in zip(self.cluster_models[cid].parameters(), client_model.parameters()):
                    server_param.data += client_param.data.clone() * weight
        
        # Optionally, update the global_model as an average of cluster models or by some other logic
        # For now, global_model is not explicitly updated from cluster models in this FedRC sketch

    def update_client_cluster_assignments(self):
        print(f"Round {self.current_round}: Updating client cluster assignments...")

        if self.num_clusters <= 1 or len(self.clients) < self.num_clusters:
            print("Not enough clients or clusters to perform clustering. Skipping.")
            # If num_clusters is 1, ensure all clients are in cluster 0.
            if self.num_clusters == 1:
                assignments_changed_to_single_cluster = False
                for client_obj in self.clients:
                    if self.client_cluster_assignments[client_obj.id] != 0:
                        assignments_changed_to_single_cluster = True
                    self.client_cluster_assignments[client_obj.id] = 0
                    client_obj.set_cluster_id(0)
                if assignments_changed_to_single_cluster:
                    print("All clients assigned to cluster 0.")
                    self.update_all_cluster_models_from_assigned_clients()
            return

        client_params_list = []
        # valid_client_indices maps position in client_params_list to original index in self.clients
        valid_client_original_indices = [] 
        
        for original_idx, client_obj in enumerate(self.clients):
            try:
                # client.model should contain the latest parameters for that client
                flat_params = get_model_flat_params(client_obj.model)
                if flat_params.size == 0: # Check if params are empty
                    print(f"Warning: Client {client_obj.id} has no model parameters. Skipping for clustering.")
                    continue
                client_params_list.append(flat_params)
                valid_client_original_indices.append(original_idx)
            except Exception as e:
                print(f"Warning: Could not get parameters for client {client_obj.id}: {e}. Skipping for clustering.")
                continue
        
        if len(client_params_list) < self.num_clusters:
            print(f"Not enough valid client models ({len(client_params_list)}) to perform clustering for {self.num_clusters} clusters. Skipping.")
            return

        client_params_matrix = np.array(client_params_list)

        # Calculate pairwise cosine distances (1 - cosine_similarity)
        try:
            distance_matrix = pairwise_distances(client_params_matrix, metric='cosine')
            # Handle potential NaN/inf values in distance_matrix if models are identical (distance 0) or params are zero
            if np.isnan(distance_matrix).any() or np.isinf(distance_matrix).any():
                print("Warning: NaN or Inf found in distance matrix. Attempting to clean.")
                distance_matrix = np.nan_to_num(distance_matrix, nan=1.0, posinf=1.0, neginf=1.0) # Replace NaN/inf
        except Exception as e:
            print(f"Error calculating distance matrix: {e}. Skipping cluster update.")
            return

        # Perform clustering
        clustering_algo = AgglomerativeClustering(
            n_clusters=self.num_clusters, 
            metric='precomputed', # Changed 'affinity' to 'metric'
            linkage='average' # 'average' linkage is common
        )
        try:
            cluster_labels = clustering_algo.fit_predict(distance_matrix)
        except Exception as e:
            print(f"Error during clustering: {e}. Skipping cluster update.")
            return

        assignments_changed = False
        for i, new_label in enumerate(cluster_labels):
            client_original_idx = valid_client_original_indices[i]
            client_obj = self.clients[client_original_idx]
            if self.client_cluster_assignments[client_obj.id] != new_label:
                assignments_changed = True
            self.client_cluster_assignments[client_obj.id] = new_label
            client_obj.set_cluster_id(new_label)
        
        if assignments_changed:
            print("Client cluster assignments updated.")
            # After assignments change, cluster models should be re-calculated
            # using all clients in their new clusters.
            self.update_all_cluster_models_from_assigned_clients()
        else:
            print("No changes in client cluster assignments.")

    def update_all_cluster_models_from_assigned_clients(self, is_initialization=False):
        if not is_initialization: # Only print if not during init
            print("Recomputing cluster models based on new assignments from all clients.")
        
        if self.num_clusters <= 0: return

        for cid in range(self.num_clusters):
            cluster_member_models = []
            cluster_member_samples = [] # For weighted averaging - Initialize as empty list
            
            for client_obj in self.clients: # Iterate over all clients
                if self.client_cluster_assignments[client_obj.id] == cid:
                    cluster_member_models.append(client_obj.model)
                    cluster_member_samples.append(client_obj.train_samples)
    
            if not cluster_member_models:
                print(f"Warning: Cluster {cid} has no clients after assignment. Resetting to global model.")
                # Reset to a copy of the global model or a fresh model
                self.cluster_models[cid] = copy.deepcopy(self.global_model) 
                # Or, if global_model is also potentially stale, re-init from args.model:
                # self.cluster_models[cid] = copy.deepcopy(self.args.model)
                # self.cluster_models[cid].to(self.device)
                continue
    
            # Weighted average for this cluster
            total_samples_in_cluster = sum(cluster_member_samples)
            
            current_cluster_model_device = next(self.cluster_models[cid].parameters()).device

            if total_samples_in_cluster == 0: # Avoid division by zero
                if cluster_member_models: # If models exist but no samples, use equal weighting
                    normalized_weights = [1.0 / len(cluster_member_models)] * len(cluster_member_models)
                else: # Should have been caught by "if not cluster_member_models"
                    continue 
            else:
                normalized_weights = [s / total_samples_in_cluster for s in cluster_member_samples]
    
            # Initialize new cluster model parameters to zero
            for param in self.cluster_models[cid].parameters():
                param.data.zero_()
            
            for weight, client_model_for_agg in zip(normalized_weights, cluster_member_models):
                for server_param, client_param in zip(self.cluster_models[cid].parameters(), client_model_for_agg.parameters()):
                    # Ensure client_param is on the same device as server_param before operation
                    server_param.data += client_param.data.clone().to(current_cluster_model_device) * weight
        if not is_initialization:
            print("Cluster models recomputed.")

    def evaluate(self, acc=None, loss=None): # Overriding to potentially evaluate per cluster
        if self.num_clusters > 0:
            # Evaluate each cluster model on the test data of clients in that cluster
            # Or evaluate all cluster models on all test data (less common for personalization)
            # For simplicity, let's report average performance of cluster models on their respective clients
            
            all_stats_collector = {'num_samples': [], 'tot_correct': [], 'tot_auc': []}
            all_stats_train_collector = {'num_samples': [], 'losses': []}

            for cid in range(self.num_clusters):
                cluster_clients = [client for client in self.clients if self.client_cluster_assignments[client.id] == cid]
                if not cluster_clients:
                    continue

                # Temporarily set the global_model to the current cluster_model for evaluation
                # This is a bit of a hack; ideally, client.test_metrics() would take a model
                original_global_model_state = copy.deepcopy(self.global_model.state_dict())
                self.global_model.load_state_dict(self.cluster_models[cid].state_dict())
                
                num_samples_cluster, tot_correct_cluster, tot_auc_cluster = [], [], []
                num_samples_train_cluster, losses_train_cluster = [], []

                for client in cluster_clients:
                    # Ensure client uses the correct (cluster) model for its test_metrics
                    client.set_parameters(self.cluster_models[cid]) # Important!
                    ct, ns, auc_val = client.test_metrics()
                    tot_correct_cluster.append(ct * 1.0)
                    tot_auc_cluster.append(auc_val * ns)
                    num_samples_cluster.append(ns)

                    cl, ns_train = client.train_metrics() # client.train_metrics uses client.model
                    losses_train_cluster.append(cl * 1.0)
                    num_samples_train_cluster.append(ns_train)

                all_stats_collector['num_samples'].extend(num_samples_cluster)
                all_stats_collector['tot_correct'].extend(tot_correct_cluster)
                all_stats_collector['tot_auc'].extend(tot_auc_cluster)
                all_stats_train_collector['num_samples'].extend(num_samples_train_cluster)
                all_stats_train_collector['losses'].extend(losses_train_cluster)
                
                # Restore global model state if it was changed
                self.global_model.load_state_dict(original_global_model_state)

                # Log per-cluster metrics if desired
                # test_acc_c = sum(tot_correct_cluster) / sum(num_samples_cluster) if sum(num_samples_cluster) > 0 else 0
                # print(f"Cluster {cid} Test Accuracy: {test_acc_c:.4f}")

            # Calculate overall federated metrics from collected stats
            test_acc_overall = sum(all_stats_collector['tot_correct']) / sum(all_stats_collector['num_samples']) if sum(all_stats_collector['num_samples']) > 0 else 0
            test_auc_overall = sum(all_stats_collector['tot_auc']) / sum(all_stats_collector['num_samples']) if sum(all_stats_collector['num_samples']) > 0 else 0
            train_loss_overall = sum(all_stats_train_collector['losses']) / sum(all_stats_train_collector['num_samples']) if sum(all_stats_train_collector['num_samples']) > 0 else 0
            
            # std_test_acc = np.std([c/n if n>0 else 0 for c,n in zip(all_stats_collector['tot_correct'], all_stats_collector['num_samples'])])
            # std_test_auc = np.std([a/n if n>0 else 0 for a,n in zip(all_stats_collector['tot_auc'], all_stats_collector['num_samples'])])


            if acc is None: self.rs_test_acc.append(test_acc_overall)
            else: acc.append(test_acc_overall)
            if loss is None: self.rs_train_loss.append(train_loss_overall)
            else: loss.append(train_loss_overall)
            # self.rs_test_auc can be appended similarly if needed for overall AUC

            print(f"Averaged Train Loss (across clusters): {train_loss_overall:.4f}")
            print(f"Averaged Test Accuracy (across clusters): {test_acc_overall:.4f}")
            print(f"Averaged Test AUC (across clusters): {test_auc_overall:.4f}")
            # print(f"Std Test Accuracy (across clusters): {std_test_acc:.4f}")
            # print(f"Std Test AUC (across clusters): {std_test_auc:.4f}")

            if self.args.use_wandb and wandb.run is not None:
                 wandb.log({
                    "Global Train Loss": train_loss_overall,
                    "Global Test Accuracy": test_acc_overall,
                    "Global Test AUC": test_auc_overall,
                    # "Std Test Accuracy": std_test_acc,
                    # "Std Test AUC": std_test_auc
                }, step=self.current_round)

        else: # No clustering, fall back to base server evaluation
            super().evaluate(acc=acc, loss=loss, current_round=self.current_round)

    # Need to ensure that the main.py can pass 'num_clusters' and 'cluster_update_frequency'
    # and 'fedrc_lambda' from args to the server.
    # This might require adding these to the argparse setup in main.py and config.yaml.
