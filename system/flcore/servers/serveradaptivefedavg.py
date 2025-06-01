import numpy as np
import torch
from torch.nn.utils import parameters_to_vector
import copy
import time
import os
import wandb # Assuming wandb might be used as in other server files

from system.flcore.servers.serverbase import Server
from system.flcore.clients.clientadaptivefedavg import AdaptiveFedAvgClient

class AdaptiveFedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # AdaptiveFedAvg specific parameters from args
        self.beta1 = args.beta1 if hasattr(args, 'beta1') else 0.9
        self.beta2 = args.beta2 if hasattr(args, 'beta2') else 0.99
        self.beta3 = args.beta3 if hasattr(args, 'beta3') else 0.9
        self.client_initial_lr = args.local_learning_rate # Store initial LR for clients

        # State for adaptive learning rate calculation
        self.prev_mean = 0.0
        self.prev_mean_norm = 0.0
        self.prev_variance = 0.0
        self.prev_variance_norm = 0.0
        self.prev_ratio = 0.0

        # Set clients using the new client class
        self.set_clients(AdaptiveFedAvgClient)
        self.Budget = []

        print("AdaptiveFedAvg Server initialized.")

    def cal_adaptive_lr(self, current_round):
        # current_round is 0-indexed, add 1 for calculations as in reference
        effective_round = current_round + 1 

        cur_params = parameters_to_vector(self.global_model.parameters()).detach().cpu().numpy()

        mean = self.beta1 * self.prev_mean + (1 - self.beta1) * cur_params
        # Bias correction for mean
        mean_norm = mean / (1 - pow(self.beta1, effective_round)) 

        variance = self.beta2 * self.prev_variance + (1 - self.beta2) * np.mean(
            (cur_params - self.prev_mean_norm) * (cur_params - self.prev_mean_norm)
        )
        # Bias correction for variance
        variance_norm = variance / (1 - pow(self.beta2, effective_round))

        if effective_round == 1:
            # No previous variance to compare against, initialize ratio
            ratio = self.beta3 * self.prev_ratio + (1 - self.beta3) * 1.0 # Initialize with 1.0 or other heuristic
        else:
            if self.prev_variance_norm == 0: # Avoid division by zero
                # Handle case where previous variance was zero (e.g. first few rounds or no change)
                # Could set ratio to 1 or some other default, or skip update for this round
                current_variance_ratio = 1.0 
            else:
                current_variance_ratio = variance_norm / self.prev_variance_norm
            ratio = self.beta3 * self.prev_ratio + (1 - self.beta3) * current_variance_ratio
        
        # Bias correction for ratio
        ratio_norm = ratio / (1 - pow(self.beta3, effective_round))

        # Update state for next round
        self.prev_mean = mean
        self.prev_mean_norm = mean_norm
        self.prev_variance = variance
        self.prev_variance_norm = variance_norm
        self.prev_ratio = ratio

        # Calculate client dynamic learning rate
        # The division by effective_round in reference seems to be an additional decay factor.
        client_dynamic_lr = min(self.client_initial_lr, self.client_initial_lr * ratio_norm / effective_round)
        # Ensure learning rate is not negative or extremely small
        client_dynamic_lr = max(client_dynamic_lr, 1e-6) 

        return client_dynamic_lr

    def train(self):
        for i in range(self.global_rounds):
            self.current_round = i
            s_t = time.time()

            # Select clients
            self.selected_clients = self.select_clients()
            if not self.selected_clients:
                print(f"Round {i}: No clients selected, skipping round.")
                continue

            # Calculate adaptive learning rate for this round
            client_lr_for_round = self.cal_adaptive_lr(i)
            if self.args.use_wandb:
                wandb.log({"AdaptiveFedAvg/ClientLR": client_lr_for_round, "round": i}, step=i)

            # Send global model and adaptive LR to selected clients
            for client in self.selected_clients:
                client.set_parameters(self.global_model) # From serverbase, sends model
                client.learning_rate = client_lr_for_round # Update client's LR
            
            # Clients perform local training
            for client in self.selected_clients:
                client.train() # Client uses its updated self.learning_rate

            # Receive and aggregate models (standard FedAvg aggregation)
            self.receive_models() # From serverbase, populates self.uploaded_models, self.uploaded_weights
            if not self.uploaded_models:
                print(f"Round {i}: No models received from clients, skipping aggregation.")
                # Optionally, could decide not to update prev_mean etc. if no aggregation happens
                # or if cal_adaptive_lr should only use successfully aggregated global models.
                # For now, cal_adaptive_lr uses the global model state *before* this round's aggregation.
                continue
            
            self.aggregate_parameters() # From serverbase, updates self.global_model

            self.Budget.append(time.time() - s_t)
            print(f"Round {i} completed in {self.Budget[-1]:.2f}s. Client LR: {client_lr_for_round:.6f}")

            if (i + 1) % self.eval_gap == 0:
                print(f"\n------------- Round {i} Evaluation -------------")
                # Evaluate uses self.global_model and selected_clients by default
                # It will use the client.test_metrics() which uses the client's current model (just trained)
                # For a more global model evaluation, ensure clients get the latest global model before test_metrics.
                # The current serverbase.evaluate() does not resend models, assumes clients have relevant model.
                # For AdaptiveFedAvg, clients should test the aggregated global model.
                
                # Temporarily update selected clients with the new global model for evaluation
                for client in self.selected_clients: # Or self.clients for all
                    client.set_parameters(self.global_model)
                
                self.evaluate(current_round=i) # serverbase.evaluate()

            if self.auto_break and self.rs_test_acc and self.rs_test_acc[-1] > self.args.goal_accuracy:
                print(f"Reached goal accuracy {self.args.goal_accuracy}. Stopping training.")
                break
        
        print("\nTraining finished.")
        # Final evaluation with all clients using the final global model
        for client in self.clients:
            client.set_parameters(self.global_model)
        self.evaluate(current_round=self.global_rounds) # Evaluate on all clients

        self.save_results()
        self.save_global_model(current_round=self.global_rounds) # Save final model
