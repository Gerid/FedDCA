import torch
import torch.nn as nn
import numpy as np
import time
import copy
from flcore.clients.clientbase import Client

class clientFedRC(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        # FedRC specific parameters
        self.cluster_id = -1 # Initially, client is not assigned to any cluster
        # Placeholder for any regularizer term or specific loss components for FedRC
        self.regularization_term = None 
        # May need to store cluster-specific model or parameters
        self.cluster_model = None
        # Initialize regularization_lambda from args, with a default if not provided
        self.regularization_lambda = args.fedrc_lambda if hasattr(args, 'fedrc_lambda') else 0.1 

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        for epoch in range(self.local_epochs): # Corrected loop variable
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                
                # Add FedRC specific regularization
                if self.cluster_model is not None and self.regularization_lambda > 0:
                    reg_term = self.calculate_regularization_term()
                    loss += self.regularization_lambda * reg_term

                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_cluster_id(self, cluster_id):
        self.cluster_id = cluster_id

    def get_cluster_id(self):
        return self.cluster_id

    def calculate_regularization_term(self):
        # Calculate L2 norm of the difference between client model and cluster model parameters
        if self.cluster_model is None:
            return 0
        
        reg_loss = 0.0
        # Ensure cluster_model is on the same device as the client model
        # self.cluster_model.to(self.device) # This might not be needed if params are just tensors

        for client_param, cluster_param_val in zip(self.model.parameters(), self.cluster_model_params_cache):
            # cluster_param_val is already a tensor on the correct device (from server)
            reg_loss += torch.norm(client_param - cluster_param_val.to(client_param.device), p=2)**2 # Using squared L2 norm
        return reg_loss / 2.0 # Common to divide by 2

    def set_cluster_model_params(self, cluster_model_params):
        # Stores the raw parameters received from the server
        # These parameters will be used in calculate_regularization_term
        # No need to deepcopy a full model, just store the parameters
        self.cluster_model_params_cache = [param.clone().detach().requires_grad_(False) for param in cluster_model_params]
        
        # For FedRC, the client usually continues training its own model,
        # but regularizes towards the cluster model.
        # So, we don't overwrite self.model with the cluster model here.
        # We create a placeholder self.cluster_model if needed for other logic,
        # but the regularization term will use self.cluster_model_params_cache.
        if self.cluster_model is None:
            self.cluster_model = copy.deepcopy(self.model) # Create a structure
        
        # If you need to load these params into self.cluster_model (e.g., for other purposes):
        # current_device = next(self.model.parameters()).device
        # for target_param, source_param in zip(self.cluster_model.parameters(), cluster_model_params):
        #     target_param.data = source_param.data.clone().to(current_device)

