import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader
import copy

from system.flcore.clients.clientbase import Client 
# Assuming clientbase.py is in system.flcore.clients.clientbase
# from ..clientbase import Client # If relative import is preferred and structure allows

class AdaptiveFedAvgClient(Client):
    def __init__(self, args, id, train_slow, send_slow):
        # serverbase.set_clients calls client(args, id, train_slow=bool, send_slow=bool)
        # clientbase.Client.__init__ expects (args, id, train_samples, test_samples, **kwargs)
        # We pass 0 for train_samples and test_samples, assuming Client base class
        # might internally use len(self.train_data) or that these are not critical if 0.
        # train_slow and send_slow are passed as kwargs to super.
        super().__init__(args, id, train_samples=0, test_samples=0, train_slow=train_slow, send_slow=send_slow)
        
        # learning_rate is initialized in Client base class from args.local_learning_rate.
        # The server will update this attribute each round.
        # self.lr = args.local_learning_rate # Redundant, already in self.learning_rate from base
        self.args = args # Store args for weight_decay, momentum if needed by optimizer

    def train(self):
        self.model.train()
        train_loader = self.load_train_data() # Uses batch_size from self.batch_size (from args)

        # Optimizer is created here using the current self.learning_rate set by the server
        optimizer = optim.SGD(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.args.weight_decay if hasattr(self.args, 'weight_decay') else 0,
            momentum=self.args.momentum if hasattr(self.args, 'momentum') else 0
        )

        for epoch in range(self.local_epochs):
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss(outputs, labels) # self.loss is from Client base (nn.CrossEntropyLoss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # self.model is updated in place.
        # No learning_rate_scheduler.step() as LR is controlled by server.
