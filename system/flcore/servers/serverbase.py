import torch
import os
import numpy as np
import h5py
import copy
import time
import random
import wandb # Added wandb import

from utils.data_utils import read_client_data
from utils.dlg import DLG


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch = args.fine_tuning_epoch

        self.current_round = 0

    def set_clients(self, clientObj):
        # Setup clients with correct id parameter
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(
                self.args, 
                id=i, 
                train_samples=len(train_data), 
                test_samples=len(test_data), 
                train_slow=train_slow, 
                send_slow=send_slow
            )
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if not self.clients:
            print("Warning: select_clients called but self.clients is empty. Returning empty list.")
            self.current_num_join_clients = 0
            return []

        if self.random_join_ratio:
            # Determine the number of clients to select when random_join_ratio is True.
            # It should be between self.num_join_clients and self.num_clients,
            # but also capped by the actual number of available clients (len(self.clients)).

            # Lower bound for random selection: at least self.num_join_clients (but not more than available)
            min_selectable = min(self.num_join_clients, len(self.clients))
            min_selectable = max(0, min_selectable) # Ensure it's not negative

            # Upper bound for random selection: at most self.num_clients (but not more than available)
            max_selectable = min(self.num_clients, len(self.clients))
            max_selectable = max(0, max_selectable) # Ensure it's not negative
            
            if min_selectable > max_selectable: # Should not happen with correct logic, but as a safeguard
                self.current_num_join_clients = max_selectable
            elif min_selectable == max_selectable:
                self.current_num_join_clients = min_selectable
            else:
                # np.random.choice takes an exclusive upper bound for range
                self.current_num_join_clients = np.random.choice(range(min_selectable, max_selectable + 1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients

        # Final check: ensure current_num_join_clients is not more than available clients and not negative.
        self.current_num_join_clients = min(self.current_num_join_clients, len(self.clients))
        self.current_num_join_clients = max(0, self.current_num_join_clients)

        if self.current_num_join_clients == 0:
            return []
            
        # np.random.choice requires the first argument 'a' to be non-empty if size > 0.
        # self.clients must be non-empty here due to the initial check and current_num_join_clients > 0.
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0 # Initialize tot_samples here
        if not active_clients: # If no clients are active, return early
            return

        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self, current_round=None): # Added current_round parameter
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_filename = self.algorithm + "_server" + ".pt"
        model_filepath = os.path.join(model_path, model_filename)
        torch.save(self.global_model, model_filepath)

        if self.args.save_global_model_to_wandb and wandb.run is not None and current_round is not None:
            try:
                model_artifact = wandb.Artifact(
                    f'{self.args.wandb_run_name_prefix}_global_model',
                    type='model',
                    description=f'Global model for {self.algorithm} at round {current_round}',
                    metadata={'dataset': self.dataset, 'algorithm': self.algorithm, 'round': current_round}
                )
                model_artifact.add_file(model_filepath, name=f'global_model_round_{current_round}.pt')
                aliases = ['latest', f'round_{current_round}']
                if current_round == self.global_rounds: # Mark as final if it's the last round
                    aliases.append('final')
                wandb.log_artifact(model_artifact, aliases=aliases)
                print(f"Global model saved to wandb as artifact at round {current_round}")
            except Exception as e:
                print(f"Error saving model to wandb: {e}")


    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo_filename = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo_filename)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
            
            if self.args.use_wandb and wandb.run is not None:
                try:
                    results_artifact = wandb.Artifact(
                        f'{self.args.wandb_run_name_prefix}_results',
                        type='results',
                        description=f'H5 results file for {self.algorithm}, run {self.times}',
                        metadata={'dataset': self.dataset, 'algorithm': self.algorithm, 'goal': self.goal, 'times': self.times}
                    )
                    results_artifact.add_file(file_path, name=f'{algo_filename}.h5')
                    wandb.log_artifact(results_artifact, aliases=[f'run_{self.times}', 'latest_results'])
                    print(f"Results H5 file saved to wandb as artifact: {algo_filename}.h5")
                except Exception as e:
                    print(f"Error saving results H5 to wandb: {e}")

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None, current_round=None): # Added current_round parameter
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1]) if sum(stats[1]) > 0 else 0
        test_auc = sum(stats[3])*1.0 / sum(stats[1]) if sum(stats[1]) > 0 else 0
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1]) if sum(stats_train[1]) > 0 else 0
        
        # Handle cases where stats[1] might contain zeros, leading to division by zero for individual accs/aucs
        accs = [a / n if n > 0 else 0 for a, n in zip(stats[2], stats[1])]
        aucs = [a / n if n > 0 else 0 for a, n in zip(stats[3], stats[1])]
        
        std_test_acc = np.std(accs)
        std_test_auc = np.std(aucs)

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("Std Test Accurancy: {:.4f}".format(std_test_acc))
        print("Std Test AUC: {:.4f}".format(std_test_auc))

        if wandb.run is not None:
            try:
                wandb.log({
                    "Global Train Loss": train_loss,
                    "Global Test Accuracy": test_acc,
                    "Global Test AUC": test_auc,
                    "Std Test Accuracy": std_test_acc,
                    "Std Test AUC": std_test_auc
                }, step=self.current_round)
            except Exception as e:
                print(f"Error logging metrics to wandb: {e}")

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc

    def apply_drift_transformation(self):
        """统一的概念漂移应用方法，由参数控制漂移时间点"""
        drift_round = getattr(self.args, 'drift_round', 100)  # 默认第100轮
        if self.current_round == drift_round:  # 使用配置的漂移轮次
            for client in self.selected_clients:
                if hasattr(client, 'use_drift_dataset') and client.use_drift_dataset:
                    if hasattr(client, 'apply_drift_transformation'):
                        print(f"Server: Applying drift for client {client.id} at round {self.current_round}")
                        # Apply drift to both training and testing datasets on the client
                        client.apply_drift_transformation()
                    else:
                        print(f"Warning: Client {client.id} is configured to use drift but does not have apply_drift_transformation method.")


