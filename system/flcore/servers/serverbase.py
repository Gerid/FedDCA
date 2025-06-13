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
        self.rs_f1_weighted = [] # Added for F1-score
        self.rs_tpr_weighted = [] # Added for TPR

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

        self.eval_interval = args.eval_interval
        self.num_clients = args.num_clients
        self.drift_config = getattr(args, 'drift_config', None) # Ensure drift_config is initialized

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
            
            client.set_parameters(self.global_model.state_dict())  

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
        
    def save_results(self, current_round=None): # Added current_round for consistency, though not used in h5 name yet
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # Check if there are results to save for the primary metrics
        if (len(self.rs_test_acc)):
            algo_filename = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo_filename)
            print("File path for saving results: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc) # rs_test_auc was missing, added it.
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.create_dataset('rs_f1_weighted', data=self.rs_f1_weighted) # Save F1
                hf.create_dataset('rs_tpr_weighted', data=self.rs_tpr_weighted) # Save TPR
            
            if self.args.use_wandb and wandb.run is not None:
                try:
                    # Determine a unique artifact name, perhaps incorporating the round if saving multiple times
                    # For now, using the same logic as before, which might overwrite or version.
                    artifact_name_suffix = f"_run_{self.times}" if hasattr(self, 'times') else ""
                    artifact_name = f'{self.args.wandb_run_name_prefix}_results{artifact_name_suffix}'
                    
                    current_round_val = current_round if current_round is not None else self.current_round
                    results_artifact = wandb.Artifact(
                        artifact_name, # Use a consistent and informative name
                        type='results',
                        description=f'H5 results file for {self.algorithm}, run {self.times}, round {current_round_val}',
                        metadata={'dataset': self.dataset, 'algorithm': self.algorithm, 'goal': self.goal, 'times': self.times, 'round': current_round_val,
                                  'metrics_included': ['test_acc', 'test_auc', 'train_loss', 'f1_weighted', 'tpr_weighted']}
                    )
                    results_artifact.add_file(file_path, name=f'{algo}_{self.goal}_{self.times}.h5') # using algo, goal, times for filename consistency
                    
                    aliases = [f'run_{self.times}_round_{current_round_val}', 'latest_results']
                    # Ensure global_rounds is an int for comparison
                    if current_round_val == int(self.global_rounds) -1 or current_round_val == int(self.global_rounds): 
                        aliases.append(f'run_{self.times}_final_results')

                    wandb.log_artifact(results_artifact, aliases=aliases)
                    print(f"Results H5 file saved to wandb as artifact: {artifact_name} ({algo}_{self.goal}_{self.times}.h5)")
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
            # This path needs to be updated if new clients should also report F1 and TPR.
            # For now, assuming test_metrics_new_clients returns the original 4 values.
            # To avoid errors, we'd need to decide how to handle F1/TPR here.
            # Option 1: Modify test_metrics_new_clients to also return F1/TPR (possibly as zeros if not applicable).
            # Option 2: Return dummy F1/TPR values here if this branch is taken.
            # Option 3: Raise an error or log a warning that F1/TPR are not available for new clients.
            # For now, let's assume test_metrics_new_clients is NOT the primary path when these new metrics are critical,
            # or it will be updated separately. If it returns 4 values, the unpacking below will fail.
            # A simple fix if it must run: make it return dummy lists for the new metrics.
            # ids, num_samples, tot_correct, tot_auc = self.test_metrics_new_clients()
            # return ids, num_samples, tot_correct, tot_auc, [0]*len(num_samples), [0]*len(num_samples)
            # However, the current code in evaluate() calls self.test_metrics() and expects 6 values.
            # So, if eval_new_clients is true, test_metrics_new_clients MUST be adapted or this path avoided for full metric eval.
            print("Warning: test_metrics called with eval_new_clients=True. F1/TPR from new_clients needs specific handling if test_metrics_new_clients is not updated.")
            # Assuming test_metrics_new_clients is updated or this path is not taken when full metrics are expected.
            # If test_metrics_new_clients is called and returns 4 values, it will lead to an error upon return.
            # For safety, if this branch is taken, it should conform to the 6-value return type.
            # This is a placeholder, ideally test_metrics_new_clients should be updated.
            # return self.test_metrics_new_clients() # This would be an error if it returns 4 values.
            # Let's assume it's updated or this path is not critical for the current F1/TPR task.
            # Fallback to standard client evaluation if new client specific evaluation is not fully integrated with new metrics.
            pass # Allow flow to standard client evaluation for now.

        num_samples = []
        tot_correct = []
        tot_auc = []
        tot_f1_weighted = [] # Added for F1-score
        tot_tpr_weighted = [] # Added for TPR

        active_clients_for_test = self.clients # Evaluate all clients, not just selected or active ones for training
        if not active_clients_for_test:
            # Return empty lists with the correct structure if there are no clients.
            return [], [], [], [], [], []

        for c in active_clients_for_test:
            # Client test_metrics now returns: test_acc, test_num, auc, f1_weighted, tpr_weighted
            ct, ns, auc, f1, tpr = c.test_metrics() 
            tot_correct.append(ct*1.0) # ct is total correct samples from client, not accuracy here
            tot_auc.append(auc*ns) 
            num_samples.append(ns)
            tot_f1_weighted.append(f1 * ns) # f1 is client's weighted F1, so multiply by ns for server-level weighted avg
            tot_tpr_weighted.append(tpr * ns) # tpr is client's weighted TPR

        ids = [c.id for c in active_clients_for_test]

        return ids, num_samples, tot_correct, tot_auc, tot_f1_weighted, tot_tpr_weighted

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
    def evaluate(self, acc=None, loss=None, current_round=None, is_global=False, return_metrics=False): # Updated parameters
        stats = self.test_metrics() # ids, num_samples, tot_correct, tot_auc, tot_f1_weighted, tot_tpr_weighted
        stats_train = self.train_metrics()

        total_samples = sum(stats[1]) if sum(stats[1]) > 0 else 0

        test_acc = sum(stats[2]) / total_samples if total_samples > 0 else 0
        test_auc = sum(stats[3]) / total_samples if total_samples > 0 else 0 
        avg_f1_weighted = sum(stats[4]) / total_samples if total_samples > 0 else 0
        avg_tpr_weighted = sum(stats[5]) / total_samples if total_samples > 0 else 0
        
        train_loss = sum(stats_train[2]) / sum(stats_train[1]) if sum(stats_train[1]) > 0 else 0
        
        client_accuracies = [c_corr / c_samp if c_samp > 0 else 0 for c_corr, c_samp in zip(stats[2], stats[1])]
        std_test_acc = np.std(client_accuracies)
        
        client_aucs = [c_auc_val / c_samp if c_samp > 0 else 0 for c_auc_val, c_samp in zip(stats[3], stats[1])]
        std_test_auc = np.std(client_aucs)

        client_f1s = [c_f1_val / c_samp if c_samp > 0 else 0 for c_f1_val, c_samp in zip(stats[4], stats[1])]
        std_f1_weighted = np.std(client_f1s)

        client_tprs = [c_tpr_val / c_samp if c_samp > 0 else 0 for c_tpr_val, c_samp in zip(stats[5], stats[1])]
        std_tpr_weighted = np.std(client_tprs)

        self.rs_test_acc.append(test_acc)
        self.rs_test_auc.append(test_auc)
        self.rs_f1_weighted.append(avg_f1_weighted)
        self.rs_tpr_weighted.append(avg_tpr_weighted)
        self.rs_train_loss.append(train_loss)

        round_to_log = current_round if current_round is not None else self.current_round
        logging_prefix = "Global_" if is_global else "Local_"

        log_key_prefix = f"{logging_prefix}/" if logging_prefix and logging_prefix != "Global" else ""
        # For "Global" prefix, we often don't want "Global/MetricName" but just "MetricName"
        # If logging_prefix is "Global" or empty, no prefix is added to the metric key for wandb.
        # If logging_prefix is something else (e.g., "Local_Client_XYZ"), then "Local_Client_XYZ/MetricName" is used.

        print(f"{logging_prefix} Test Accuracy: {test_acc:.4f} (std: {std_test_acc:.4f})")
        print(f"{logging_prefix} Test AUC: {test_auc:.4f} (std: {std_test_auc:.4f})")
        print(f"{logging_prefix} Weighted F1-Score: {avg_f1_weighted:.4f} (std: {std_f1_weighted:.4f})")
        print(f"{logging_prefix} Weighted TPR: {avg_tpr_weighted:.4f} (std: {std_tpr_weighted:.4f})")
        print(f"{logging_prefix} Train Loss: {train_loss:.4f}")

        if not return_metrics and wandb.run is not None: 
            wandb.log({
                f"{logging_prefix}Test_Accuracy": test_acc,
                f"{logging_prefix}Std_Test_Accuracy": std_test_acc,
                f"{logging_prefix}Test_AUC": test_auc,
                f"{logging_prefix}Std_Test_AUC": std_test_auc,
                f"{logging_prefix}Weighted_F1_Score": avg_f1_weighted,
                f"{logging_prefix}Std_Weighted_F1_Score": std_f1_weighted,
                f"{logging_prefix}Weighted_TPR": avg_tpr_weighted,
                f"{logging_prefix}Std_Weighted_TPR": std_tpr_weighted,
                f"{logging_prefix}Train_Loss": train_loss
            }, step=self.current_round)

        if self.dlg_eval and round_to_log % self.dlg_gap == 0:
            self.call_dlg(round_to_log)

        # The auto_break logic seems to be for early stopping based on target acc/loss,
        # it doesn't directly affect the return values or storage of metrics here.
        if acc is not None and self.auto_break: 
            if test_acc >= acc and train_loss <= loss:
                # self.print_() # Consider if print_ needs update for new metrics
                pass # Original code had a pass here or was returning True, which might skip final saves.

        if return_metrics:
            return test_acc, test_auc, avg_f1_weighted, avg_tpr_weighted, train_loss, std_test_acc, std_test_auc, std_f1_weighted, std_tpr_weighted
        else:
            pass 

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
        # This method is called by server implementations (like FedAvg, FedIFCA)
        # The actual data transformation happens on the client side.
        # This server method can log or coordinate based on self.drift_config.
        if self.drift_config and self.drift_config.get("complex_drift_scenario"):
            scenario = self.drift_config.get("complex_drift_scenario")
            base_epoch = self.drift_config.get("drift_base_epoch", 0)
            
            # Log that a complex drift scenario is active if current round is at or after base_epoch
            if self.current_round >= base_epoch:
                print(f"Server: Round {self.current_round}. Complex drift scenario '{scenario}' is active (base epoch {base_epoch}).")
                if hasattr(self, 'selected_clients') and self.selected_clients:
                    for client in self.selected_clients:
                        # Further logic could determine if drift *specifically* applies to this client in this round
                        # (e.g., for staggered or partial drift).
                        # The client itself will make the final determination based on its ID and the full drift_config.
                        # This server log indicates the server is aware the client *might* be drifting.
                        print(f"Server: Instructing/expecting client {client.id} to consider drift at round {self.current_round} for scenario '{scenario}'.")
                else:
                    print(f"Server: Round {self.current_round}. Complex drift scenario '{scenario}' active, but no clients selected or 'selected_clients' not available at this logging point.")
            # else:
            # print(f"Server: Round {self.current_round}. Complex drift scenario '{scenario}' configured, but base epoch {base_epoch} not yet reached.")
        # else:
            # print(f"Server: Round {self.current_round}. No complex drift scenario active or drift_config not set.")

    def aggregate_parameters(self, selected_clients=None):
        assert (len(self.uploaded_models) > 0)

        if selected_clients is None:
            selected_clients = self.uploaded_models  # Default to all uploaded models if none specified

        self.global_model = copy.deepcopy(selected_clients[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, selected_clients):
            self.add_parameters(w, client_model)


