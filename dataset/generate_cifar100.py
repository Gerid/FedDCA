import os
import json
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file
import random # For concept generation
import sys # For __main__

dir_path = "Cifar100/"
random.seed(1)
np.random.seed(1)

# Allocate data to users
def generate_cifar100(data_dir, client_count, class_count, is_niid, is_balanced, partition_type):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    # Setup directory for train/test data
    config_path = data_dir + "config.json"
    train_path = data_dir + "train/"
    test_path = data_dir + "test/"

    if check(config_path, train_path, test_path, client_count, is_niid, is_balanced, partition_type):
        return
        
    # Get Cifar100 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(
        root=data_dir+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(
        root=data_dir+"rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    X, y, statistic = separate_data((dataset_image, dataset_label), client_count, class_count, 
                                    is_niid, is_balanced, partition_type, class_per_client=20)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, client_count, class_count, 
        statistic, is_niid, is_balanced, partition_type)


def apply_label_swaps(original_labels, swaps):
    labels = np.copy(original_labels)
    for class_a, class_b in swaps:
        mask_a = (labels == class_a)
        mask_b = (labels == class_b)
        labels[mask_a] = class_b
        labels[mask_b] = class_a
    return labels

def generate_cifar100_with_clusters(data_dir, client_count, class_count, is_niid, is_balanced, partition_type, num_concepts=5, iterations=200, num_drifts=3):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    drift_info_path = os.path.join(data_dir, "drift_info")
    if not os.path.exists(drift_info_path):
        os.makedirs(drift_info_path)
    concept_config_path = os.path.join(drift_info_path, "concept_config.json")

    loaded_config_values = False
    concept_definitions = {}
    client_concepts = {}
    client_drift_types = {}
    drift_iterations = np.array([])
    client_concept_trajectories = {}

    if os.path.exists(concept_config_path):
        try:
            with open(concept_config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            print(f"Successfully loaded concept configuration from {concept_config_path}")

            loaded_num_concepts = config_data.get('num_concepts', num_concepts)
            num_concepts = loaded_num_concepts
            client_count = config_data.get('client_count', client_count)
            class_count = config_data.get('class_count', class_count)
            iterations = config_data.get('iterations', iterations)
            num_drifts = config_data.get('num_drifts', num_drifts)
            
            concept_definitions = config_data['concept_definitions']
            client_concepts = {int(k): v for k, v in config_data['client_concepts'].items()}
            client_drift_types = {int(k): v for k, v in config_data['client_drift_types'].items()}
            drift_iterations = np.array(config_data['drift_iterations'])
            client_concept_trajectories = {int(k): v for k, v in config_data['client_concept_trajectories'].items()}
            
            loaded_config_values = True
        except Exception as e:
            print(f"Warning: Could not load or parse {concept_config_path}: {e}. A new configuration will be generated.")
            loaded_config_values = False
    
    if not loaded_config_values:
        print(f"Generating new concept configuration for data_dir: {data_dir}...")
        
        concept_definitions['0'] = [] # Concept 0: no swaps
        for concept_id in range(1, num_concepts):
            num_swaps_this_concept = np.random.randint(1, 3) # 1 or 2 pairs of swaps
            current_swaps = []
            available_labels_for_swapping = list(range(class_count))
            random.shuffle(available_labels_for_swapping)
            
            for _ in range(num_swaps_this_concept):
                if len(available_labels_for_swapping) < 2:
                    break 
                label_a = available_labels_for_swapping.pop()
                label_b = available_labels_for_swapping.pop()
                current_swaps.append(sorted([int(label_a), int(label_b)]))
            concept_definitions[str(concept_id)] = current_swaps
        
        for client_id in range(client_count):
            num_concepts_per_client = np.random.randint(1, min(num_concepts, 3) + 1) # Client uses 1 to 3 concepts
            possible_concept_ids = list(range(num_concepts))
            random.shuffle(possible_concept_ids)
            client_concepts[client_id] = sorted(possible_concept_ids[:num_concepts_per_client])
        
        drift_types_list = ['sudden', 'gradual', 'recurring']
        for client_id in range(client_count):
            client_drift_types[client_id] = np.random.choice(drift_types_list)
        
        drift_iterations = np.linspace(1, iterations - max(10, int(iterations*0.1)), num_drifts, dtype=int) # Ensure drifts don't happen too late
        
        for client_id in range(client_count):
            available_concepts_for_client = client_concepts[client_id]
            trajectory = np.zeros(iterations, dtype=int)
            
            if not available_concepts_for_client:
                client_concept_trajectories[client_id] = trajectory.tolist()
                continue

            initial_concept_idx_in_client_list = 0 
            trajectory[0] = available_concepts_for_client[initial_concept_idx_in_client_list]
            
            drift_type = client_drift_types[client_id]
            current_concept_list_idx = initial_concept_idx_in_client_list

            if drift_type == 'sudden':
                for i in range(1, iterations):
                    if i in drift_iterations:
                        current_concept_list_idx = (current_concept_list_idx + 1) % len(available_concepts_for_client)
                    trajectory[i] = available_concepts_for_client[current_concept_list_idx]
            elif drift_type == 'gradual':
                gradient_window = 10 
                for i in range(1, iterations):
                    target_concept_list_idx = current_concept_list_idx
                    if i in drift_iterations:
                        target_concept_list_idx = (current_concept_list_idx + 1) % len(available_concepts_for_client)
                    
                    active_concept_list_idx = target_concept_list_idx # Default to target
                    
                    is_in_gradient_window = False
                    nearest_drift_for_gradual = -1
                    for drift_iter_scan in drift_iterations:
                        if abs(i - drift_iter_scan) < gradient_window:
                            is_in_gradient_window = True
                            nearest_drift_for_gradual = drift_iter_scan
                            break
                    
                    if is_in_gradient_window:
                        distance = abs(i - nearest_drift_for_gradual)
                        # Determine the concept index *before* the target of the nearest drift
                        # This requires knowing which drift point nearest_drift_for_gradual is
                        num_drifts_passed = sum(1 for di in drift_iterations if di <= nearest_drift_for_gradual and di <= i)
                        
                        concept_idx_before_drift_target = (initial_concept_idx_in_client_list + num_drifts_passed -1) % len(available_concepts_for_client)
                        concept_idx_at_drift_target = (initial_concept_idx_in_client_list + num_drifts_passed) % len(available_concepts_for_client)


                        if i < nearest_drift_for_gradual: # Approaching drift
                            prob_new_concept = (gradient_window - distance) / gradient_window 
                            if np.random.random() < prob_new_concept: # Higher chance of new as it gets closer
                                active_concept_list_idx = concept_idx_at_drift_target
                            else:
                                active_concept_list_idx = concept_idx_before_drift_target
                        else: # Moving away from drift
                            prob_new_concept = distance / gradient_window # Higher chance of new as it gets further
                            if np.random.random() < prob_new_concept:
                                active_concept_list_idx = concept_idx_at_drift_target
                            else:
                                active_concept_list_idx = concept_idx_before_drift_target
                    else: # Not in any gradient window
                         # Update current_concept_list_idx only when passing a drift point
                        if i in drift_iterations:
                             current_concept_list_idx = (current_concept_list_idx + 1) % len(available_concepts_for_client)
                        active_concept_list_idx = current_concept_list_idx

                    trajectory[i] = available_concepts_for_client[active_concept_list_idx]
            else: # recurring
                period_length = np.random.randint(max(15, int(iterations/10)), max(30, int(iterations/5)))
                for i in range(1, iterations):
                    if i in drift_iterations:
                         period_length = np.random.randint(max(15, int(iterations/10)), max(30, int(iterations/5)))
                    current_concept_list_idx_for_recurring = (initial_concept_idx_in_client_list + (i // period_length)) % len(available_concepts_for_client)
                    trajectory[i] = available_concepts_for_client[current_concept_list_idx_for_recurring]
            client_concept_trajectories[client_id] = trajectory.tolist()

        concept_info_to_save = {
            'num_concepts': num_concepts, 'client_count': client_count, 'class_count': class_count,
            'iterations': iterations, 'num_drifts': num_drifts,
            'concept_definitions': concept_definitions, 'client_concepts': client_concepts,
            'client_drift_types': client_drift_types, 'drift_iterations': drift_iterations.tolist(),
            'client_concept_trajectories': client_concept_trajectories
        }
        try:
            with open(concept_config_path, 'w', encoding='utf-8') as f:
                json.dump(concept_info_to_save, f, indent=4, cls=NumpyArrayEncoder)
            print(f"Generated and saved new concept configuration to {concept_config_path}")
        except Exception as e:
            print(f"Error saving new concept configuration to {concept_config_path}: {e}")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset_full = torchvision.datasets.CIFAR100(
        root=os.path.join(data_dir, "rawdata"), train=True, download=True, transform=transform)
    
    trainloader_full = torch.utils.data.DataLoader(
        trainset_full, batch_size=len(trainset_full.data), shuffle=False)

    train_images_all, train_labels_all = None, None
    for _, data_loaded in enumerate(trainloader_full, 0):
        train_images_all, train_labels_all = data_loaded
        break 
    
    train_images_all_np = train_images_all.cpu().detach().numpy()
    train_labels_all_np = train_labels_all.cpu().detach().numpy()

    X_base_clients, y_base_clients, statistic_base = separate_data(
        (train_images_all_np, train_labels_all_np), 
        client_count, class_count, is_niid, is_balanced, partition_type, class_per_client=30 # class_per_client for initial distribution
    )
    
    print(f"Effective configuration: {num_concepts} concepts, {client_count} clients, {class_count} classes.")
    print(f"Drift iterations: {drift_iterations.tolist()}")
    
    for iteration in range(iterations):
        print(f"\\nGenerating data for iteration {iteration}")
        iteration_dir = os.path.join(data_dir, f"iteration_{iteration}")
        if not os.path.exists(iteration_dir): os.makedirs(iteration_dir)
        
        iteration_train_path = os.path.join(iteration_dir, "train/")
        iteration_test_path = os.path.join(iteration_dir, "test/")
        iteration_config_path = os.path.join(iteration_dir, "config.json")
        if not os.path.exists(iteration_train_path): os.makedirs(iteration_train_path)
        if not os.path.exists(iteration_test_path): os.makedirs(iteration_test_path)

        y_iter_clients_drifted = {}
        for client_id in range(client_count):
            current_concept_id = client_concept_trajectories[client_id][iteration]
            swaps_to_apply = concept_definitions[str(current_concept_id)] # concept_definitions keys are strings
            
            client_original_labels = y_base_clients[client_id].copy()
            y_iter_clients_drifted[client_id] = apply_label_swaps(client_original_labels, swaps_to_apply)

        # X_base_clients images do not change due to label swapping for this iteration
        # Use X_base_clients and y_iter_clients_drifted with split_data
        train_data_iter, test_data_iter = split_data(X_base_clients, y_iter_clients_drifted)
        
        # Recompute statistic for the training portion of this iteration's data
        statistic_iter = []
        for c_idx in range(client_count):
            if c_idx in train_data_iter and 'y' in train_data_iter[c_idx]:
                 labels_for_stat = train_data_iter[c_idx]['y']
                 stat_dict = {cls_label: np.sum(labels_for_stat == cls_label) for cls_label in range(class_count)}
                 statistic_iter.append(stat_dict)
            else: # Handle if client has no training data this iteration
                 statistic_iter.append({cls_label: 0 for cls_label in range(class_count)})
        
        save_file(
            iteration_config_path, iteration_train_path, iteration_test_path,
            train_data_iter, test_data_iter, client_count, class_count,
            statistic_iter, is_niid, is_balanced, partition_type
        )
        # print(f"Iteration {iteration} data generation complete.")
    
    print("\\nAll iteration data generation complete.")
    final_client_concepts_map = {}
    final_iteration_idx = iterations - 1
    for client_id in range(client_count):
        if client_id in client_concept_trajectories and final_iteration_idx < len(client_concept_trajectories[client_id]):
            final_client_concepts_map[client_id] = client_concept_trajectories[client_id][final_iteration_idx]
        else:
            final_client_concepts_map[client_id] = None 
    
    final_client_concepts_path = os.path.join(drift_info_path, "final_client_concepts.json")
    with open(final_client_concepts_path, 'w', encoding='utf-8') as f:
        json.dump(final_client_concepts_map, f, indent=4)

    effective_concept_info = {
        'num_concepts': num_concepts, 'client_count': client_count, 'class_count': class_count,
        'iterations': iterations, 'num_drifts': num_drifts,
        'concept_definitions': concept_definitions, 'client_concepts': client_concepts,
        'client_drift_types': client_drift_types, 'drift_iterations': drift_iterations.tolist(),
        'client_concept_trajectories': client_concept_trajectories
    }
    return effective_concept_info

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super(NumpyArrayEncoder, self).default(o)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        niid = True if sys.argv[1] == "noniid" else False
        
    else:
        niid = False

    if len(sys.argv) > 2:
        balance = False if sys.argv[2] == "balance" else False
    else:
        balance = False

    partition = "dir"


    generate_cifar100(
                data_dir="Cifar100_clustered_1",  # 数据存储路径
                client_count=20,                 # 客户端数量
                class_count=100,                 # 类别数量
                is_niid=True,                    # 非独立同分布设置
                is_balanced=False,               # 非平衡分配
                partition_type="dir")          
    #         )
    #     else:
    #         cluster_info = generate_cifar100_with_clusters(
    #             data_dir="Cifar100_clustered/",  # 数据存储路径
    #             client_count=20,                 # 客户端数量
    #             class_count=100,                 # 类别数量
    #             is_niid=True,                    # 非独立同分布设置
    #             is_balanced=False,               # 非平衡分配
    #             partition_type="dir",            # 分区方式
    #             num_concepts=5,                  # 概念数量
    #             iterations=200,                  # 时间迭代数量
    #             num_drifts=5                     # 漂移次数
    #         )
    # else:
    #     print("need dataset arg")