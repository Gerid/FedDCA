import random
import json
import os
import sys # Add sys module
import numpy as np
from torchvision import datasets, transforms # Keep for potential fallback or other dataset types
from torch.utils.data import TensorDataset, DataLoader
import torch

# Add project root to sys.path to allow absolute import from system package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from system.utils.data_utils import read_client_data

# Constants
# NUM_CLIENTS = 10 # Defined in main or passed as arg
# TOTAL_ITERATIONS = 100 # Defined in main or passed as arg
# PREGEN_DIR_NAME = "CIFAR10_pregenerated_drift" # Defined in main or passed as arg
# CONFIG_FILE_NAME = "drift_patterns_config.json" # Defined in main or passed as arg

# Placeholder for actual dataset utilities
# from ..utils.dataset_utils import read_client_data 

def get_original_client_data(client_id, dataset_name, data_path, is_train=True):
    """
    Loads pre-partitioned data for a specific client using read_client_data.
    data_path is the root directory for datasets (e.g., '../dataset').
    dataset_name is the name of the dataset (e.g., 'CIFAR100').
    read_client_data is expected to find data like 'data_path/dataset_name/train/client_id.json'.
    """
    # print(f"Loading original {'train' if is_train else 'test'} data for client {client_id} from dataset {dataset_name} in {data_path}")
    
    try:
        # Construct the actual path to the dataset directory, e.g., "../dataset/CIFAR100"
        # read_client_data might expect the dataset name and internally construct paths relative to a known root,
        # or it might expect the direct path to the dataset folder.
        # Assuming read_client_data(dataset_name, client_id, is_train) works like in clientbase.py,
        # where dataset_name is "CIFAR100". This suggests read_client_data might use dataset_name
        # to find files within a pre-configured structure relative to data_path.
        # Let's assume read_client_data is called with dataset_name and it handles path resolution,
        # possibly using data_path if it's made available to it (e.g. via a global config or if it's in sys.path).
        # For this script, data_path is passed from main_pregenerate.
        # We will assume read_client_data needs the base path where dataset_name subdirectories are located.
        
        # The files are expected to be in data_path/dataset_name/train/ or data_path/dataset_name/test/
        # e.g. ../dataset/CIFAR100/train/0.json
        # So, read_client_data should be called with dataset_name, and it should know to look inside data_path.
        # Let's adjust read_client_data if necessary, or assume it works this way:
        # read_client_data(dataset_name, client_id, is_train, base_dir=data_path)

        # Sticking to the signature from clientbase: read_client_data(dataset_name, client_id, is_train)
        # This implies that `dataset_utils.py` must know where `data_path/dataset_name` is.
        # This is often true if the project has a standard data directory structure.
        client_data_dict = read_client_data(dataset_name, client_id, is_train=is_train)

        if client_data_dict is None or 'x' not in client_data_dict or 'y' not in client_data_dict:
            print(f"Warning: No data or incomplete data returned by read_client_data for client {client_id}, dataset {dataset_name}, {'train' if is_train else 'test'}.")
            return TensorDataset(torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.long)) # Return empty TensorDataset

        features_np = np.array(client_data_dict['x'])
        labels_np = np.array(client_data_dict['y'])

        if features_np.size == 0 or labels_np.size == 0:
             # print(f"Warning: Client {client_id} has empty features or labels for {dataset_name} {'train' if is_train else 'test'}.")
             # Return empty tensor dataset, pregenerate_data_for_client will skip if original_data_tensor_for_split is empty.
             return TensorDataset(torch.tensor(features_np, dtype=torch.float32), torch.tensor(labels_np, dtype=torch.long))


        features_tensor = torch.tensor(features_np, dtype=torch.float32)
        labels_tensor = torch.tensor(labels_np, dtype=torch.long)
        
        return TensorDataset(features_tensor, labels_tensor)
    except FileNotFoundError:
        print(f"Warning: Data file not found for client {client_id}, dataset {dataset_name}, {'train' if is_train else 'test'} using base_data_dir='{data_path}'.")
        return TensorDataset(torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.long)) # Return empty on file not found
    except Exception as e:
        print(f"Error loading data for client {client_id}, dataset {dataset_name}, {'train' if is_train else 'test'}: {e}")
        # Depending on severity, either raise e or return empty
        return TensorDataset(torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.long))

def apply_concept_distribution_pregen(data_tensor_dataset, concept):
    """
    Applies label mapping from the concept to the data_tensor_dataset.
    data_tensor_dataset is a TensorDataset.
    concept contains 'label_mapping_func'.
    """
    if not isinstance(data_tensor_dataset, TensorDataset):
        print("Warning: apply_concept_distribution_pregen expects a TensorDataset.")
        return data_tensor_dataset

    original_features, original_labels = data_tensor_dataset.tensors
    
    if original_labels is None or len(original_labels) == 0:
        # print("Warning: No labels to map.")
        return data_tensor_dataset

    # The label_mapping_func in the concept is already a lambda y_data: base_concept['label_mapping_func'](y_data, params)
    # So it expects only the label data (e.g., a numpy array or torch tensor of labels)
    mapped_labels_np = concept['label_mapping_func'](original_labels.numpy())
    
    return TensorDataset(original_features, torch.from_numpy(mapped_labels_np).long())

def get_predefined_shared_concepts(num_classes):
    shared_concepts = [
        {
            "name": "identity",
            "label_mapping_func": lambda y, params: y, # y is numpy array
            "class_weights_func": lambda nc, params: np.ones(nc) / nc if nc > 0 else np.array([])
        },
        {
            "name": "swap_labels",
            "label_mapping_func": lambda y, params: np.array([params.get('class_b', 0) if lbl == params.get('class_a', 0) else params.get('class_a', 0) if lbl == params.get('class_b', 0) else lbl for lbl in y]),
            "class_weights_func": lambda nc, params: np.ones(nc) / nc if nc > 0 else np.array([])
        },
        {
            "name": "bias_one_class_positive",
            "label_mapping_func": lambda y, params: y,
            "class_weights_func": lambda nc, params: np.array([params.get('factor', 0.9) if i == params.get('bias_class', 0) else (1.0-params.get('factor', 0.9))/(nc-1 if nc > 1 else 1) for i in range(nc)]) if nc > 0 else np.array([])
        },
        {
            "name": "bias_one_class_negative", # factor should be small
            "label_mapping_func": lambda y, params: y,
            "class_weights_func": lambda nc, params: np.array([params.get('factor', 0.1) if i == params.get('bias_class', 0) else (1.0-params.get('factor', 0.1))/(nc-1 if nc > 1 else 1) for i in range(nc)]) if nc > 0 else np.array([])
        },
        {
            "name": "rotate_labels", # Rotates labels by a shift
            "label_mapping_func": lambda y, params: np.array([(lbl + params.get('shift', 1)) % num_classes for lbl in y]) if num_classes > 0 else y,
            "class_weights_func": lambda nc, params: np.ones(nc) / nc if nc > 0 else np.array([])
        },
        {
            "name": "subset_classes", # Keeps only a subset of classes
            "label_mapping_func": lambda y, params: np.array([lbl if lbl in params.get('keep_classes', list(range(num_classes))) else params.get('remap_to', lbl) for lbl in y]),
            "class_weights_func": lambda nc, params: np.array([1.0/len(params.get('keep_classes', [1])) if i in params.get('keep_classes', list(range(nc))) and len(params.get('keep_classes', [1])) > 0 else 0 for i in range(nc)]) if nc > 0 else np.array([])
        }
    ]
    return shared_concepts

def generate_and_save_drift_config(config_path, num_clients, total_iterations, shared_concepts, num_classes_for_config):
    drift_patterns = {}
    if not num_classes_for_config or num_classes_for_config <= 0:
        print("Warning: num_classes_for_config is not valid for config generation. Defaulting to 10.")
        num_classes_for_config = 10 # Use this local variable for safety

    for i in range(num_clients):
        client_id = f"client_{i}"
        drift_type = random.choice(["sudden", "gradual"]) 
        drift_point = random.randint(total_iterations // 4, 3 * total_iterations // 4) if total_iterations > 1 else 0
        
        stages = []
        stages.append({
            "until_iteration": drift_point -1 if drift_point > 0 else 0,
            "concept_id": 0, 
            "params": {}
        })

        post_drift_concept_id = 0 # Default to identity
        if shared_concepts and len(shared_concepts) > 1: # Ensure there are concepts other than identity
            candidate_concepts = [idx for idx in range(len(shared_concepts)) if idx != 0]
            if candidate_concepts:
                 post_drift_concept_id = random.choice(candidate_concepts)

        params = {}
        chosen_concept_name = shared_concepts[post_drift_concept_id]['name'] if post_drift_concept_id < len(shared_concepts) else "identity"

        if chosen_concept_name == "swap_labels":
            if num_classes_for_config >= 2:
                params["class_a"], params["class_b"] = random.sample(range(num_classes_for_config), 2)
            else: 
                post_drift_concept_id = 0; params = {} 
        elif "bias_one_class" in chosen_concept_name:
            if num_classes_for_config >= 1:
                params["bias_class"] = random.randint(0, num_classes_for_config - 1)
                params["factor"] = random.uniform(0.6, 0.9) if "positive" in chosen_concept_name else random.uniform(0.05, 0.2)
            else:
                post_drift_concept_id = 0; params = {}
        elif chosen_concept_name == "rotate_labels":
            if num_classes_for_config >= 1:
                params["shift"] = random.randint(1, max(1, num_classes_for_config - 1))
            else:
                post_drift_concept_id = 0; params = {}
        elif chosen_concept_name == "subset_classes":
            if num_classes_for_config >= 1:
                num_to_keep = random.randint(1, num_classes_for_config)
                params["keep_classes"] = sorted(random.sample(range(num_classes_for_config), num_to_keep))
                if params["keep_classes"]: # Ensure not empty
                     params["remap_to"] = params["keep_classes"][0] 
                else: # Should not happen if num_to_keep >=1
                     params["remap_to"] = 0 
            else: 
                post_drift_concept_id = 0; params = {}
        
        stages.append({
            "until_iteration": total_iterations, 
            "concept_id": post_drift_concept_id,
            "params": params
        })
        
        drift_patterns[client_id] = {
            "drift_type": drift_type,
            "drift_point": drift_point, 
            "stages": stages
        }

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(drift_patterns, f, indent=4)
    print(f"Drift configuration saved to {config_path}")
    return drift_patterns

def get_concept_for_iteration_pregen(client_drift_config, iteration, shared_concepts, num_classes_dataset):
    active_stage = None
    for stage in client_drift_config.get("stages", []):
        if iteration <= stage["until_iteration"]:
            active_stage = stage
            break
    
    if not active_stage: 
        concept_id = 0
        params = {}
    else:
        concept_id = active_stage["concept_id"]
        params = active_stage["params"]

    if not (0 <= concept_id < len(shared_concepts)):
        print(f"Warning: Invalid concept_id {concept_id} for client. Defaulting to identity.")
        base_concept = shared_concepts[0] 
        params = {} # Reset params for safety if concept_id was invalid
    else:
        base_concept = shared_concepts[concept_id]

    # `num_classes_dataset` is the actual number of classes in the data being processed.
    # The lambda functions in shared_concepts capture `num_classes` from their definition scope (via `get_predefined_shared_concepts`).
    # If `num_classes_dataset` differs from the `num_classes` used to define concepts (e.g. rotate_labels),
    # the behavior of rotate_labels might be unexpected.
    # For now, we assume they are consistent or that concepts are robust.
    
    final_concept = {
        'name': base_concept['name'],
        'params': params, 
        'label_mapping_func': lambda y_data: base_concept['label_mapping_func'](y_data, params),
        'class_weights_func': lambda nc: base_concept['class_weights_func'](nc, params)
    }
    return final_concept

def pregenerate_data_for_client(client_id, client_original_data_train, client_original_data_test,
                                client_drift_config, shared_concepts, total_iterations,
                                save_dir_path_base, num_classes_dataset):
    print(f"Pregenerating data for client {client_id} using {num_classes_dataset} classes.")
    
    for data_split_name, original_data_tensor_for_split in [("train", client_original_data_train), ("test", client_original_data_test)]:
        if original_data_tensor_for_split is None:
            print(f"Warning: Original {data_split_name} data for client {client_id} is None. Skipping.")
            continue
        
        if not isinstance(original_data_tensor_for_split, TensorDataset) or len(original_data_tensor_for_split.tensors) < 2:
            print(f"Warning: Original {data_split_name} data for client {client_id} is not a valid TensorDataset. Skipping.")
            continue

        original_features_tensor, _ = original_data_tensor_for_split.tensors
        original_features_np = original_features_tensor.numpy()

        print(f"  Processing {data_split_name} split...")
        
        client_split_save_dir = os.path.join(save_dir_path_base, client_id, data_split_name)
        os.makedirs(client_split_save_dir, exist_ok=True)

        # Save features once
        features_save_path = os.path.join(client_split_save_dir, "features.npy")
        try:
            np.save(features_save_path, original_features_np)
            print(f"    Saved features for {data_split_name} split to {features_save_path}")
        except Exception as e:
            print(f"    Error saving features for {data_split_name} split to {features_save_path}: {e}")
            continue # If features can't be saved, don't proceed with labels for this split

        for iter_num in range(total_iterations):
            concept = get_concept_for_iteration_pregen(client_drift_config, iter_num, shared_concepts, num_classes_dataset)
            
            # Apply concept (currently only label mapping)
            # apply_concept_distribution_pregen works on the original data for each iteration
            drifted_data_tensor = apply_concept_distribution_pregen(original_data_tensor_for_split, concept)
            
            if drifted_data_tensor is None or not isinstance(drifted_data_tensor, TensorDataset):
                 print(f"    Iteration {iter_num}: Failed to apply concept or received invalid data. Skipping save.")
                 continue

            _, current_labels_tensor = drifted_data_tensor.tensors
            current_labels_np = current_labels_tensor.numpy()
            
            labels_save_path = os.path.join(client_split_save_dir, f"labels_iter_{iter_num}.npy")
            try:
                np.save(labels_save_path, current_labels_np)
                # if iter_num % (total_iterations // 10 if total_iterations >= 10 else 1) == 0 : # Log progress
                #     print(f"    Iteration {iter_num}: Saved labels to {labels_save_path} with concept '{concept['name']}'")
            except Exception as e:
                print(f"    Iteration {iter_num}: Error saving labels to {labels_save_path}: {e}")
        print(f"  Finished pregenerating {data_split_name} data for client {client_id} across {total_iterations} iterations.")

def main_pregenerate(dataset_name="CIFAR10", data_path="../data", num_clients=10, total_iterations=100, 
                     pregen_dir_name_template="{dataset_name}_pregenerated_drift", 
                     config_file_name="drift_patterns_config.json",
                     force_regenerate_config=False):
    
    actual_pregen_dir_name = pregen_dir_name_template.format(dataset_name=dataset_name)
    
    temp_num_classes = 0
    # Try to determine NUM_CLASSES from client 0's data first
    # This now uses the new get_original_client_data
    # The data_path here is the one passed to main_pregenerate, e.g., "../dataset"
    print(f"Attempting to infer NUM_CLASSES for {dataset_name} from client 0 data using data_path: {data_path}...")
    try:
        # Load a sample from client 0's training data to infer num_classes
        # Pass the correct data_path which is the root for all datasets
        sample_data_client0_train = get_original_client_data(0, dataset_name, data_path, is_train=True)
        if sample_data_client0_train and len(sample_data_client0_train) > 0:
            _, sample_labels = sample_data_client0_train.tensors
            if len(sample_labels) > 0:
                unique_labels = torch.unique(sample_labels)
                temp_num_classes = len(unique_labels)
                # It's possible client 0 doesn't have all classes, so this might be an underestimation.
                # Consider checking max label value as well if classes are 0-indexed.
                max_label_val = torch.max(sample_labels)
                if temp_num_classes <= max_label_val: # If unique count is less than max label value + 1
                    temp_num_classes = int(max_label_val.item() + 1)

                print(f"Inferred NUM_CLASSES = {temp_num_classes} for {dataset_name} from client 0's training data sample.")
            else:
                print("Client 0's sample training labels are empty, cannot infer num_classes from it.")
        else:
            print("Could not load client 0's sample training data to infer num_classes.")
    except Exception as e:
        print(f"Error inferring num_classes from client 0's data for {dataset_name}: {e}")

    if temp_num_classes == 0: # If inference failed, fallback to hardcoded/default
        print(f"NUM_CLASSES inference failed for {dataset_name}. Falling back to predefined or default.")
        if dataset_name == "CIFAR10":
            temp_num_classes = 10
        elif dataset_name == "CIFAR100":
            temp_num_classes = 100
        elif dataset_name == "MNIST":
            temp_num_classes = 10
        # Add other known datasets here
        else:
            temp_num_classes = 10 # Fallback default for unknown datasets
            print(f"Warning: num_classes for dataset {dataset_name} is unknown and inference failed. Defaulting to {temp_num_classes}. This may be incorrect.")
            
    NUM_CLASSES = int(temp_num_classes)
    if NUM_CLASSES <=0:
        print("Error: Number of classes determined to be <=0. Aborting pregeneration.")
        return
        
    print(f"Using NUM_CLASSES = {NUM_CLASSES} for dataset {dataset_name}")

    shared_concepts = get_predefined_shared_concepts(NUM_CLASSES)

    # save_base_path is where the pregenerated data for this dataset_name will be stored.
    # e.g., data_path/CIFAR100_pregenerated_drift/
    save_base_path = os.path.join(data_path, actual_pregen_dir_name)
    os.makedirs(save_base_path, exist_ok=True)
    
    config_file_path = os.path.join(save_base_path, config_file_name)

    drift_patterns_config = None
    if force_regenerate_config or not os.path.exists(config_file_path):
        print(f"Generating new drift configuration file: {config_file_path}")
        drift_patterns_config = generate_and_save_drift_config(config_file_path, num_clients, total_iterations, shared_concepts, NUM_CLASSES)
    else:
        try:
            with open(config_file_path, 'r') as f:
                drift_patterns_config = json.load(f)
            print(f"Loaded existing drift configuration from {config_file_path}")
            if len(drift_patterns_config) != num_clients:
                print(f"Warning: Config file has {len(drift_patterns_config)} clients, but script is set for {num_clients}. Regenerating.")
                drift_patterns_config = generate_and_save_drift_config(config_file_path, num_clients, total_iterations, shared_concepts, NUM_CLASSES)
        except Exception as e:
            print(f"Error loading drift configuration: {e}. Regenerating.")
            drift_patterns_config = generate_and_save_drift_config(config_file_path, num_clients, total_iterations, shared_concepts, NUM_CLASSES)

    if not drift_patterns_config:
        print("Error: Failed to load or generate drift configuration. Exiting.")
        return

    for client_idx in range(num_clients):
        client_id_for_config = f"client_{client_idx}" # Config keys are "client_0", "client_1", etc.
        
        print(f"Processing data for client {client_idx} (config key: {client_id_for_config})...")
        
        try:
            # Load client-specific original train and test data
            # data_path is the root of datasets, e.g., ../dataset
            client_original_train_data_tensor = get_original_client_data(client_idx, dataset_name, data_path, is_train=True)
            client_original_test_data_tensor = get_original_client_data(client_idx, dataset_name, data_path, is_train=False)
        except Exception as e: # Catch any error during data loading for a specific client
            print(f"Critical error loading original data for client {client_idx} for dataset {dataset_name}. Skipping. Details: {e}")
            continue

        # Check if data loading was successful and data is not empty
        if not client_original_train_data_tensor or len(client_original_train_data_tensor) == 0:
            print(f"Warning: Original train data for client {client_idx} is missing or empty. Skipping pregeneration for this client.")
            continue
        # Test data can be optional for some pregeneration tasks, but if it's expected, check it too.
        if not client_original_test_data_tensor: # Allow empty test data if that's a valid scenario
             print(f"Warning: Original test data for client {client_idx} is missing. Proceeding with train data only for pregeneration if applicable.")
             # client_original_test_data_tensor will be None, pregenerate_data_for_client handles this.

        client_config = drift_patterns_config.get(client_id_for_config)
        if not client_config:
            print(f"Warning: No drift config found for {client_id_for_config}. Skipping pregeneration for this client.")
            continue

        pregenerate_data_for_client(
            client_id=client_id_for_config, # Pass the string client_id used in paths/configs
            client_original_data_train=client_original_train_data_tensor, 
            client_original_data_test=client_original_test_data_tensor,   
            client_drift_config=client_config,
            shared_concepts=shared_concepts,
            total_iterations=total_iterations,
            save_dir_path_base=save_base_path, # This is data_path/dataset_name_pregenerated_drift/
            num_classes_dataset=NUM_CLASSES 
        )
    
    print(f"Pregeneration process completed for {num_clients} clients and {total_iterations} iterations.")
    print(f"Generated data saved in: {save_base_path}")

if __name__ == '__main__':
    # Example usage:
    # Ensure the data_path points to the directory containing dataset folders like CIFAR100/, MNIST/
    # e.g., if CIFAR100 data is in 'D:/my_datasets/CIFAR100/', then data_path should be 'D:/my_datasets/'
    # The script expects partitioned data to exist in data_path/dataset_name/train/ and data_path/dataset_name/test/
    
    # Default data_path assumes 'dataset' folder is a sibling to the script's parent, or CWD is project root.
    # If running from 'PFL-Non-IID/dataset/', then '../dataset' is 'PFL-Non-IID/dataset/'.
    # If your data (like Cifar100/) is directly inside 'PFL-Non-IID/dataset/', then data_path should be "."
    # or an absolute path.
    # Given the __main__ example was: main_pregenerate(dataset_name="CIFAR100", data_path="../dataset", ...)
    # and assuming the script is in PFL-Non-IID/dataset/, then ../dataset refers to PFL-Non-IID/dataset/.
    # This means it expects PFL-Non-IID/dataset/CIFAR100/, PFL-Non-IID/dataset/MNIST/ etc.
    
    # Corrected example assuming 'CIFAR100' folder (containing train/test with client files)
    # is directly inside 'PFL-Non-IID/dataset/'
    # And the script is run from 'PFL-Non-IID/dataset/'
    # Then data_path should be "." to mean current directory, so it finds ./CIFAR100/
    # Or, if script is run from project root 'PFL-Non-IID/', then data_path="dataset"
    
    # Let's use the provided example from user, assuming it implies correct pathing for their setup.
    # data_path="../dataset" means the "dataset" folder one level up from where the script is, then go into "dataset" again.
    # If script is in PFL-Non-IID/dataset, then ../ is PFL-Non-IID. So ../dataset is PFL-Non-IID/dataset.
    # This path seems correct if the CIFAR100, MNIST folders are inside PFL-Non-IID/dataset/.

    main_pregenerate(
        dataset_name="CIFAR100", 
        data_path="../dataset",  # This should be the root containing CIFAR100/, MNIST/ etc.
        num_clients=20, 
        total_iterations=200, 
        force_regenerate_config=True
    )
    
    # main_pregenerate(dataset_name="MNIST", data_path="../dataset", num_clients=10, total_iterations=50, force_regenerate_config=True)
