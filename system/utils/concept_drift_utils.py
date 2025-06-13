# import numpy as np # Not strictly needed for the revised drift_dataset logic if not used elsewhere
import torch  # Import torch
import random
import numpy as np
import json


def drift_dataset(dataset, class_a, class_b):
    """
    Perform concept drift by swapping two labels class_a and class_b.
    The input 'dataset' is expected to be a list of tuples, where each tuple
    is (features, label). The labels are modified in-place by reconstructing
    the tuples. Labels will be stored as torch.Tensor.
    :param dataset: List of (features, label) tuples. Modified in-place.
    :param class_a: One of the labels to be swapped.
    :param class_b: The other label to be swapped.
    :return: None.
    """
    if not dataset:
        # print("Warning: drift_dataset called with an empty or None dataset.")
        return
    if class_a == class_b:
        # print("Warning: class_a and class_b are the same in drift_dataset. No swap performed.")
        return

    # Step 1: Identify original indices for class_a and class_b based on current labels in dataset.
    indices_of_a = []
    indices_of_b = []
    try:
        for i, item in enumerate(dataset):
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                # print(f"Warning: Item at index {i} in dataset is not in (feature, label) format. Skipping.")
                continue

            label = item[1]  # Access the label from the tuple (features, label)
            # Ensure label is a Python int for comparison, in case it's already a tensor
            if isinstance(label, torch.Tensor):
                label_val = label.item()
            else:
                label_val = label

            if label_val == class_a:
                indices_of_a.append(i)
            elif label_val == class_b:
                indices_of_b.append(i)
    except TypeError:
        # print("Error: Dataset is not iterable or items are not subscriptable in drift_dataset.")
        return  # Cannot proceed

    # Step 2: Perform the swaps using the identified indices.
    # Since tuples are immutable, we create new tuples for modification.

    # Change original A to class_b
    for idx in indices_of_a:
        try:
            feature_at_idx = dataset[idx][0]
            # Store the new label as a torch.Tensor
            dataset[idx] = (feature_at_idx, torch.tensor(class_b, dtype=torch.int64))
        except IndexError:
            # print(f"Warning: Could not access feature at index {idx} during A->B swap.")
            pass  # Or handle more robustly

    # Change original B to class_a
    for idx in indices_of_b:
        try:
            feature_at_idx = dataset[idx][0]
            # Store the new label as a torch.Tensor
            dataset[idx] = (feature_at_idx, torch.tensor(class_a, dtype=torch.int64))
        except IndexError:
            # print(f"Warning: Could not access feature at index {idx} during B->A swap.")
            pass  # Or handle more robustly


def sudden_drift(clients, global_test_sets, _round):
    print(f"Sudden drift occurs at {_round}")

    drift_dataset(global_test_sets[1], 1, 2)
    drift_dataset(global_test_sets[2], 3, 4)
    drift_dataset(global_test_sets[3], 5, 6)

    for client in clients:
        if client.id % 10 < 3:
            drift_dataset(client.train_set, 1, 2)
            drift_dataset(client.test_set, 1, 2)
            client.global_test_id = 1
        elif client.id % 10 < 6:
            drift_dataset(client.train_set, 3, 4)
            drift_dataset(client.test_set, 3, 4)
            client.global_test_id = 2
        else:
            drift_dataset(client.train_set, 5, 6)
            drift_dataset(client.test_set, 5, 6)
            client.global_test_id = 3


def incremental_drift(clients, global_test_sets, _round):
    print(f"Incremental drift occurs at {_round}")

    if _round == 100:
        drift_dataset(global_test_sets[1], 1, 2)
    elif _round == 110:
        drift_dataset(global_test_sets[2], 3, 4)
    elif _round == 120:
        drift_dataset(global_test_sets[3], 5, 6)

    for client in clients:
        if _round == 100 and client.id % 10 < 3:
            drift_dataset(client.train_set, 1, 2)
            drift_dataset(client.test_set, 1, 2)
            client.global_test_id = 1
        elif _round == 110 and 3 <= client.id % 10 < 6:
            drift_dataset(client.train_set, 3, 4)
            drift_dataset(client.test_set, 3, 4)
            client.global_test_id = 2
        elif _round == 120 and client.id % 10 >= 6:
            drift_dataset(client.train_set, 5, 6)
            drift_dataset(client.test_set, 5, 6)
            client.global_test_id = 3


# Added functions for complex concept drift scenarios
def load_superclass_maps(map_path):
    """Loads superclass mapping from a JSON file."""
    try:
        with open(map_path, 'r') as f:
            maps = json.load(f)

        # Ensure keys are integers if they represent labels
        fine_to_coarse = {int(k): v for k, v in maps.get("fine_to_coarse", {}).items()}

        coarse_to_fine = {}
        if fine_to_coarse:
            for fine, coarse in fine_to_coarse.items():
                if coarse not in coarse_to_fine:
                    coarse_to_fine[coarse] = []
                coarse_to_fine[coarse].append(fine)

        all_fine_labels = sorted(list(fine_to_coarse.keys()))
        # Ensure coarse_labels_list contains the actual superclass labels, not just indices if they differ
        all_coarse_labels = maps.get("coarse_labels_list", sorted(list(set(fine_to_coarse.values()))))

        return fine_to_coarse, coarse_to_fine, all_fine_labels, all_coarse_labels
    except FileNotFoundError:
        print(f"Error: Superclass map file not found at {map_path}")
        return {}, {}, [], []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from superclass map file: {map_path}")
        return {}, {}, [], []
    except Exception as e:
        print(f"Error loading superclass map from {map_path}: {e}")
        return {}, {}, [], []


def get_new_label_intra_superclass(old_fine_label, fine_to_coarse_map, coarse_to_fine_map):
    """Gets a new fine label within the same superclass, different from the old one."""
    if old_fine_label not in fine_to_coarse_map:
        # print(f"Warning: old_fine_label {old_fine_label} not in fine_to_coarse_map.")
        return old_fine_label

    coarse_label = fine_to_coarse_map[old_fine_label]
    if coarse_label not in coarse_to_fine_map:
        # print(f"Warning: coarse_label {coarse_label} not in coarse_to_fine_map.")
        return old_fine_label

    possible_new_fine_labels = [l for l in coarse_to_fine_map[coarse_label] if l != old_fine_label]

    if not possible_new_fine_labels:
        # print(f"Warning: No other fine labels in superclass {coarse_label} for old_fine_label {old_fine_label}.")
        return old_fine_label
    return random.choice(possible_new_fine_labels)


def get_new_label_inter_superclass(old_fine_label, fine_to_coarse_map, coarse_to_fine_map, all_coarse_labels, all_fine_labels):
    """Gets a new fine label from a different superclass."""
    if not fine_to_coarse_map or not coarse_to_fine_map or not all_coarse_labels or not all_fine_labels:
        # print("Warning: Superclass map information is incomplete for inter-superclass drift.")
        return old_fine_label

    if old_fine_label not in fine_to_coarse_map:
        # print(f"Warning: old_fine_label {old_fine_label} not in fine_to_coarse_map for inter-superclass drift.")
        possible_new_labels = [l for l in all_fine_labels if l != old_fine_label]
        return random.choice(possible_new_labels) if possible_new_labels else old_fine_label

    current_coarse_label = fine_to_coarse_map[old_fine_label]
    other_coarse_labels = [cl for cl in all_coarse_labels if cl != current_coarse_label]

    if not other_coarse_labels:
        # print(f"Warning: No other superclasses available to drift from {current_coarse_label}.")
        return old_fine_label

    chosen_new_coarse_label = random.choice(other_coarse_labels)

    if chosen_new_coarse_label not in coarse_to_fine_map or not coarse_to_fine_map[chosen_new_coarse_label]:
        # print(f"Warning: Chosen new superclass {chosen_new_coarse_label} is empty or not in map.")
        possible_new_labels = [l for l in all_fine_labels if l != old_fine_label and fine_to_coarse_map.get(l) != current_coarse_label]
        return random.choice(possible_new_labels) if possible_new_labels else old_fine_label

    return random.choice(coarse_to_fine_map[chosen_new_coarse_label])


def apply_complex_drift(dataset_targets, client_id, current_epoch, drift_config, superclass_maps):
    """
    Applies complex concept drift to dataset_targets (a list or numpy array of labels).
    Modifies dataset_targets in place.
    Returns True if drift was applied, False otherwise.
    """
    if not drift_config or "scenario_type" not in drift_config:
        # print("Debug: No drift_config or scenario_type provided.")
        return False

    scenario = drift_config["scenario_type"]
    base_epoch = drift_config.get("base_epoch", 100)
    fine_to_coarse, coarse_to_fine, all_fine, all_coarse = superclass_maps

    if not fine_to_coarse or not coarse_to_fine or not all_fine or not all_coarse:
        print(f"Client {client_id}: Warning: Superclass maps are incomplete. Cannot apply complex drift for scenario {scenario} at epoch {current_epoch}.")
        return False

    num_samples = len(dataset_targets)
    if num_samples == 0:
        # print(f"Client {client_id}: Debug: dataset_targets is empty for scenario {scenario} at epoch {current_epoch}.")
        return False

    indices_to_drift = []
    apply_drift_now = False

    if scenario.startswith("staggered_"):
        stagger_interval = drift_config.get("stagger_interval", 10)
        drift_epoch_for_client = base_epoch + client_id * stagger_interval
        if current_epoch == drift_epoch_for_client:
            apply_drift_now = True
            indices_to_drift = list(range(num_samples))
    elif scenario.startswith("partial_"):
        drift_epoch_for_client = base_epoch
        if current_epoch == drift_epoch_for_client:
            apply_drift_now = True
            partial_percentage = drift_config.get("partial_percentage", 0.1)
            num_to_drift = int(num_samples * partial_percentage)
            if num_to_drift > 0:
                indices_to_drift = random.sample(range(num_samples), num_to_drift)
            else:
                # print(f"Client {client_id}: Debug: num_to_drift is 0 for partial drift at epoch {current_epoch}.")
                pass  # No samples to drift
    elif scenario.startswith("global_"):
        drift_epoch_for_client = base_epoch
        if current_epoch == drift_epoch_for_client:
            apply_drift_now = True
            indices_to_drift = list(range(num_samples))
    else:
        print(f"Client {client_id}: Warning: Unknown drift scenario type: {scenario} at epoch {current_epoch}.")
        return False

    if not apply_drift_now or not indices_to_drift:
        # print(f"Client {client_id}: Debug: Drift not applied for scenario {scenario} at epoch {current_epoch}. Apply_now: {apply_drift_now}, Indices_empty: {not indices_to_drift}")
        return False

    drift_applied_count = 0
    for i in indices_to_drift:
        old_label = dataset_targets[i]
        new_label = old_label

        if "intra_superclass" in scenario:
            new_label = get_new_label_intra_superclass(old_label, fine_to_coarse, coarse_to_fine)
        elif "inter_superclass" in scenario:
            new_label = get_new_label_inter_superclass(old_label, fine_to_coarse, coarse_to_fine, all_coarse, all_fine)
        else:
            # print(f"Client {client_id}: Warning: Scenario {scenario} does not specify intra/inter superclass drift type.")
            continue  # Skip this sample if drift type unclear

        if new_label != old_label:
            dataset_targets[i] = new_label
            drift_applied_count += 1

    if drift_applied_count > 0:
        print(f"Client {client_id}: Applied {scenario} drift at epoch {current_epoch}. {drift_applied_count}/{len(indices_to_drift)} labels changed.")
        return True
    # else:
    # print(f"Client {client_id}: Debug: No labels were actually changed for scenario {scenario} at epoch {current_epoch} despite conditions met.")

    return False

# Example of how you might structure the superclass map JSON file (e.g., cifar100_superclass_map.json)
# {
#   "fine_to_coarse": {
#     "0": 0, "1": 0, "2": 1, "3": 1, "4": 2, ...
#   },
#   "coarse_labels_list": [0, 1, 2, ..., 19]
#   "coarse_label_names": ["aquatic_mammals", "fish", ..., "vehicles_2"],
#   "fine_label_names": ["beaver", "dolphin", ..., "wolf"]
# }
# Note: coarse_label_names and fine_label_names are for reference and not strictly used by these functions.
# Ensure fine_to_coarse keys are strings if loading directly from json, then convert to int.
# Ensure coarse_labels_list contains the actual superclass label values used in fine_to_coarse.

