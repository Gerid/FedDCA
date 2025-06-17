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
        # Ensure coarse_labels_list contains the actual superclass label values used in fine_to_coarse
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


def generate_k_fixed_label_mappings(num_concepts, num_classes, fine_to_coarse_map, coarse_to_fine_map, all_fine_labels, all_coarse_labels, drift_type, original_labels, drift_percentage=0.2):
    """
    Generates K fixed label mapping patterns by modifying a specific percentage of labels.

    Args:
        num_concepts (int): K, the number of fixed mapping patterns to generate.
        num_classes (int): Total number of fine-grain classes.
        fine_to_coarse_map (dict): Mapping from fine label to coarse label.
        coarse_to_fine_map (dict): Mapping from coarse label to list of fine labels.
        all_fine_labels (list): List of all fine-grain labels.
        all_coarse_labels (list): List of all coarse-grain labels.
        drift_type (str): "intra_superclass" or "inter_superclass".
        original_labels (list): List of original labels (e.g., range(num_classes)).
        drift_percentage (float): The proportion of labels to select for modification (0.0 to 1.0). Default is 1.0.

    Returns:
        list: A list of K dictionaries, where each dictionary is a label mapping {old_label: new_label}.
              Returns None if inputs are invalid or generation fails.
    """
    if not all(isinstance(m, dict) for m in [fine_to_coarse_map, coarse_to_fine_map]) or \
       not all(isinstance(l, list) for l in [all_fine_labels, all_coarse_labels, original_labels]):
        print("Error: Invalid map or label list types for generating fixed mappings.")
        return None
    if not (0.0 <= drift_percentage <= 1.0):
        print("Error: drift_percentage must be between 0.0 and 1.0.")
        return None

    k_mappings = []
    num_original_labels = len(original_labels)
    num_labels_to_drift = int(num_original_labels * drift_percentage)
    for _ in range(num_concepts):
        current_mapping = {}
        # Determine which labels to modify for this specific mapping concept
        if num_labels_to_drift > 0 and num_original_labels > 0:
            labels_to_modify_this_concept = random.sample(original_labels, num_labels_to_drift)
        else:
            labels_to_modify_this_concept = []
        labels_to_modify_set = set(labels_to_modify_this_concept)
        for old_label in original_labels:
            new_label = old_label
            if old_label in labels_to_modify_set: # Only attempt to drift selected labels
                if drift_type == "intra_superclass":
                    if fine_to_coarse_map and coarse_to_fine_map: # Ensure maps are valid
                        new_label = get_new_label_intra_superclass(old_label, fine_to_coarse_map, coarse_to_fine_map)
                    else:
                        # Fallback: random different label if superclass info missing for intra
                        possible_labels = [l for l in original_labels if l != old_label]
                        if possible_labels:
                            new_label = random.choice(possible_labels)
                elif drift_type == "inter_superclass":
                    if fine_to_coarse_map and coarse_to_fine_map and all_coarse_labels and all_fine_labels: # Ensure maps are valid
                        new_label = get_new_label_inter_superclass(old_label, fine_to_coarse_map, coarse_to_fine_map, all_coarse_labels, all_fine_labels)
                    else:
                        # Fallback: random different label if superclass info missing for inter
                        possible_labels = [l for l in original_labels if l != old_label]
                        if possible_labels:
                            new_label = random.choice(possible_labels)
                else:
                    print(f"Warning: Unknown drift_type \'{drift_type}\' for generating fixed mappings. Label {old_label} will not be changed.")
            current_mapping[old_label] = new_label
        k_mappings.append(current_mapping)
    
    if not k_mappings:
        print("Warning: No fixed mappings were generated.")
        return None
        
    return k_mappings


def apply_complex_drift(
    labels_list,
    client_id,
    current_round,
    drift_config,
    superclass_maps=None,
    fixed_concept_mappings=None,
    client_chosen_concept_index=None,
    drift_history=None
):
    """
    labels_list: list of int, will be modified in-place if drift occurs.
    client_id: str or int
    current_round: int
    drift_config: dict, must contain 'complex_drift_scenario' and related keys
    superclass_maps: optional, for superclass-based drift
    fixed_concept_mappings: optional, list of label map dicts for fixed/recurring concept switch
    client_chosen_concept_index: optional, int index into fixed_concept_mappings for fixed concept switch
    drift_history: optional, for recurring drift (dict: client_id -> [(start, end, concept_id), ...])
    Returns: (drift_applied: bool, current_concept_id: int)
    """
    scenario = drift_config.get("complex_drift_scenario", "")
    base_epoch = drift_config.get("drift_base_epoch", 100)
    
    current_concept_id = 0  # Default concept_id (e.g., initial state)
    drift_applied = False

    # 1. fixed_concept_switch
    if scenario == "fixed_concept_switch":
        # This scenario requires fixed_concept_mappings and client_chosen_concept_index
        if fixed_concept_mappings is None or client_chosen_concept_index is None:
            print(f"Warning (client {client_id}, round {current_round}): fixed_concept_switch scenario called without fixed_concept_mappings or client_chosen_concept_index.")
            return False, current_concept_id # Return default concept_id

        drift_epoch = drift_config.get("fixed_drift_epoch", base_epoch) # Round at which the switch occurs

        if current_round < drift_epoch:
            current_concept_id = 0 # Or a specific initial_concept_id if defined
        else:
            # After (or at) the drift epoch, the client is on its chosen concept
            current_concept_id = client_chosen_concept_index 
                                    # This index itself serves as the identifier for the concept.

        # Apply the label transformation only ONCE at the drift_epoch
        if current_round == drift_epoch:
            if 0 <= client_chosen_concept_index < len(fixed_concept_mappings):
                label_map = fixed_concept_mappings[client_chosen_concept_index] # This IS the map {old_label: new_label}
                if isinstance(label_map, dict):
                    for i, label_val in enumerate(labels_list):
                        labels_list[i] = label_map.get(label_val, label_val) # Apply mapping
                    drift_applied = True
                else:
                    print(f"Warning (client {client_id}, round {current_round}): Expected a dict for label_map at index {client_chosen_concept_index} for fixed_concept_switch, but got {type(label_map)}.")
            else:
                print(f"Warning (client {client_id}, round {current_round}): client_chosen_concept_index {client_chosen_concept_index} is out of bounds for fixed_concept_mappings (len: {len(fixed_concept_mappings)}).")
        
        return drift_applied, current_concept_id

    # 2. 渐进漂移 gradual
    if scenario == "gradual":
        from_concept_val = drift_config.get('from_concept_val', 0) # Actual label value of the old concept
        to_concept_val = drift_config.get('to_concept_val', 1)     # Actual label value of the new concept
        
        # Concept IDs for reporting (can be same as label values or different if needed)
        # For simplicity, using label values as concept IDs here.
        from_concept_id_report = from_concept_val 
        to_concept_id_report = to_concept_val

        start_round = drift_config.get('gradual_drift_start_round', base_epoch)
        W = drift_config.get('gradual_window', 10)
        if W <= 0: 
            print(f"Warning (client {client_id}, round {current_round}): gradual_window is {W}. Setting to 1.")
            W = 1 

        r = current_round - start_round # rounds since drift process initiated

        if r < 0: # Before drift window starts
            current_concept_id = from_concept_id_report
        elif 0 <= r < W: # Within the drift window
            # p is the probability that a sample that was 'from_concept_val' should now be 'to_concept_val'
            # or the probability that the client is considered in 'to_concept_id_report' state.
            # p goes from 0/W (at r=0) to (W-1)/W (at r=W-1).
            p_progress = float(r) / W 

            # Determine reported concept ID probabilistically during transition
            if random.random() < p_progress: # With probability p_progress, report as 'to_concept'
                current_concept_id = to_concept_id_report
            else:
                current_concept_id = from_concept_id_report
            
            # Apply probabilistic drift to labels
            for i, label_val in enumerate(labels_list):
                if label_val == from_concept_val:
                    if random.random() < p_progress: # This specific sample drifts
                        labels_list[i] = to_concept_val
                        drift_applied = True
        else: # r >= W, drift window finished or passed
            current_concept_id = to_concept_id_report
            # Ensure all 'from_concept_val' are now 'to_concept_val'
            for i, label_val in enumerate(labels_list):
                if label_val == from_concept_val:
                    if labels_list[i] != to_concept_val: # Check to set drift_applied correctly
                        labels_list[i] = to_concept_val
                        drift_applied = True
        
        return drift_applied, current_concept_id

    # 3. 循环漂移 recurring
    if scenario == "recurring":
        recurring_schedule = drift_config.get("recurring_schedule", None) 
        if recurring_schedule is None:
            # Default schedule if none provided: switches between concept_map_idx 1 and 0
            # Assumes fixed_concept_mappings[0] and fixed_concept_mappings[1] exist.
            print(f"Warning (client {client_id}, round {current_round}): recurring_schedule is None. Using a default: [(base, base+50, 1), (base+50, base+100, 0)].")
            # Durations for default schedule
            duration1 = drift_config.get("recurring_default_duration1", 50)
            duration2 = drift_config.get("recurring_default_duration2", 50)
            default_concept_idx1 = drift_config.get("recurring_default_concept_idx1", 1)
            default_concept_idx0 = drift_config.get("recurring_default_concept_idx0", 0)

            recurring_schedule = [
                (base_epoch, base_epoch + duration1, default_concept_idx1),
                (base_epoch + duration1, base_epoch + duration1 + duration2, default_concept_idx0)
            ]
        
        active_concept_map_idx = 0 # Default to concept map index 0 (e.g., initial state)
        # Determine the current concept_map_idx for this round based on the schedule
        for (sch_start_round, sch_end_round, concept_map_idx_from_schedule) in recurring_schedule:
            if sch_start_round <= current_round < sch_end_round:
                active_concept_map_idx = concept_map_idx_from_schedule
                break 
        current_concept_id = active_concept_map_idx # This map index is the reported concept ID

        # Apply label transformation only at the START of a new concept period in the schedule
        for (sch_start_round, sch_end_round, concept_map_idx_from_schedule) in recurring_schedule:
            if current_round == sch_start_round: # This is a switch point
                if fixed_concept_mappings:
                    if 0 <= concept_map_idx_from_schedule < len(fixed_concept_mappings):
                        label_map = fixed_concept_mappings[concept_map_idx_from_schedule]
                        if isinstance(label_map, dict):
                            for i, label_val in enumerate(labels_list):
                                labels_list[i] = label_map.get(label_val, label_val)
                            drift_applied = True
                            # current_concept_id is already set to active_concept_map_idx based on the current round.
                            break # Applied drift for this round's switch point
                        else:
                            print(f"Warning (client {client_id}, round {current_round}): Expected a dict for label_map for recurring concept index {concept_map_idx_from_schedule}, but got {type(label_map)}.")
                    else:
                        print(f"Warning (client {client_id}, round {current_round}): Concept index {concept_map_idx_from_schedule} for recurring drift is out of bounds for fixed_concept_mappings (len: {len(fixed_concept_mappings) if fixed_concept_mappings else 'None'}).")
                else:
                    print(f"Warning (client {client_id}, round {current_round}): fixed_concept_mappings is None, cannot apply recurring drift map.")
                break # Only apply one mapping if multiple schedules (improbably) start on the same round.
        
        return drift_applied, current_concept_id

    # If scenario not matched or other cases
    # print(f"Info (client {client_id}, round {current_round}): No complex drift applied for scenario '{scenario}'. Returning default concept_id {current_concept_id}.")
    return False, current_concept_id

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

