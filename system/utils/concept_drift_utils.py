# import numpy as np # Not strictly needed for the revised drift_dataset logic if not used elsewhere
import torch  # Import torch


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

