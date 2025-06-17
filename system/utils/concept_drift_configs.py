import torch

# -----------------------------------------------------------------------------
# Configuration for Concept Drift Scenarios
# -----------------------------------------------------------------------------

# Scenario 1: Global Intra-Superclass Drift
# All clients experience drift simultaneously at 'base_epoch'.
# Labels change to other labels within the same superclass.
config_global_intra = {
    "scenario_type": "global_intra_superclass",
    "base_epoch": 50  # Epoch at which drift occurs for all clients
}

# Scenario 2: Staggered Inter-Superclass Drift
# Clients experience drift at different epochs, determined by 'base_epoch' and 'stagger_interval'.
# Labels change to labels from a different superclass.
config_staggered_inter = {
    "scenario_type": "staggered_inter_superclass",
    "base_epoch": 70,         # Epoch for client 0 to drift
    "stagger_interval": 5     # Interval in epochs between subsequent clients drifting
}

# Scenario 3: Partial Intra-Superclass Drift
# All clients experience drift at 'base_epoch', but only a 'partial_percentage'
# of their data samples are affected.
# Labels change to other labels within the same superclass.
config_partial_intra = {
    "scenario_type": "partial_intra_superclass",
    "base_epoch": 120,
    "partial_percentage": 0.20 # 20% of samples will drift
}

# Scenario 4: Global Inter-Superclass Drift
# All clients experience drift simultaneously at 'base_epoch'.
# Labels change to labels from a different superclass.
config_global_inter = {
    "scenario_type": "global_inter_superclass",
    "base_epoch": 150
}

# Scenario 5: Staggered Intra-Superclass Drift
# Clients experience drift at different epochs.
# Labels change to other labels within the same superclass.
config_staggered_intra = {
    "scenario_type": "staggered_intra_superclass",
    "base_epoch": 100,
    "stagger_interval": 10
}

# Scenario 6: Partial Inter-Superclass Drift
# All clients experience drift at 'base_epoch', affecting a 'partial_percentage' of samples.
# Labels change to labels from a different superclass.
config_partial_inter = {
    "scenario_type": "partial_inter_superclass",
    "base_epoch": 180,
    "partial_percentage": 0.30 # 30% of samples will drift
}

# --- How to use these configurations ---
# 1. Import these configurations into your main training script or server logic.
#    from .concept_drift_configs import config_global_intra, config_staggered_inter # Adjust path as needed

# 2. Load your superclass maps (typically once):
#    from system.utils.concept_drift_utils import load_superclass_maps
#    fine_to_coarse, coarse_to_fine, all_fine, all_coarse = load_superclass_maps("path/to/your/superclass_map.json")
#    superclass_maps = (fine_to_coarse, coarse_to_fine, all_fine, all_coarse)

# 3. In your training loop (e.g., per client, per epoch), call apply_complex_drift:
#    from system.utils.concept_drift_utils import apply_complex_drift
#
#    # Example for a client within its training routine:
#    # current_epoch = ... (current global training epoch)
#    # client_id = client.id
#    # client_dataset_targets = [label for _, label in client.train_loader.dataset.dataset_targets] # Or however you access raw labels
#
#    # Choose a config, e.g., config_staggered_inter
#    chosen_drift_config = config_staggered_inter
#
#    # Make a copy of targets to modify if your dataset stores them directly
#    # If your dataset is a list of (data, target) tuples, you'll need to reconstruct it.
#    modifiable_targets = list(client_dataset_targets) 
#
#    drift_applied = apply_complex_drift(
#        dataset_targets=modifiable_targets, # This list will be modified in-place
#        client_id=client_id,
#        current_epoch=current_epoch,
#        drift_config=chosen_drift_config,
#        superclass_maps=superclass_maps
#    )
#
#    if drift_applied:
#        # Update the client's actual dataset with the modified targets
#        # This depends on how your Dataset/DataLoader is structured.
#        # For example, if client.train_loader.dataset.dataset_targets was the source:
#        # client.train_loader.dataset.dataset_targets = modifiable_targets
#        # Or if it's a list of tuples:
#        # client.train_dataset = [(client.train_dataset[i][0], modifiable_targets[i]) for i in range(len(client.train_dataset))]
#        print(f"Client {client_id}: Drift scenario '{chosen_drift_config['scenario_type']}' applied at epoch {current_epoch}.")

# Note: The `apply_complex_drift` function expects `dataset_targets` to be a list of labels
# that it can modify in place. Ensure that the labels passed are indeed modifiable and that
# any changes are correctly propagated back to the client's dataset used for training.
# The labels in `dataset_targets` should be the raw fine-grained labels (e.g., integers).
