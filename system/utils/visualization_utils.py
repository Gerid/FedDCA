import os
import numpy as np
import torch # For params_dict[param_name] tensor check and other torch specific operations
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.spatial import Voronoi

def visualize_clustering_results(
    client_cluster_assignments: dict,
    client_cluster_assignments_history: dict,
    client_label_profiles: dict,
    client_classifier_params: dict,
    current_round: int,
    dataset_name: str,
    algorithm_name: str,
    save_folder_name: str,
    cluster_tsne_feature_source: str,
    cluster_visualization_type: str,
    ablation_no_lp: bool,
    args_seed: int,
    client_true_concept_id_history: dict = None # Added parameter for true concept IDs
):
    # First, try to plot the t-SNE/Voronoi visualization as before
    if not client_cluster_assignments:
        print("Visualization: No client cluster assignments available for t-SNE. Skipping t-SNE plot.")
        # Still try to plot heatmap if history exists
    else:
        client_ids_for_tsne = []
        feature_vectors = []
        cluster_labels_for_tsne = []
        
        plot_title_suffix = ""
        plot_filename_suffix_part = ""

        if cluster_tsne_feature_source == 'lp':
            plot_title_suffix = "LP"
            plot_filename_suffix_part = "lp_tsne"
            if ablation_no_lp or not any(client_label_profiles.values()):
                print("Visualization: Label profiles are disabled or no profiles available for t-SNE (source: LP). Skipping t-SNE plot.")
            else:
                for client_id, cluster_id in client_cluster_assignments.items():
                    profile = client_label_profiles.get(client_id)
                    if profile:
                        all_samples_for_client = []
                        for label, data_for_label in profile.items():
                            samples = None
                            if isinstance(data_for_label, tuple) and len(data_for_label) > 0:
                                samples = data_for_label[0]
                            elif isinstance(data_for_label, np.ndarray):
                                samples = data_for_label

                            if samples is not None and samples.size > 0:
                                if samples.ndim == 2 and samples.shape[1] > 0:
                                    all_samples_for_client.append(samples)
                                elif samples.ndim == 1:
                                    all_samples_for_client.append(samples.reshape(1, -1))
                        
                        if all_samples_for_client:
                            try:
                                if len(all_samples_for_client) > 1 and any(s.shape[1] != all_samples_for_client[0].shape[1] for s in all_samples_for_client if s.ndim == 2 and all_samples_for_client[0].ndim == 2 and all_samples_for_client[0].size > 0 and s.size > 0):
                                    if all_samples_for_client[0].shape[0] > 0 :
                                         client_feature_vector = np.mean(all_samples_for_client[0], axis=0)
                                    else:
                                        continue
                                else:
                                    concatenated_samples = np.vstack(all_samples_for_client)
                                    if concatenated_samples.shape[0] > 0:
                                        client_feature_vector = np.mean(concatenated_samples, axis=0)
                                    else:
                                        continue
                                
                                feature_vectors.append(client_feature_vector)
                                client_ids_for_tsne.append(client_id)
                                cluster_labels_for_tsne.append(cluster_id)
                            except (ValueError, IndexError) as e:
                                # print(f"Visualization (LP): Error processing samples for client {client_id}: {e}. Skipping.")
                                continue
                if not feature_vectors:
                    print("Visualization (LP): No LP feature vectors extracted for t-SNE. Skipping t-SNE plot.")

        elif cluster_tsne_feature_source == 'model_params':
            plot_title_suffix = "Model Params"
            plot_filename_suffix_part = "params_tsne"
            relevant_client_params = {cid: params for cid, params in client_classifier_params.items() if cid in client_cluster_assignments}

            if not relevant_client_params:
                print("Visualization: No relevant client classifier parameters available for t-SNE. Skipping t-SNE plot.")
            else:
                for client_id, cluster_id in client_cluster_assignments.items():
                    params_dict = relevant_client_params.get(client_id)
                    if params_dict:
                        all_param_tensors_flat = []
                        for param_name in sorted(params_dict.keys()):
                            param_tensor = params_dict[param_name]
                            if isinstance(param_tensor, torch.Tensor):
                                all_param_tensors_flat.append(param_tensor.detach().cpu().numpy().flatten())
                        
                        if all_param_tensors_flat:
                            client_feature_vector = np.concatenate(all_param_tensors_flat)
                            if client_feature_vector.size > 0:
                                feature_vectors.append(client_feature_vector)
                                client_ids_for_tsne.append(client_id)
                                cluster_labels_for_tsne.append(cluster_id)
                
                if not feature_vectors:
                     print("Visualization (Model Params): No model parameter feature vectors extracted for t-SNE. Skipping t-SNE plot.")
        else:
            print(f"Visualization: Unknown tsne_feature_source: {cluster_tsne_feature_source}. Skipping t-SNE plot.")

        if feature_vectors:
            feature_vectors_np = np.array(feature_vectors)
            
            if feature_vectors_np.ndim == 1:
                if len(feature_vectors) == 1:
                     feature_vectors_np = feature_vectors_np.reshape(1, -1)
                else: 
                    print("Visualization: Feature vectors array is 1D unexpectedly for multiple clients for t-SNE. Skipping t-SNE plot.")
                    feature_vectors_np = None 
            
            if feature_vectors_np is not None and (feature_vectors_np.shape[0] <= 1 or feature_vectors_np.shape[1] == 0):
                print("Visualization: Not enough data points or zero features for t-SNE. Skipping t-SNE plot.")
                feature_vectors_np = None
                
            if feature_vectors_np is not None:
                n_samples_for_tsne = feature_vectors_np.shape[0]
                perplexity_value = min(30.0, float(n_samples_for_tsne - 1))
                
                tsne_results = None # Initialize tsne_results
                if perplexity_value < 1.0: 
                    print("Visualization: Perplexity too low for t-SNE. Skipping t-SNE plot.")
                else:
                    try:
                        if cluster_tsne_feature_source == 'model_params':
                            scaler = StandardScaler()
                            feature_vectors_np = scaler.fit_transform(feature_vectors_np)

                        tsne = TSNE(n_components=2, 
                                    random_state=args_seed, 
                                    perplexity=perplexity_value, 
                                    n_iter=1000, 
                                    init='pca', 
                                    learning_rate='auto')
                        tsne_results = tsne.fit_transform(feature_vectors_np)
                    except Exception as e: 
                        try:
                            tsne = TSNE(n_components=2, random_state=args_seed, perplexity=perplexity_value, n_iter=1000, init='pca', learning_rate=200.0) 
                            tsne_results = tsne.fit_transform(feature_vectors_np)
                        except Exception as e_retry:
                            print(f"Visualization: Error during t-SNE: {e_retry}. Skipping t-SNE plot.")
                            # tsne_results remains None

                    if tsne_results is not None:
                        fig, ax = plt.subplots(figsize=(13, 11))
                        unique_clusters = sorted(list(set(cluster_labels_for_tsne)))
                        
                        if not unique_clusters:
                            plt.close(fig)
                        else:
                            if len(unique_clusters) <= 10 and len(unique_clusters) > 0:
                                colors = plt.cm.get_cmap('tab10', len(unique_clusters)).colors
                            elif len(unique_clusters) > 0:
                                colors = plt.cm.get_cmap('viridis', len(unique_clusters)).colors
                            else:
                                colors = ['grey']
                            cluster_to_color = {cluster_id: colors[i % len(colors)] for i, cluster_id in enumerate(unique_clusters)}

                            current_viz_type = cluster_visualization_type
                            if current_viz_type == 'voronoi' and n_samples_for_tsne < 4:
                                print(f"Visualization: Not enough points ({n_samples_for_tsne}) for Voronoi plot. Falling back to scatter plot.")
                                current_viz_type = 'scatter'

                            if current_viz_type == 'scatter':
                                cluster_points_indices = {uid: [] for uid in unique_clusters}
                                for idx, cl_id in enumerate(cluster_labels_for_tsne):
                                    cluster_points_indices[cl_id].append(idx)

                                for i, client_id_val in enumerate(client_ids_for_tsne): 
                                    cluster_id = cluster_labels_for_tsne[i]
                                    label = f'Cluster {cluster_id}' if i == cluster_points_indices[cluster_id][0] else None
                                    ax.scatter(tsne_results[i, 0], tsne_results[i, 1], 
                                               color=cluster_to_color.get(cluster_id, 'grey'), 
                                               label=label,
                                               s=60, alpha=0.85, edgecolors='w')
                            
                            elif current_viz_type == 'voronoi':
                                vor = Voronoi(tsne_results)
                                for r_idx, region_idx in enumerate(vor.point_region):
                                    region = vor.regions[region_idx]
                                    if not -1 in region:
                                        polygon = [vor.vertices[i] for i in region]
                                        cluster_id = cluster_labels_for_tsne[r_idx] 
                                        ax.fill(*zip(*polygon), alpha=0.4, color=cluster_to_color.get(cluster_id, 'lightgrey'))
                                
                                for i, client_id_val in enumerate(client_ids_for_tsne):
                                    cluster_id = cluster_labels_for_tsne[i]
                                    ax.scatter(tsne_results[i, 0], tsne_results[i, 1], 
                                               color=cluster_to_color.get(cluster_id, 'black'), 
                                               edgecolor='white', s=35, zorder=3)
                                ax.set_xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
                                ax.set_ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)

                            if current_viz_type == 'scatter':
                                handles, labels = ax.get_legend_handles_labels()
                                if handles:
                                    try:
                                        # Attempt to sort legend items numerically by cluster ID
                                        sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: int(x[1].split(' ')[-1]))
                                        ax.legend([h for h,l in sorted_handles_labels], [l for h,l in sorted_handles_labels], title="Client Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
                                    except (ValueError, IndexError): # Fallback if label format is unexpected
                                         ax.legend(handles=handles, labels=labels, title="Client Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
                            elif current_viz_type == 'voronoi':
                                handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {uid}', 
                                                      markerfacecolor=cluster_to_color.get(uid, 'grey'), markersize=10) for uid in unique_clusters]
                                if handles:
                                    ax.legend(handles=handles, title="Client Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')

                            title = f'Client Clustering (t-SNE Source: {plot_title_suffix}, Viz: {current_viz_type.capitalize()}) - Round {current_round}'
                            ax.set_title(title, fontsize=14)
                            ax.set_xlabel('t-SNE Component 1', fontsize=12)
                            ax.set_ylabel('t-SNE Component 2', fontsize=12)
                            ax.grid(True, linestyle='--', alpha=0.7)
                            fig.tight_layout(rect=[0, 0, 0.85, 1])

                            plot_filename_base = f"{dataset_name}_{algorithm_name}_round_{current_round}"
                            plot_filename = f"{plot_filename_base}_{plot_filename_suffix_part}_viz_{current_viz_type}.png"
                            
                            results_subdir = os.path.join(save_folder_name, "results_extra_visualizations")
                            if not os.path.exists(results_subdir):
                                try:
                                    os.makedirs(results_subdir)
                                except OSError as e:
                                    print(f"Error creating directory {results_subdir}: {e}")
                                    plt.close(fig) # Close the figure if directory creation fails
                            if os.path.exists(results_subdir): # Proceed only if directory exists or was created
                                plot_path = os.path.join(results_subdir, plot_filename)
                                try:
                                    fig.savefig(plot_path, bbox_inches='tight')
                                except Exception as e:
                                    print(f"Visualization: Error saving t-SNE plot to {plot_path}: {e}")
                            plt.close(fig)

    # Now, generate the heatmap for cluster assignments over rounds
    if not client_cluster_assignments_history:
        print("Visualization: No client cluster assignment history available for heatmap (cluster IDs for text). Skipping heatmap.")
        return

    # Check if true concept ID history is provided for coloring
    if client_true_concept_id_history is None:
        print("Visualization: True concept ID history not provided. Cannot color heatmap by true concept IDs. Skipping heatmap.")
        return

    client_ids_sorted = sorted(list(client_cluster_assignments_history.keys()))
    if not client_ids_sorted:
        print("Visualization: No client IDs in cluster assignment history for heatmap. Skipping heatmap.")
        return

    all_rounds_with_assignments = set()
    for client_id in client_ids_sorted:
        all_rounds_with_assignments.update(client_cluster_assignments_history[client_id].keys())
    
    if not all_rounds_with_assignments:
        print("Visualization: No rounds with assignments in cluster history for heatmap. Skipping heatmap.")
        return
            
    rounds_sorted = sorted(list(all_rounds_with_assignments))
    
    # Data for cell text (cluster IDs)
    cluster_id_heatmap_data = np.full((len(client_ids_sorted), len(rounds_sorted)), np.nan) 
    for i, client_id in enumerate(client_ids_sorted):
        for j, round_num in enumerate(rounds_sorted):
            cluster_id_heatmap_data[i, j] = client_cluster_assignments_history[client_id].get(round_num, np.nan)

    if np.all(np.isnan(cluster_id_heatmap_data)):
        print("Visualization: Cluster ID heatmap data is all NaN. Skipping heatmap.")
        return

    # Data for cell color (true concept IDs)
    true_concept_heatmap_data = np.full((len(client_ids_sorted), len(rounds_sorted)), np.nan)
    for i, client_id in enumerate(client_ids_sorted):
        if client_id in client_true_concept_id_history:
            for j, round_num in enumerate(rounds_sorted):
                true_concept_heatmap_data[i, j] = client_true_concept_id_history[client_id].get(round_num, np.nan)
        else:
            # If a client has cluster assignments but no true concept ID history, fill with NaN for color
            # This case should ideally be handled by ensuring complete true concept ID history is passed
            print(f"Visualization Warning: Client {client_id} missing in true_concept_id_history. Colors for this client will be NaN-based.")


    if np.all(np.isnan(true_concept_heatmap_data)):
        print("Visualization: True concept ID heatmap data is all NaN. Skipping heatmap based on true concept IDs.")
        return

    fig_heatmap, ax_heatmap = plt.subplots(figsize=(max(10, len(rounds_sorted) * 0.5), max(6, len(client_ids_sorted) * 0.3)))
    
    # Determine min/max true concept IDs from the actual data in true_concept_heatmap_data
    min_true_concept_in_data = float('inf')
    max_true_concept_in_data = -1 # Changed from float('-inf') to handle cases where all are NaN or only non-negative IDs
    has_valid_true_concept_id_in_data = False
    for r_idx in range(true_concept_heatmap_data.shape[0]):
        for c_idx in range(true_concept_heatmap_data.shape[1]):
            val = true_concept_heatmap_data[r_idx, c_idx]
            if val is not None and not np.isnan(val):
                min_true_concept_in_data = min(min_true_concept_in_data, int(val))
                max_true_concept_in_data = max(max_true_concept_in_data, int(val))
                has_valid_true_concept_id_in_data = True
    
    if not has_valid_true_concept_id_in_data:
        print("Visualization: No valid true concept IDs found in history data for heatmap coloring. Skipping heatmap.")
        plt.close(fig_heatmap)
        return

    # Define known concept counts for datasets (for true concept ID range)
    dataset_to_n_concepts = {
        'cifar100': 20,
        'cifar10': 10,
        'fmnist': 10,
        'mnist': 10,
        # Add other datasets and their true concept counts here
    }

    final_vmin_color = 0
    final_vmax_color = 0
    effective_num_colors = 1

    n_concepts_for_dataset = dataset_to_n_concepts.get(dataset_name.lower())

    if n_concepts_for_dataset is not None:
        print(f"Visualization: Using fixed concept count {n_concepts_for_dataset} for dataset {dataset_name} for true concept ID heatmap colormap.")
        final_vmin_color = 0
        final_vmax_color = n_concepts_for_dataset -1 # Max concept ID is count - 1
        effective_num_colors = n_concepts_for_dataset
        if min_true_concept_in_data < final_vmin_color or max_true_concept_in_data > final_vmax_color:
            print(f"Visualization Warning: True concept ID data range [{min_true_concept_in_data}, {max_true_concept_in_data}] is outside the expected concept range [{final_vmin_color}, {final_vmax_color}] for {dataset_name}. Colors will be clipped.")
    else:
        print(f"Visualization: Dataset {dataset_name} not in predefined concept map or concept count not specified. Using dynamic range for true concept ID heatmap colormap.")
        final_vmin_color = min_true_concept_in_data
        final_vmax_color = max_true_concept_in_data
        if final_vmin_color > final_vmax_color:
             final_vmax_color = final_vmin_color 
        effective_num_colors = final_vmax_color - final_vmin_color + 1
    
    if effective_num_colors <= 0:
        effective_num_colors = 1

    cmap_name = 'viridis' if effective_num_colors > 10 else 'tab10' 
    try:
        if cmap_name == 'tab10' and effective_num_colors > 10:
             cmap = plt.get_cmap('viridis', effective_num_colors if effective_num_colors > 0 else None)
        else:
             cmap = plt.get_cmap(cmap_name, effective_num_colors if effective_num_colors > 0 else None)
    except ValueError:
        cmap = plt.get_cmap(cmap_name)

    # Use true_concept_heatmap_data for coloring
    cax = ax_heatmap.matshow(true_concept_heatmap_data, aspect='auto', cmap=cmap, vmin=final_vmin_color, vmax=final_vmax_color)
    
    # Colorbar setup for true concept IDs
    colorbar_label = 'True Concept ID'
    if final_vmax_color >= final_vmin_color :
        ticks = np.arange(final_vmin_color, final_vmax_color + 1)
        if len(ticks) > 0:
             if len(ticks) > 20: 
                 ticks = np.linspace(final_vmin_color, final_vmax_color, num=min(effective_num_colors, 10), dtype=int)
             cbar = fig_heatmap.colorbar(cax, ticks=ticks, fraction=0.046, pad=0.04)
             cbar.set_label(colorbar_label, rotation=270, labelpad=15)
        else: 
             cbar = fig_heatmap.colorbar(cax, ticks=[final_vmin_color], fraction=0.046, pad=0.04)
             cbar.set_label(colorbar_label, rotation=270, labelpad=15)
    else:
        cbar = fig_heatmap.colorbar(cax, fraction=0.046, pad=0.04)
        cbar.set_label(colorbar_label, rotation=270, labelpad=15)

    ax_heatmap.set_xticks(np.arange(len(rounds_sorted)))
    ax_heatmap.set_xticklabels([str(r) for r in rounds_sorted], rotation=45, ha="left")
    ax_heatmap.set_yticks(np.arange(len(client_ids_sorted)))
    ax_heatmap.set_yticklabels([str(cid) for cid in client_ids_sorted])

    ax_heatmap.set_xlabel("Communication Round")
    ax_heatmap.set_ylabel("Client ID")
    ax_heatmap.set_title(f"Client Cluster Assignments Over Rounds (Current: {current_round})", pad=20)
    
    if cluster_id_heatmap_data.size < 500: # Text based on cluster_id_heatmap_data
        for i in range(cluster_id_heatmap_data.shape[0]):
            for j in range(cluster_id_heatmap_data.shape[1]):
                cluster_id_val = cluster_id_heatmap_data[i, j]
                true_concept_id_val = true_concept_heatmap_data[i,j]

                if not np.isnan(cluster_id_val): # Only show text if cluster ID is valid
                    # Determine text color based on cell background color (from true_concept_id_val)
                    norm_val = 0.0 
                    if not np.isnan(true_concept_id_val): # Ensure true concept ID is valid for color normalization
                        if final_vmax_color > final_vmin_color:
                            clipped_value = np.clip(true_concept_id_val, final_vmin_color, final_vmax_color)
                            norm_val = (clipped_value - final_vmin_color) / (final_vmax_color - final_vmin_color)
                        elif final_vmax_color == final_vmin_color: 
                             norm_val = 0.5 
                    else: # Fallback if true_concept_id_val is NaN, e.g. use a mid-grey for norm_val
                        norm_val = 0.5


                    cell_color_rgb = cmap(norm_val)
                    text_color = 'white' if cell_color_rgb[0]*0.299 + cell_color_rgb[1]*0.587 + cell_color_rgb[2]*0.114 < 0.5 else 'black'
                    ax_heatmap.text(j, i, str(int(cluster_id_val)), va='center', ha='center', color=text_color, fontsize=8)

    fig_heatmap.tight_layout(rect=[0, 0, 0.95, 1]) 
    heatmap_plot_filename = f"{dataset_name}_{algorithm_name}_round_{current_round}_cluster_heatmap.png"
    results_subdir_heatmap = os.path.join(save_folder_name, "results_extra_visualizations")
    if not os.path.exists(results_subdir_heatmap):
        try:
            os.makedirs(results_subdir_heatmap)
        except OSError as e:
            print(f"Error creating directory {results_subdir_heatmap} for heatmap: {e}")
            plt.close(fig_heatmap)
            return
    if os.path.exists(results_subdir_heatmap): # Proceed only if directory exists
        heatmap_plot_path = os.path.join(results_subdir_heatmap, heatmap_plot_filename)
        try:
            fig_heatmap.savefig(heatmap_plot_path, bbox_inches='tight')
            print(f"Visualization: Cluster assignment heatmap saved to {heatmap_plot_path}")
        except Exception as e:
            print(f"Visualization: Error saving heatmap plot to {heatmap_plot_path}: {e}")
    plt.close(fig_heatmap)
