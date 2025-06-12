import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import shutil
from matplotlib.ticker import MaxNLocator # Added for better tick control

try:
    import wandb
except ImportError:
    print("wandb library not found. Please install it ('pip install wandb') to use WandB integration.")
    wandb = None # Set to None so checks for 'api' and 'wandb' will fail gracefully


# --- Configuration ---
BASE_DIR_PATTERN = "feddca_direct_commands_output_*"
ANALYSIS_OUTPUT_DIR = "analysis_results_auto" # Directory to save plots and tables

# --- WandB Configuration (NEEDS TO BE SET BY USER IF USING WANDB) ---
WANDB_PROJECT = "FedDCA_Impact_Studies_Cifar100"  # As per your Experiments.sh
WANDB_ENTITY = "Gerid"  # !!! REPLACE THIS with your WandB username or team name !!!
WANDB_ACCURACY_METRIC_CANDIDATES = [
    'Global_Test_Accuracy', 'Global_ Test Accuracy', 'Test Accuracy', 
    'val_acc', 'test_accuracy', 'accuracy', 'Average Test Acc',
    'server_agg_accuracy' # Added from a pattern
]
WANDB_ROUND_METRIC_CANDIDATES = ['_step', 'epoch', 'round', 'Round', 'Communication round']


# Regex patterns to extract info from .out files
PATTERNS_ROUND_ACC = [
    # New pattern for the provided format, prioritizing Global_ Test Accuracy
    re.compile(r"-------------Round number:\\s*(\\d+)-------------.*?Global_ Test Accuracy:\\s*([0-9.]+)", re.DOTALL | re.IGNORECASE),
    # Existing patterns (some might be redundant or less specific now, can be pruned later if needed)
    re.compile(r"Round\\\\s*(\\\\d+).*?Test Accuracy:\\\\s*([0-9.]+)", re.IGNORECASE),
    re.compile(r"Global Round:\\\\s*(\\\\d+).*?Average Test Acc(?:uracy)?:\\\\s*([0-9.]+)", re.IGNORECASE),
    re.compile(r"Communication round[:\\\\s]*(\\\\d+).*Test accuracy[:\\\\s]*([0-9.]+)", re.IGNORECASE),
    re.compile(r"epoch[:\\\\s]*(\\\\d+).*accuracy[:\\\\s]*([0-9.]+)", re.IGNORECASE), 
    re.compile(r"iter[:\\\\s]*(\\\\d+).*test_acc[:\\\\s]*([0-9.]+)", re.IGNORECASE), 
    re.compile(r"round\\\\s+(\\\\d+)\\\\s*test_accuracy:\\\\s*([0-9.]+)", re.IGNORECASE),
    re.compile(r"round\\\\s+(\\\\d+),\\\\s*server_agg_accuracy:\\\\s*([0-9.]+)", re.IGNORECASE),
    re.compile(r"Acc:\\\\s*([0-9.]+).*?round\\\\s*(\\\\d+)", re.IGNORECASE),
]

BEST_ACC_PATTERNS = [
    # New pattern for best global accuracy if it follows a similar new format
    re.compile(r"Best Global_ Test Accuracy:\\s*([0-9.]+)", re.IGNORECASE), 
    # Existing patterns
    re.compile(r"Best Test Accuracy:\\\\s*([0-9.]+)", re.IGNORECASE),
    re.compile(r"Best Global Test Accuracy:\\\\s*([0-9.]+)", re.IGNORECASE), # Potentially redundant if covered by the new one
    re.compile(r"Highest Test Accuracy:\\\\s*([0-9.]+)", re.IGNORECASE),
]

# --- Helper Functions ---

def find_latest_output_directory(base_path=".", pattern=BASE_DIR_PATTERN):
    """Finds the most recent directory matching the pattern."""
    dirs = glob.glob(os.path.join(base_path, pattern))
    if not dirs:
        return None
    
    latest_dir = None
    latest_time = datetime.min
    
    for d in dirs:
        if os.path.isdir(d):
            try:
                timestamp_str = os.path.basename(d).split('_')[-2:]
                timestamp_str = "_".join(timestamp_str)
                dir_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                if dir_time > latest_time:
                    latest_time = dir_time
                    latest_dir = d
            except (ValueError, IndexError):
                pass

    if latest_dir:
        return latest_dir
    
    if dirs:
        return sorted(dirs)[-1]
    return None


def parse_experiment_from_filename(filename_no_ext):
    parts = filename_no_ext.split('_')
    # Assuming experiment_group is the first two parts if available
    if len(parts) >= 2:
        experiment_group = parts[0] + "_" + parts[1]
        run_specific_name_parts = parts[2:]
    else: # Fallback if filename is too short (e.g. "LP_Ablation_NoLP" vs "MyRun")
        experiment_group = parts[0] 
        run_specific_name_parts = parts[1:]

    run_specific_name = "_".join(run_specific_name_parts)
    if not run_specific_name and len(parts) > 1: # Handle cases like "LP_Ablation_NoLP" where split might be "LP_Ablation", "NoLP"
        run_specific_name = parts[-1] if len(parts) > 1 else "_".join(parts)


    param_name = "variant"
    param_value_str = run_specific_name

    if experiment_group == "LP_Ablation":
        param_name = "lp_setting"
        if "FeatureBasedLP" in run_specific_name:
            param_value_str = "FeatureBasedLP"
        elif "NoLP" in run_specific_name:
            param_value_str = "NoLP"
        elif "ClassCountLP" in run_specific_name:
            param_value_str = "ClassCountLP"
        else:
            param_value_str = run_specific_name
    elif experiment_group == "Impact_NumProfileSamples":
        param_name = "num_profile_samples"
        match = re.search(r"NPS_(\\d+)", run_specific_name)
        if match: param_value_str = match.group(1)
    elif experiment_group == "Impact_OTReg":
        param_name = "dca_ot_reg"
        match = re.search(r"OTReg_(\\dp\\d+)", run_specific_name) 
        if match: param_value_str = match.group(1).replace('p', '.')
    elif experiment_group == "Impact_VWCReg":
        param_name = "dca_vwc_reg"
        match = re.search(r"VWCReg_(\\dp\\d+)", run_specific_name)
        if match: param_value_str = match.group(1).replace('p', '.')
    elif experiment_group == "Impact_NumClusters":
        param_name = "dca_target_num_clusters"
        match = re.search(r"Clusters_(\\d+)", run_specific_name)
        if match: param_value_str = match.group(1)
    elif experiment_group.startswith("UserExample"): # Handle UserExample group
        param_name = "user_config"
        param_value_str = run_specific_name # e.g., Cifar100_DriftDataset_Cmd
    else: 
        # Generic fallback if group name doesn't match specific patterns
        # This might happen if experiment_group was just one word.
        if not run_specific_name_parts: # If parts[2:] was empty
             run_specific_name = "_".join(parts[1:]) # Try to get a run_specific_name from parts[1:]
        
        # Re-evaluate experiment_group if it was too broad
        # Example: filename "Impact_OTReg_OTReg_0p01" -> parts[0]="Impact", parts[1]="OTReg"
        # exp_group should be "Impact_OTReg"
        # run_specific_name should be "OTReg_0p01"
        # The initial split might make exp_group "Impact_OTReg" and run_specific_name "OTReg_0p01" which is fine.
        # Consider a filename like "MyCustomExperiment_SettingA"
        # parts[0]="MyCustomExperiment", parts[1]="SettingA"
        # exp_group = "MyCustomExperiment_SettingA", run_specific_name = "" (problem)
        # Need to ensure run_specific_name is properly captured.
        
        # Let's refine the initial split logic for robustness
        # The goal is: experiment_group = first two words, run_specific_name = the rest
        # But if only two words, exp_group = first, run_specific_name = second
        
        # The current logic:
        # experiment_group = parts[0] + "_" + parts[1] (if len >=2)
        # run_specific_name = "_".join(parts[2:])
        # This seems mostly fine for the defined structures in Experiments.sh

        param_name = "config" # Default param_name
        param_value_str = run_specific_name # Default param_value_str

    try:
        if '.' in param_value_str:
            param_value = float(param_value_str)
        else:
            param_value = int(param_value_str)
    except ValueError:
        param_value = param_value_str

    return experiment_group, run_specific_name, param_name, param_value


def parse_output_file_content(filepath):
    round_accuracies = {} 
    best_overall_accuracy = 0.0
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                for pattern in PATTERNS_ROUND_ACC:
                    match = pattern.search(line)
                    if match:
                        try:
                            round_num = int(match.group(1))
                            acc = float(match.group(2))
                            if acc > 1.0 and acc <= 100.0:
                                acc = acc / 100.0 
                            if 0.0 <= acc <= 1.0:
                                round_accuracies[round_num] = acc
                            break 
                        except ValueError:
                            continue

                for pattern in BEST_ACC_PATTERNS:
                    match = pattern.search(line)
                    if match:
                        try:
                            acc = float(match.group(1))
                            if acc > 1.0 and acc <= 100.0:
                                acc = acc / 100.0
                            if 0.0 <= acc <= 1.0:
                                best_overall_accuracy = max(best_overall_accuracy, acc)
                            break 
                        except ValueError:
                            continue
    except Exception as e:
        print(f"Error reading or parsing file {filepath}: {e}")
        return [], 0.0, 0.0

    final_accuracy = 0.0
    calculated_best_from_rounds = 0.0

    if round_accuracies:
        sorted_rounds = sorted(round_accuracies.keys())
        if sorted_rounds:
            final_accuracy = round_accuracies[sorted_rounds[-1]]
            calculated_best_from_rounds = max(round_accuracies.values())
    
    best_accuracy = max(best_overall_accuracy, calculated_best_from_rounds)
    detailed_round_accuracies = sorted(round_accuracies.items())

    return detailed_round_accuracies, best_accuracy, final_accuracy


def extract_wandb_run_id_from_log(filepath):
    """Extracts WandB run ID from the .out log file."""
    # Regex to find the wandb run URL and capture the run ID
    # Example: wandb: 🚀 View run at https://wandb.ai/gerid/FedDCA_Impact_Studies_Cifar100/runs/zqhaahfy
    # It should capture 'zqhaahfy'
    wandb_url_pattern = re.compile(r"wandb:.*?https://wandb\.ai/.+?/.+?/runs/([a-zA-Z0-9]+)")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                match = wandb_url_pattern.search(line)
                if match:
                    run_id = match.group(1)
                    print(f"Found WandB run ID in log {os.path.basename(filepath)}: {run_id}")
                    return run_id
    except Exception as e:
        print(f"Error reading log file {filepath} to find WandB run ID: {e}")
    return None


def fetch_and_process_wandb_data(api, project, entity, run_name_or_id):
    """Fetches and processes data for a given run from WandB."""
    try:
        run = api.run(f"{entity}/{project}/{run_name_or_id}")
        # Fetch history with potentially relevant keys, then select
        history_df = run.history(pandas=True, stream="default") 
        config = run.config

        if history_df is None or history_df.empty:
            print(f"WandB: No history found for run {run_name_or_id}")
            return None

        round_col, acc_col = None, None
        for r_cand in WANDB_ROUND_METRIC_CANDIDATES:
            if r_cand in history_df.columns:
                round_col = r_cand
                break
        for a_cand in WANDB_ACCURACY_METRIC_CANDIDATES:
            if a_cand in history_df.columns:
                acc_col = a_cand
                break
        
        if not round_col or not acc_col:
            print(f"WandB: Could not find required round/accuracy columns in history for {run_name_or_id}.")
            print(f"  Available columns: {history_df.columns.tolist()}")
            print(f"  Searched round keys: {WANDB_ROUND_METRIC_CANDIDATES}, (found: {round_col})")
            print(f"  Searched accuracy keys: {WANDB_ACCURACY_METRIC_CANDIDATES}, (found: {acc_col})")
            return None

        # Ensure round and accuracy columns are numeric and drop NaNs
        history_df = history_df[pd.to_numeric(history_df[round_col], errors='coerce').notnull()]
        history_df = history_df[pd.to_numeric(history_df[acc_col], errors='coerce').notnull()]
        history_df[round_col] = pd.to_numeric(history_df[round_col])
        history_df[acc_col] = pd.to_numeric(history_df[acc_col])
        
        history_df = history_df.dropna(subset=[round_col, acc_col])
        if history_df.empty:
            print(f"WandB: History became empty after processing for {run_name_or_id}")
            return None

        round_accuracies_list = []
        for _, row in history_df.iterrows():
            round_num = row[round_col]
            acc = row[acc_col]
            if acc > 1.0 and acc <= 100.0:  # Normalize if percentage
                acc /= 100.0
            if 0.0 <= acc <= 1.0: # Ensure accuracy is a valid fraction
                round_accuracies_list.append((int(round_num), float(acc)))
        
        if not round_accuracies_list:
            print(f"WandB: No valid round accuracies extracted for {run_name_or_id}")
            return None

        round_accuracies_list.sort() # Sort by round number

        accuracies_only = [acc for _, acc in round_accuracies_list]
        best_accuracy = max(accuracies_only) if accuracies_only else 0.0
        final_accuracy = accuracies_only[-1] if accuracies_only else 0.0
        
        print(f"WandB: Successfully processed data for {run_name_or_id}. Rounds: {len(round_accuracies_list)}, Best Acc: {best_accuracy:.4f}")
        return round_accuracies_list, best_accuracy, final_accuracy, config

    except wandb.errors.CommError as e:
        print(f"WandB API error for run '{run_name_or_id}': {e}. (Run not found or auth issue)")
        return None
    except Exception as e:
        print(f"Error fetching/processing WandB data for run '{run_name_or_id}': {e}")
        return None


def generate_plots(all_results_df, output_dir_plots):
    if all_results_df.empty:
        print("No data to generate plots.")
        return

    os.makedirs(output_dir_plots, exist_ok=True)
    sns.set_theme(style="whitegrid")

    ablation_df = all_results_df[all_results_df['experiment_group'] == 'LP_Ablation'].copy()
    if not ablation_df.empty:
        ablation_df['lp_setting'] = ablation_df['param_value'].astype(str)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='lp_setting', y='best_accuracy', data=ablation_df, palette="viridis", order=['NoLP', 'ClassCountLP', 'FeatureBasedLP'])
        plt.title('Ablation Study: Impact of Label Profile (LP) Setting on Best Accuracy')
        plt.xlabel('Label Profile Setting')
        plt.ylabel('Best Test Accuracy')
        
        # Adjust Y-axis for ablation bar plot
        min_val_abl = ablation_df['best_accuracy'].min()
        max_val_abl = ablation_df['best_accuracy'].max()
        padding_abl = (max_val_abl - min_val_abl) * 0.1 if (max_val_abl - min_val_abl) > 0 else 0.05
        plt.ylim(max(0, min_val_abl - padding_abl), min(1, max_val_abl + padding_abl))

        plot_path = os.path.join(output_dir_plots, 'lp_ablation_study.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved ablation study plot to {plot_path}")

    sensitivity_groups = {
        'Impact_NumProfileSamples': 'num_profile_samples',
        'Impact_OTReg': 'dca_ot_reg',
        'Impact_VWCReg': 'dca_vwc_reg',
        'Impact_NumClusters': 'dca_target_num_clusters'
    }

    for group_name, param_col_name in sensitivity_groups.items():
        group_df = all_results_df[all_results_df['experiment_group'] == group_name].copy()
        if not group_df.empty:
            try:
                group_df[param_col_name] = pd.to_numeric(group_df['param_value'])
                group_df = group_df.sort_values(by=param_col_name)
            except ValueError:
                print(f"Warning: Could not convert param_value to numeric for {group_name}. Skipping plot.")
                continue

            plt.figure(figsize=(10, 6))
            sns.lineplot(x=param_col_name, y='best_accuracy', data=group_df, marker='o', palette="magma")
            plt.title(f'Sensitivity Analysis: {param_col_name} vs. Best Accuracy')
            plt.xlabel(param_col_name.replace('_', ' ').title())
            plt.ylabel('Best Test Accuracy')

            # Dynamic Y-axis for sensitivity plots
            if not group_df['best_accuracy'].empty:
                min_val = group_df['best_accuracy'].min()
                max_val = group_df['best_accuracy'].max()
                padding = (max_val - min_val) * 0.1 if (max_val - min_val) > 0 else 0.05 # Add 10% padding or a small fixed padding
                
                y_lower_bound = max(0, min_val - padding)
                y_upper_bound = min(1, max_val + padding) # Assuming accuracy is <= 1
                
                if y_upper_bound <= y_lower_bound: # Handle cases where all values are the same or padding makes bounds cross
                    y_upper_bound = y_lower_bound + 0.1 # Ensure a minimum range
                
                plt.ylim(y_lower_bound, y_upper_bound)
            else:
                plt.ylim(0, 1) # Default if no data

            # Dynamic X-axis for sensitivity plots
            if pd.api.types.is_numeric_dtype(group_df[param_col_name]):
                unique_param_values = sorted(group_df[param_col_name].unique())
                if len(unique_param_values) > 1:
                    if len(unique_param_values) <= 10: # Show all ticks if not too many
                        plt.xticks(unique_param_values)
                    else: # Otherwise, let Matplotlib decide or use MaxNLocator for fewer ticks
                        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10, prune='both')) 
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir_plots, f'{group_name.lower()}_sensitivity.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved sensitivity plot to {plot_path}")

    training_curves_dir = os.path.join(output_dir_plots, "training_curves")
    os.makedirs(training_curves_dir, exist_ok=True)
    
    plotted_groups = set() # To avoid re-plotting same group graph

    for exp_group in all_results_df['experiment_group'].unique():
        if exp_group in plotted_groups:
            continue

        current_exp_group_df = all_results_df[all_results_df['experiment_group'] == exp_group]
        if current_exp_group_df.empty:
            continue
            
        temp_curves_list = []
        for _, row_main in current_exp_group_df.iterrows():
             if isinstance(row_main['round_accuracies_list'], list) and row_main['round_accuracies_list']: # Check if list is not empty
                # Use param_value as legend label, ensure it's descriptive
                legend_label = str(row_main['param_value'])
                # For ablation, param_value is already descriptive. For sensitivity, it's the value.
                # If run_name is more descriptive for some cases, can use: row_main['run_name']
                
                for r_num, acc_val in row_main['round_accuracies_list']:
                    temp_curves_list.append({'round': r_num, 'accuracy': acc_val, 'variant': legend_label})
        
        if not temp_curves_list: continue
        
        plot_df_for_group = pd.DataFrame(temp_curves_list)
        if plot_df_for_group.empty: continue

        plt.figure(figsize=(12, 7))
        sns.lineplot(data=plot_df_for_group, x='round', y='accuracy', hue='variant', palette="tab10", legend="full")
        plt.title(f'Training Curves: Accuracy vs. Round for {exp_group}')
        plt.xlabel('Global Round')
        plt.ylabel('Test Accuracy')

        # Dynamic Y-axis for training curves
        if not plot_df_for_group['accuracy'].empty:
            min_acc = plot_df_for_group['accuracy'].min()
            max_acc = plot_df_for_group['accuracy'].max()
            padding_acc = (max_acc - min_acc) * 0.05 if (max_acc - min_acc) > 0 else 0.02 # 5% padding or small fixed
            
            y_lower = max(0, min_acc - padding_acc)
            y_upper = min(1, max_acc + padding_acc)

            if y_upper <= y_lower:
                 y_upper = y_lower + 0.1
            plt.ylim(y_lower, y_upper)
        else:
            plt.ylim(0,1)

        # Dynamic X-axis for training curves
        if not plot_df_for_group['round'].empty:
            min_round = plot_df_for_group['round'].min()
            max_round = plot_df_for_group['round'].max()
            # No explicit padding for x-axis usually needed, let sns/matplotlib handle it unless specific issues arise
            # plt.xlim(min_round, max_round) # This can be too tight, usually auto is better
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, nbins='auto', prune='both'))


        plt.legend(title='Variant/Setting', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust rect to make space for legend outside
        
        plot_filename = f"{exp_group.replace(' ', '_').lower()}_training_curves.png"
        plot_path = os.path.join(training_curves_dir, plot_filename)
        
        plt.savefig(plot_path)
        print(f"Saved training curves plot to {plot_path}")
        plt.close()
        plotted_groups.add(exp_group)


def generate_tables(all_results_df, output_dir_tables):
    if all_results_df.empty:
        print("No data to generate tables.")
        return

    os.makedirs(output_dir_tables, exist_ok=True)

    detailed_cols = ['experiment_group', 'run_name', 'param_name', 'param_value', 
                     'best_accuracy', 'final_accuracy', 'data_source', 'filepath', 'wandb_config_summary']
    
    # Ensure all columns exist, fill with defaults if not
    for col in detailed_cols:
        if col not in all_results_df.columns:
            if col in ['best_accuracy', 'final_accuracy']:
                all_results_df[col] = 0.0
            elif col == 'wandb_config_summary':
                 all_results_df[col] = ""
            else:
                all_results_df[col] = None
    
    # Create a summary of wandb_config for the table
    def summarize_config(cfg):
        if not isinstance(cfg, dict) or not cfg:
            return ""
        # Select a few key parameters to show, or convert all to string
        # For now, just a simple string representation
        return str({k: v for k, v in cfg.items() if not k.startswith('_') and isinstance(v, (str, int, float, bool))})

    all_results_df['wandb_config_summary'] = all_results_df['wandb_config'].apply(summarize_config)


    detailed_df_to_save = all_results_df[detailed_cols].copy()
    # Attempt to sort by numeric param_value if possible, else by string
    try:
        detailed_df_to_save['param_value_sort'] = pd.to_numeric(detailed_df_to_save['param_value'])
        detailed_df_to_save.sort_values(by=['experiment_group', 'param_value_sort'], inplace=True)
        detailed_df_to_save.drop(columns=['param_value_sort'], inplace=True)
    except (ValueError, TypeError): # TypeError if param_value is already numeric from parsing
         detailed_df_to_save.sort_values(by=['experiment_group', 'param_value'], inplace=True)

    
    csv_path = os.path.join(output_dir_tables, 'all_experiments_detailed_results.csv')
    md_path = os.path.join(output_dir_tables, 'all_experiments_detailed_results.md')
    detailed_df_to_save.to_csv(csv_path, index=False)
    detailed_df_to_save.to_markdown(md_path, index=False)
    print(f"Saved detailed results table to {csv_path} and {md_path}")

    ablation_summary = all_results_df[all_results_df['experiment_group'] == 'LP_Ablation']
    if not ablation_summary.empty:
        ablation_summary = ablation_summary[['param_value', 'best_accuracy', 'final_accuracy']].rename(
            columns={'param_value': 'LP Setting'}
        ).sort_values(by='best_accuracy', ascending=False)
        csv_path = os.path.join(output_dir_tables, 'summary_lp_ablation.csv')
        md_path = os.path.join(output_dir_tables, 'summary_lp_ablation.md')
        ablation_summary.to_csv(csv_path, index=False)
        ablation_summary.to_markdown(md_path, index=False)
        print(f"Saved LP Ablation summary table to {csv_path} and {md_path}")

    sensitivity_groups = {
        'Impact_NumProfileSamples': 'num_profile_samples',
        'Impact_OTReg': 'dca_ot_reg',
        'Impact_VWCReg': 'dca_vwc_reg',
        'Impact_NumClusters': 'dca_target_num_clusters'
    }
    for group_name, param_col_name_actual in sensitivity_groups.items():
        sensitivity_df = all_results_df[all_results_df['experiment_group'] == group_name]
        if not sensitivity_df.empty:
            # The sort_by_col logic based on 'param_value_numeric' was for the original incorrect sort key.
            # 'param_value' from sensitivity_df (which becomes param_col_name_actual in summary_df)
            # should already have the correct type from parse_experiment_from_filename.
            # Thus, sorting summary_df directly by param_col_name_actual is sufficient.
            # The try-except block for 'param_value_numeric' on sensitivity_df is not directly
            # used for summary_df sorting if we correct the key.

            summary_df = sensitivity_df[['param_value', 'best_accuracy', 'final_accuracy']].rename(
                columns={'param_value': param_col_name_actual}
            ).sort_values(by=param_col_name_actual) # Corrected: Sort by the actual column name in summary_df
            
            csv_path = os.path.join(output_dir_tables, f'summary_{group_name.lower()}.csv')
            md_path = os.path.join(output_dir_tables, f'summary_{group_name.lower()}.md')
            summary_df.to_csv(csv_path, index=False)
            summary_df.to_markdown(md_path, index=False)
            print(f"Saved {group_name} summary table to {csv_path} and {md_path}")


def main():
    print("Starting FedDCA experiment results analysis...")
    if WANDB_ENTITY == "YOUR_WANDB_ENTITY" and wandb:
        print("\nWARNING: WANDB_ENTITY is not set. Please replace 'YOUR_WANDB_ENTITY' in the script with your WandB username or team name if you want to use WandB integration.")
        print("Proceeding with log file parsing only if WandB API cannot be initialized without entity.\n")

    os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)
    output_dir_plots = os.path.join(ANALYSIS_OUTPUT_DIR, "plots")
    output_dir_tables = os.path.join(ANALYSIS_OUTPUT_DIR, "tables")
    os.makedirs(output_dir_plots, exist_ok=True)
    os.makedirs(output_dir_tables, exist_ok=True)

    latest_exp_dir = find_latest_output_directory(base_path=".") 
    if not latest_exp_dir:
        print(f"Error: No experiment output directory found matching pattern '{BASE_DIR_PATTERN}'. Exiting.")
        return
    print(f"Found latest experiment output directory: {latest_exp_dir}")

    out_files = glob.glob(os.path.join(latest_exp_dir, "*.out"))
    if not out_files:
        print(f"Error: No .out files found in {latest_exp_dir}. Exiting.")
        return
    print(f"Found {len(out_files)} .out files to process.")

    results_list = []
    
    api = None
    if wandb and WANDB_ENTITY and WANDB_PROJECT and WANDB_ENTITY != "YOUR_WANDB_ENTITY":
        try:
            print(f"Initializing wandb.Api() for project: {WANDB_ENTITY}/{WANDB_PROJECT}...")
            api = wandb.Api(timeout=20) # Increased timeout
            print("Successfully initialized wandb.Api(). Will attempt to fetch data from WandB.")
        except Exception as e:
            print(f"Failed to initialize wandb.Api(): {e}. Will only use .out files.")
            api = None
    elif wandb:
        print("WandB integration skipped (WANDB_ENTITY or WANDB_PROJECT not configured, or using placeholder).")


    for filepath in out_files:
        filename = os.path.basename(filepath)
        filename_no_ext, _ = os.path.splitext(filename)
        
        print(f"\\nProcessing file: {filename}...")
        
        exp_group, run_specific_name_from_filename, param_name, param_value = parse_experiment_from_filename(filename_no_ext)
        
        round_acc_list, best_acc, final_acc, run_config = [], 0.0, 0.0, {}
        data_source = "logfile" # Default source
        wandb_data_fetched = False

        if api:
            # Try to get run_id from the .out file first
            wandb_run_id_from_log = extract_wandb_run_id_from_log(filepath)
            target_wandb_run_identifier = None

            if wandb_run_id_from_log:
                target_wandb_run_identifier = wandb_run_id_from_log
                print(f"Using WandB run ID from log: {target_wandb_run_identifier}")
            else:
                # Fallback: Construct wandb_run_name from filename as before
                # This constructed name might not always match the actual wandb run name if it was changed in wandb UI
                # or if the script `Experiments.sh` had a different naming logic for `wandb_run_name` than `full_run_name` for the file.
                constructed_wandb_run_name = f"{exp_group}_{run_specific_name_from_filename}"
                # Sanitize run name (Wandb replaces special chars with '-')
                # This sanitization might not be perfect if WandB's internal sanitization is different.
                target_wandb_run_identifier = re.sub(r'[^a-zA-Z0-9_.-]+', '-', constructed_wandb_run_name)
                print(f"WandB run ID not found in log. Using constructed run name from filename: {target_wandb_run_identifier}")

            print(f"Attempting to fetch data from WandB for run: {target_wandb_run_identifier} (Project: {WANDB_ENTITY}/{WANDB_PROJECT})")
            wandb_result = fetch_and_process_wandb_data(
                api, WANDB_PROJECT, WANDB_ENTITY, target_wandb_run_identifier
            )
            if wandb_result:
                round_acc_list, best_acc, final_acc, run_config = wandb_result
                wandb_data_fetched = True
                data_source = "wandb"
            else:
                print(f"Could not fetch data from WandB for run: {target_wandb_run_identifier}. Falling back to .out file.")

        if not wandb_data_fetched:
            print(f"Parsing .out file: {filepath}")
            round_acc_list_log, best_acc_log, final_acc_log = parse_output_file_content(filepath)
            # Only overwrite if log file parsing yields results or if wandb didn't fetch
            if round_acc_list_log or (best_acc_log > 0 or final_acc_log > 0):
                 round_acc_list, best_acc, final_acc = round_acc_list_log, best_acc_log, final_acc_log
                 run_config = {} # No config from .out file directly in this structure
                 data_source = "logfile" # Ensure source is logfile if fallback occurs
                 print(f"Using data from .out file for {filename_no_ext}. Rounds: {len(round_acc_list)}, Best: {best_acc:.4f}")


        if not round_acc_list and best_acc == 0.0 and final_acc == 0.0:
            print(f"Warning: No accuracy data found for {filename_no_ext} (source: {data_source}). Skipping.")
            continue

        results_list.append({
            'filepath': filepath,
            'experiment_group': exp_group,
            'run_name': run_specific_name_from_filename, # This is the specific part of the name from filename
            'param_name': param_name,
            'param_value': param_value,
            'round_accuracies_list': round_acc_list,
            'best_accuracy': best_acc,
            'final_accuracy': final_acc,
            'wandb_config': run_config if run_config else {}, # Ensure it's a dict
            'data_source': data_source
        })

    if not results_list:
        print("No results were successfully parsed from any .out file. Exiting.")
        return

    all_results_df = pd.DataFrame(results_list)
    print(f"\\nSuccessfully parsed data for {len(all_results_df)} experiment runs.")
    
    print("\\nGenerating plots...")
    generate_plots(all_results_df.copy(), output_dir_plots)

    print("\\nGenerating tables...")
    generate_tables(all_results_df.copy(), output_dir_tables)

    print(f"\\nAnalysis complete. Results saved in '{ANALYSIS_OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()
