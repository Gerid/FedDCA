import wandb
import pandas as pd
import os

# --- 用户配置 ---
# TODO: 请务必修改以下配置项
WANDB_ENTITY = "Gerid"  # 替换为您的 W&B实体 (用户名或团队名)
WANDB_PROJECT = "FedDCA_Baseline_Comparisions"  # 替换为您的 W&B 项目名称

# 定义需要提取的指标及其获取 "最佳" 值的方式
# 例如: "accuracy": "max" 表示提取名为 'accuracy' 的指标，并取其最大值
# 如果您的 W&B 日志中直接记录了最佳值 (例如 'best_accuracy'),
# 可以直接使用该指标名，并将获取方式设为 "direct" 或省略 (默认从 summary 获取)
METRICS_TO_EXTRACT = {
    "Local_Test_Accuracy": "max",  # 假设 'accuracy' 是逐步记录的，并且您想要最大值
    "Local_Test_AUC": "max",       # 假设 'auc' 是逐步记录的，并且您想要最大值
    "Local_Weighted_TPR": "max",       # 假设 'tpr' 是逐步记录的，并且您想要最大值
    # "val_loss": "min", # 示例：提取验证损失的最小值
    # "epoch_accuracy": "direct" # 示例：如果 'epoch_accuracy' 在 summary 中就是最终/最佳值
}

# 定义如何从 W&B 运行的 config 中识别数据集和方法等信息
# 示例: 如果您的 run.config 包含 {'dataset_name': 'cifar10', 'algorithm': 'fedavg'}
CONFIG_FIELDS = {
    "dataset": "dataset",  # run.config 中对应数据集名称的键
    "method": "algorithm",     # run.config 中对应方法/算法名称的键
    # "learning_rate": "lr"    # 示例：提取学习率
}

OUTPUT_CSV_FILE = "experiment_results_from_wandb.csv" # 输出CSV文件名
# --- 用户配置结束 ---

def fetch_wandb_data(entity, project, metrics_map, config_fields_map):
    """
    从 W&B 获取实验数据并进行处理。
    """
    try:
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}")
    except Exception as e:
        print(f"无法连接到 W&B 或获取项目运行列表: {e}")
        print("请检查您的 W&B 实体和项目名称是否正确，以及网络连接和 W&B API 密钥。")
        return

    all_run_data = []
    
    print(f"在项目 '{entity}/{project}' 中找到 {len(runs)} 个运行。正在处理...")
    
    for i, run in enumerate(runs):
        print(f"正在处理运行 {i+1}/{len(runs)}: {run.name} (ID: {run.id})...")
        run_data = {}
        
        # 提取配置字段 (数据集, 方法等)
        for display_key, config_key_in_wandb in config_fields_map.items():
            run_data[display_key] = run.config.get(config_key_in_wandb, None)
            
        # 提取运行名称和ID
        run_data["run_name"] = run.name
        run_data["run_id"] = run.id
        
        # 提取指标
        for metric_name_in_wandb, aggregation_type in metrics_map.items():
            target_metric_key = f"best_{metric_name_in_wandb}" # CSV中的列名
            metric_value = float('nan') # 默认为 NaN

            # 优先尝试从 run.summary 获取 (通常包含最终值或预定义的最佳值)
            if metric_name_in_wandb in run.summary:
                summary_value = run.summary.get(metric_name_in_wandb)
                # 如果聚合类型是 'direct' 或 summary 值本身就是数字，则直接使用
                if aggregation_type == "direct" or isinstance(summary_value, (int, float)):
                    metric_value = summary_value
                # 如果 summary 中存的是字典 (例如 { 'max': value, 'min': value }), 则尝试提取
                elif isinstance(summary_value, dict) and aggregation_type in summary_value:
                     metric_value = summary_value.get(aggregation_type)

            # 如果 summary 中没有直接获取到，或者需要从 history 中计算 max/min
            if pd.isna(metric_value) and aggregation_type in ["max", "min"]:
                try:
                    # 获取指标历史记录 (限制样本数以提高效率)
                    history_df = run.history(keys=[metric_name_in_wandb], pandas=True, samples=10000)
                    if not history_df.empty and metric_name_in_wandb in history_df:
                        clean_metric_values = history_df[metric_name_in_wandb].dropna()
                        if not clean_metric_values.empty:
                            if aggregation_type == "max":
                                metric_value = clean_metric_values.max()
                            elif aggregation_type == "min":
                                metric_value = clean_metric_values.min()
                        else:
                            print(f"  指标 '{metric_name_in_wandb}' 在运行 {run.name} 的历史记录中没有有效值。")
                    else:
                        print(f"  指标 '{metric_name_in_wandb}' 在运行 {run.name} 的历史记录中未找到或历史记录为空。")
                except Exception as e:
                    print(f"  获取运行 {run.name} 的指标 '{metric_name_in_wandb}' 历史记录时出错: {e}")
            
            run_data[target_metric_key] = metric_value
            if pd.isna(metric_value) and metric_name_in_wandb not in run.summary and aggregation_type not in ["max", "min"]:
                 print(f"  警告: 指标 '{metric_name_in_wandb}' 未在运行 {run.name} 的 summary 中找到，且未指定 'max' 或 'min' 聚合方式。")


        all_run_data.append(run_data)
        
    if not all_run_data:
        print("没有处理任何数据。退出。")
        return

    df = pd.DataFrame(all_run_data)
    
    # 保存到 CSV
    try:
        df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')
        print(f"\n成功将数据保存到 {OUTPUT_CSV_FILE}")
    except Exception as e:
        print(f"保存CSV文件时出错: {e}")
        
    print("\nCSV文件的前5行示例:")
    print(df.head().to_string())
    
    print(f"\n要分析数据，您可以使用 pandas 加载 '{OUTPUT_CSV_FILE}'。")
    print("示例代码:")
    print("import pandas as pd")
    print(f"df = pd.read_csv('{OUTPUT_CSV_FILE}')")
    print("\n# 获取 'cifar10' 数据集上所有方法的最佳准确率")
    print("# df_cifar10 = df[df['dataset'] == 'cifar10']")
    print("# print(df_cifar10.groupby('method')['best_accuracy'].max())")
    print("\n# 获取特定方法 'fedavg' 在所有数据集上的最佳AUC")
    print("# df_fedavg = df[df['method'] == 'fedavg']")
    print("# print(df_fedavg.groupby('dataset')['best_auc'].max())")

if __name__ == "__main__":
    # 检查 WANDB_API_KEY 是否已设置，或者 wandb login 是否已运行
    if not os.getenv("WANDB_API_KEY") and not wandb.api.api_key:
        print("未在环境变量中找到 WANDB_API_KEY。")
        print("请设置该环境变量，或在终端运行 'wandb login' 进行登录。")
        # 或者，可以取消下一行的注释以在脚本运行时提示登录：
        # try:
        #     wandb.login()
        # except Exception as e:
        #     print(f"W&B login failed: {e}")
        #     exit(1)
        # 对于非交互式环境，必须预先设置 API 密钥。
        if not wandb.api.api_key: # 再次检查，因为login()可能失败或被跳过
             exit(1)

    if WANDB_ENTITY == "your_wandb_entity" or WANDB_PROJECT == "your_wandb_project":
        print("请在脚本中更新 WANDB_ENTITY 和 WANDB_PROJECT 的值。")
        exit(1)
    
    if not METRICS_TO_EXTRACT:
        print("请在脚本中配置 METRICS_TO_EXTRACT 字典，指定要下载的指标。")
        exit(1)
        
    fetch_wandb_data(WANDB_ENTITY, WANDB_PROJECT, METRICS_TO_EXTRACT, CONFIG_FIELDS)
