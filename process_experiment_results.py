#!/usr/bin/env python3
"""
处理实验结果CSV文件，生成方法对比表格
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def extract_method_and_dataset(filename):
    """
    从文件名中提取方法名和数据集名
    例如: exp_FedALA_Cifar10_run0_drift.csv -> (FedALA, Cifar10)
    """
    # 移除文件扩展名
    name = filename.replace('.csv', '')
    
    # 分割文件名
    parts = name.split('_')
    
    # 提取方法名 (通常是第二部分)
    method = parts[1] if len(parts) > 1 else 'Unknown'
    
    # 提取数据集名 (通常是第三部分)
    dataset = parts[2] if len(parts) > 2 else 'Unknown'
    
    return method, dataset

def process_csv_files(wandb_dir):
    """
    处理wandb目录下的所有CSV文件
    """
    wandb_path = Path(wandb_dir)
    
    if not wandb_path.exists():
        print(f"目录不存在: {wandb_dir}")
        return None, None
      # 存储结果的字典
    max_results = {}  # 最大准确率
    avg_results = {}  # 平均准确率 (step 100-120)
    avg_results_har = {}  # HAR数据集平均准确率 (step 250-300)
    
    # 获取所有CSV文件
    csv_files = list(wandb_path.glob("*.csv"))
    
    if not csv_files:
        print(f"在目录 {wandb_dir} 中没有找到CSV文件")
        return None, None
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    for csv_file in csv_files:
        try:
            print(f"处理文件: {csv_file.name}")
            
            # 提取方法名和数据集名
            method, dataset = extract_method_and_dataset(csv_file.name)
            
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            
            # 检查是否包含必要的列
            if 'Global Test Accuracy' not in df.columns:
                print(f"警告: 文件 {csv_file.name} 中没有找到 'Global Test Accuracy' 列")
                continue
            
            if '_step' not in df.columns:
                print(f"警告: 文件 {csv_file.name} 中没有找到 '_step' 列")
                continue
            
            # 获取最优的Global Test Accuracy
            max_accuracy = df['Global Test Accuracy'].max()
              # 计算step 100-120的平均准确率
            step_range_df = df[(df['_step'] >= 100) & (df['_step'] <= 120)]
            if not step_range_df.empty:
                avg_accuracy_100_120 = step_range_df['Global Test Accuracy'].mean()
            else:
                avg_accuracy_100_120 = np.nan
                print(f"  警告: 文件 {csv_file.name} 中没有找到step 100-120的数据")
            
            # 计算HAR数据集step 250-300的平均准确率
            if dataset.lower() == 'har':
                step_range_df_har = df[(df['_step'] >= 250) & (df['_step'] <= 300)]
                if not step_range_df_har.empty:
                    avg_accuracy_250_300 = step_range_df_har['Global Test Accuracy'].mean()
                else:
                    avg_accuracy_250_300 = np.nan
                    print(f"  警告: 文件 {csv_file.name} 中没有找到step 250-300的数据")
            else:
                avg_accuracy_250_300 = np.nan
            
            # 存储最大准确率结果
            if method not in max_results:
                max_results[method] = {}
            max_results[method][dataset] = max_accuracy
              # 存储平均准确率结果 (仅对指定数据集)
            if dataset.lower() in ['cifar100', 'cifar10', 'fmnist']:
                if method not in avg_results:
                    avg_results[method] = {}
                avg_results[method][dataset] = avg_accuracy_100_120
            
            # 存储HAR数据集的平均准确率结果
            if dataset.lower() == 'har':
                if method not in avg_results_har:
                    avg_results_har[method] = {}
                avg_results_har[method][dataset] = avg_accuracy_250_300
            print(f"  方法: {method}, 数据集: {dataset}")
            print(f"    最佳准确率: {max_accuracy:.4f}")
            if dataset.lower() in ['cifar100', 'cifar10', 'fmnist']:
                if not np.isnan(avg_accuracy_100_120):
                    print(f"    Step 100-120平均准确率: {avg_accuracy_100_120:.4f}")
                else:
                    print(f"    Step 100-120平均准确率: 无数据")
            
            if dataset.lower() == 'har':
                if not np.isnan(avg_accuracy_250_300):
                    print(f"    Step 250-300平均准确率: {avg_accuracy_250_300:.4f}")
                else:
                    print(f"    Step 250-300平均准确率: 无数据")
            
        except Exception as e:
            print(f"处理文件 {csv_file.name} 时出错: {e}")
            continue
    
    return max_results, avg_results, avg_results_har

def create_results_table(results):
    """
    创建结果表格
    """
    if not results:
        print("没有可用的结果数据")
        return None
    
    # 获取所有方法和数据集
    all_methods = sorted(results.keys())
    all_datasets = set()
    
    for method_results in results.values():
        all_datasets.update(method_results.keys())
    
    all_datasets = sorted(list(all_datasets))
    
    # 创建DataFrame
    table_data = []
    
    for method in all_methods:
        row = [method]  # 第一列是方法名
        for dataset in all_datasets:
            accuracy = results[method].get(dataset, np.nan)
            row.append(accuracy)
        table_data.append(row)
    
    # 创建DataFrame
    columns = ['Method'] + all_datasets
    df_table = pd.DataFrame(table_data, columns=columns)
    
    return df_table

def save_results(df_table_max, df_table_avg, df_table_har, output_dir):
    """
    保存结果到文件
    """
    if df_table_max is None and df_table_avg is None and df_table_har is None:
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    files_saved = []
    
    # 保存最大准确率表格
    if df_table_max is not None:
        # 保存为CSV文件
        csv_output_max = output_path / "experiment_results_max_accuracy.csv"
        df_table_max.to_csv(csv_output_max, index=False, float_format='%.4f')
        print(f"最大准确率结果已保存到: {csv_output_max}")
        files_saved.append(csv_output_max)
        
        # 保存为Excel文件
        excel_output_max = output_path / "experiment_results_max_accuracy.xlsx"
        df_table_max.to_excel(excel_output_max, index=False, float_format='%.4f')
        print(f"最大准确率结果已保存到: {excel_output_max}")
        files_saved.append(excel_output_max)
    
    # 保存平均准确率表格 (Cifar100, Cifar10, FMNIST)
    if df_table_avg is not None:
        # 保存为CSV文件
        csv_output_avg = output_path / "experiment_results_avg_accuracy_step100_120.csv"
        df_table_avg.to_csv(csv_output_avg, index=False, float_format='%.4f')
        print(f"Step 100-120平均准确率结果已保存到: {csv_output_avg}")
        files_saved.append(csv_output_avg)
        
        # 保存为Excel文件
        excel_output_avg = output_path / "experiment_results_avg_accuracy_step100_120.xlsx"
        df_table_avg.to_excel(excel_output_avg, index=False, float_format='%.4f')
        print(f"Step 100-120平均准确率结果已保存到: {excel_output_avg}")
        files_saved.append(excel_output_avg)
    
    # 保存HAR数据集平均准确率表格
    if df_table_har is not None:
        # 保存为CSV文件
        csv_output_har = output_path / "experiment_results_avg_accuracy_har_step250_300.csv"
        df_table_har.to_csv(csv_output_har, index=False, float_format='%.4f')
        print(f"HAR Step 250-300平均准确率结果已保存到: {csv_output_har}")
        files_saved.append(csv_output_har)
        
        # 保存为Excel文件
        excel_output_har = output_path / "experiment_results_avg_accuracy_har_step250_300.xlsx"
        df_table_har.to_excel(excel_output_har, index=False, float_format='%.4f')
        print(f"HAR Step 250-300平均准确率结果已保存到: {excel_output_har}")
        files_saved.append(excel_output_har)
    
    return files_saved

def print_table(df_table, title):
    """
    打印格式化的表格
    """
    if df_table is None:
        return
    
    print("\n" + "="*80)
    print(title)
    print("="*80)
    
    # 设置pandas显示选项
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    print(df_table.to_string(index=False))
    print("="*80)

def main():
    """
    主函数
    """
    # 设置路径
    wandb_dir = r"d:\repos\PFL-Non-IID\results\wandb"
    output_dir = r"d:\repos\PFL-Non-IID\results"
    
    print(f"开始处理目录: {wandb_dir}")
      # 处理CSV文件
    max_results, avg_results, avg_results_har = process_csv_files(wandb_dir)
    
    if not max_results and not avg_results and not avg_results_har:
        print("没有成功处理任何文件")
        return
    
    # 创建结果表格
    df_table_max = None
    df_table_avg = None
    df_table_har = None
    
    if max_results:
        df_table_max = create_results_table(max_results)
        print_table(df_table_max, "实验结果汇总表格 - 最大准确率")
    
    if avg_results:
        df_table_avg = create_results_table(avg_results)
        print_table(df_table_avg, "实验结果汇总表格 - Step 100-120平均准确率 (Cifar100, Cifar10, FMNIST)")
    
    if avg_results_har:
        df_table_har = create_results_table(avg_results_har)
        print_table(df_table_har, "实验结果汇总表格 - Step 250-300平均准确率 (HAR)")
    
    # 保存结果
    saved_files = save_results(df_table_max, df_table_avg, df_table_har, output_dir)
    
    print(f"\n处理完成!")
    
    if max_results:
        print(f"最大准确率统计: 共处理了 {len(max_results)} 种方法")
        
        # 显示数据集和方法统计
        all_datasets = set()
        for method_results in max_results.values():
            all_datasets.update(method_results.keys())
        
        print(f"数据集: {sorted(list(all_datasets))}")
        print(f"方法: {sorted(list(max_results.keys()))}")
    if avg_results:
        print(f"\nStep 100-120平均准确率统计: 共处理了 {len(avg_results)} 种方法")
        
        # 显示数据集和方法统计
        all_datasets_avg = set()
        for method_results in avg_results.values():
            all_datasets_avg.update(method_results.keys())
        
        print(f"数据集 (仅Cifar100, Cifar10, FMNIST): {sorted(list(all_datasets_avg))}")
        print(f"方法: {sorted(list(avg_results.keys()))}")
    
    if avg_results_har:
        print(f"\nHAR Step 250-300平均准确率统计: 共处理了 {len(avg_results_har)} 种方法")
        
        # 显示数据集和方法统计
        all_datasets_har = set()
        for method_results in avg_results_har.values():
            all_datasets_har.update(method_results.keys())
        
        print(f"数据集 (仅HAR): {sorted(list(all_datasets_har))}")
        print(f"方法: {sorted(list(avg_results_har.keys()))}")
    
    print(f"\n保存的文件:")
    for file_path in saved_files:
        print(f"  {file_path}")

if __name__ == "__main__":
    main()
