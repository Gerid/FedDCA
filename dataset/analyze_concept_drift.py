import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict
from matplotlib import font_manager as fm, rcParams
import matplotlib

# 配置中文字体支持
def setup_chinese_font():
    """配置中文字体支持"""
    # 尝试设置微软雅黑字体（Windows系统常见字体）
    try:
        # 检查系统中文字体
        chinese_fonts = [f for f in fm.findSystemFonts() if os.path.basename(f).startswith(('msyh', 'simhei', 'simsun', 'simkai', 'Microsoft YaHei'))]
        
        if chinese_fonts:
            # 优先使用微软雅黑
            msyh = [f for f in chinese_fonts if 'msyh' in f.lower() or 'microsoft yahei' in f.lower()]
            if msyh:
                matplotlib.rcParams['font.family'] = ['sans-serif']
                matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei'] + rcParams['font.sans-serif']
            else:
                # 使用找到的第一个中文字体
                font_path = chinese_fonts[0]
                font_prop = fm.FontProperties(fname=font_path)
                matplotlib.rcParams['font.family'] = font_prop.get_name()
            
            # 修复负号显示问题
            matplotlib.rcParams['axes.unicode_minus'] = False
            print("已配置中文字体支持")
        else:
            print("警告: 未找到中文字体，图表中的中文可能无法正确显示")
    except Exception as e:
        print(f"配置中文字体时出错: {e}")
        print("图表中的中文可能无法正确显示")

def load_concept_config(data_dir="Cifar100_clustered/"):
    """
    加载概念漂移配置文件
    
    参数:
        data_dir: 数据存储路径，默认为"Cifar100_clustered/"
        
    返回:
        concept_config: 包含概念配置信息的字典
    """
    config_path = os.path.join(data_dir, "drift_info", "concept_config.json")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        concept_config = json.load(f)
    
    print(f"成功加载了 {data_dir} 的概念配置")
    print(f"共有 {concept_config['num_concepts']} 个概念")
    print(f"共有 {len(concept_config['client_concepts'])} 个客户端")
    print(f"漂移点设置在: {concept_config['drift_iterations']}")
    
    return concept_config

def get_client_concept_at_iteration(concept_config, client_id, iteration):
    """
    获取指定客户端在特定迭代点的概念
    
    参数:
        concept_config: 概念配置字典
        client_id: 客户端ID
        iteration: 迭代点
        
    返回:
        concept_id: 该客户端在此迭代点的概念ID
    """
    # 确保client_id是字符串
    client_id_str = str(client_id)
    
    # 如果在配置中存在客户端的轨迹
    if 'client_concept_trajectories' in concept_config and client_id_str in concept_config['client_concept_trajectories']:
        # 获取客户端的概念轨迹
        trajectory = concept_config['client_concept_trajectories'][client_id_str]
        # 确保迭代点在轨迹范围内
        if 0 <= iteration < len(trajectory):
            return trajectory[iteration]
    
    # 无法确定概念时返回-1
    return -1

def get_all_client_concepts_at_iteration(concept_config, iteration):
    """
    获取所有客户端在特定迭代点的概念分配
    
    参数:
        concept_config: 概念配置字典
        iteration: 迭代点
        
    返回:
        client_concepts: 字典，将客户端ID映射到概念ID
    """
    client_concepts = {}
    
    # 检查客户端轨迹是否存在
    if 'client_concept_trajectories' not in concept_config:
        print("警告: 配置中未找到客户端概念轨迹")
        return client_concepts
    
    # 获取所有客户端在指定迭代的概念
    for client_id, trajectory in concept_config['client_concept_trajectories'].items():
        if 0 <= iteration < len(trajectory):
            client_concepts[client_id] = trajectory[iteration]
        else:
            print(f"警告: 客户端 {client_id} 没有在迭代 {iteration} 的概念记录")
    
    return client_concepts

def generate_concept_evolution_report(concept_config, data_dir="Cifar100_clustered/", save_plots=True):
    """
    生成概念演变的完整报告
    
    参数:
        concept_config: 概念配置字典
        data_dir: 数据存储路径
        save_plots: 是否保存图表
    """
    # 确保存在保存路径
    output_dir = os.path.join(data_dir, "concept_analysis")
    if save_plots and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取基本信息
    num_concepts = concept_config['num_concepts']
    num_clients = len(concept_config['client_concepts'])
    drift_iterations = concept_config['drift_iterations']
    
    # 创建用于追踪的数据结构
    concept_distribution = defaultdict(lambda: defaultdict(int))  # {iteration: {concept_id: count}}
    client_concept_history = defaultdict(list)  # {client_id: [concept_id_at_iter0, ...]}
    concept_changes = []  # 记录概念变化点
    
    # 确定总迭代次数
    if 'client_concept_trajectories' in concept_config:
        first_client = list(concept_config['client_concept_trajectories'].keys())[0]
        num_iterations = len(concept_config['client_concept_trajectories'][first_client])
    else:
        print("警告: 无法确定总迭代次数")
        return
    
    # 收集每个迭代点的概念分布和变化
    for iteration in range(num_iterations):
        # 获取当前迭代的所有客户端概念
        iteration_concepts = get_all_client_concepts_at_iteration(concept_config, iteration)
        
        # 更新分布统计
        for client_id, concept_id in iteration_concepts.items():
            concept_distribution[iteration][concept_id] += 1
            client_concept_history[client_id].append(concept_id)
        
        # 如果不是第一个迭代，检查概念变化
        if iteration > 0:
            changes = sum(1 for client_id in iteration_concepts 
                        if client_concept_history[client_id][iteration] != client_concept_history[client_id][iteration-1])
            change_percentage = changes / num_clients * 100
            concept_changes.append((iteration, change_percentage))
    
    # ======================= 生成报告 =======================
    
    # 1. 基本信息报告
    print("\n========== 概念漂移分析报告 ==========")
    print(f"数据集路径: {data_dir}")
    print(f"概念数量: {num_concepts}")
    print(f"客户端数量: {num_clients}")
    print(f"总迭代次数: {num_iterations}")
    print(f"预设漂移点: {drift_iterations}")
    
    # 2. 漂移类型分布
    if 'client_drift_types' in concept_config:
        drift_type_counts = defaultdict(int)
        for client_id, drift_type in concept_config['client_drift_types'].items():
            drift_type_counts[drift_type] += 1
        
        print("\n漂移类型分布:")
        for drift_type, count in drift_type_counts.items():
            print(f"  {drift_type}: {count} 客户端 ({count/num_clients*100:.1f}%)")
    
    # 3. 概念分配情况
    if 'client_concepts' in concept_config:
        concept_client_map = defaultdict(list)
        for client_id, concepts in concept_config['client_concepts'].items():
            for concept in concepts:
                concept_client_map[concept].append(client_id)
        
        print("\n每个概念的客户端分配:")
        for concept, clients in concept_client_map.items():
            print(f"  概念 {concept}: {len(clients)} 客户端")
            print(f"    客户端: {', '.join(map(str, clients[:10]))}" + ("..." if len(clients) > 10 else ""))
    
    # 4. 概念变化统计
    print("\n概念变化统计:")
    # 显示变化最大的几个迭代点
    sorted_changes = sorted(concept_changes, key=lambda x: x[1], reverse=True)
    for iteration, change_pct in sorted_changes[:5]:
        print(f"  迭代 {iteration}: 变化率 {change_pct:.1f}%")
    
    # 检查预设漂移点是否确实引起了大量变化
    drift_detected = []
    for drift_iter in drift_iterations:
        # 找到最接近的实际迭代点（因为drift_iterations可能是小数）
        closest_iter = min(range(num_iterations), key=lambda i: abs(i - drift_iter))
        # 检查前后几轮的变化率
        surrounding_changes = [change for iter, change in concept_changes 
                              if abs(iter - closest_iter) <= 3]
        max_change = max(surrounding_changes) if surrounding_changes else 0
        drift_detected.append((drift_iter, closest_iter, max_change))
    
    print("\n预设漂移点检测:")
    for drift_iter, closest_iter, max_change in drift_detected:
        detection = "强烈" if max_change > 50 else "明显" if max_change > 30 else "较弱" if max_change > 10 else "不明显"
        print(f"  预设点 {drift_iter} (最近迭代: {closest_iter}): 最大变化率 {max_change:.1f}% - {detection}")
      # ======================= 生成可视化 =======================
    
    if save_plots:
        # 确保中文字体配置已应用
        setup_chinese_font()
        
        # 1. 概念分布随时间变化图
        plt.figure(figsize=(15, 7))
        concept_evolution = {concept_id: [] for concept_id in range(num_concepts)}
        
        for iteration in range(num_iterations):
            distribution = concept_distribution[iteration]
            for concept_id in range(num_concepts):
                concept_evolution[concept_id].append(distribution.get(concept_id, 0) / num_clients * 100)
        
        for concept_id, percentages in concept_evolution.items():
            plt.plot(range(num_iterations), percentages, label=f'概念 {concept_id}')
        
        # 标记预设漂移点
        for drift_iter in drift_iterations:
            plt.axvline(x=drift_iter, color='r', linestyle='--', alpha=0.5)
            plt.text(drift_iter, 5, f'漂移点', rotation=90, alpha=0.7)
        
        plt.title('客户端概念分布随时间变化')
        plt.xlabel('迭代')
        plt.ylabel('客户端百分比 (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'concept_distribution_evolution.png'), dpi=300)
        plt.close()
        
        # 2. 概念变化率图
        plt.figure(figsize=(15, 5))
        iterations, change_rates = zip(*concept_changes)
        plt.plot(iterations, change_rates, 'b-', marker='o', markersize=3)
        
        # 标记预设漂移点
        for drift_iter in drift_iterations:
            plt.axvline(x=drift_iter, color='r', linestyle='--', alpha=0.5)
            plt.text(drift_iter, max(change_rates)/2, f'漂移点', rotation=90, alpha=0.7)
        
        plt.title('客户端概念变化率')
        plt.xlabel('迭代')
        plt.ylabel('变化率 (%)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'concept_change_rate.png'), dpi=300)
        plt.close()
        
        # 3. 客户端漂移轨迹热力图
        # 选择前20个客户端进行可视化
        sample_clients = list(client_concept_history.keys())[:20]
        
        # 创建数据框
        trajectory_data = []
        for client_id in sample_clients:
            for iteration, concept_id in enumerate(client_concept_history[client_id]):
                trajectory_data.append({
                    'Client': f'客户端 {client_id}',
                    'Iteration': iteration,
                    'Concept': concept_id
                })
        
        trajectory_df = pd.DataFrame(trajectory_data)
        
        plt.figure(figsize=(15, 10))
        pivot_table = trajectory_df.pivot(index='Client', columns='Iteration', values='Concept')
        sns.heatmap(pivot_table, cmap='tab10', cbar_kws={'label': '概念 ID'})
        
        # 标记预设漂移点
        for drift_iter in drift_iterations:
            if drift_iter < num_iterations:
                plt.axvline(x=drift_iter, color='r', linestyle='--', alpha=0.5)
        
        plt.title('客户端概念漂移轨迹')
        plt.xlabel('迭代')
        plt.ylabel('客户端')
        plt.savefig(os.path.join(output_dir, 'client_concept_trajectories.png'), dpi=300)
        plt.close()
        
        # 4. 为每个漂移类型生成典型的客户端轨迹图
        if 'client_drift_types' in concept_config:
            # 收集每种漂移类型的代表性客户端
            drift_type_clients = defaultdict(list)
            for client_id, drift_type in concept_config['client_drift_types'].items():
                drift_type_clients[drift_type].append(client_id)
            
            plt.figure(figsize=(15, 10))
            
            for i, drift_type in enumerate(['sudden', 'gradual', 'recurring']):
                if drift_type in drift_type_clients and drift_type_clients[drift_type]:
                    plt.subplot(3, 1, i+1)
                    
                    # 为每种类型选择3个客户端
                    sample_clients = drift_type_clients[drift_type][:3]
                    for client_id in sample_clients:
                        if client_id in client_concept_history:
                            plt.plot(range(num_iterations), client_concept_history[client_id], 
                                    label=f'客户端 {client_id}')
                    
                    # 标记预设漂移点
                    for drift_iter in drift_iterations:
                        plt.axvline(x=drift_iter, color='r', linestyle='--', alpha=0.5)
                    
                    plt.title(f'{drift_type} 漂移类型的客户端轨迹')
                    plt.xlabel('迭代')
                    plt.ylabel('概念 ID')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'drift_type_trajectories.png'), dpi=300)
            plt.close()
    
    # 5. 生成特定迭代点的客户端概念分布
    if save_plots:
        # 选择关键迭代点进行可视化
        key_iterations = list(drift_iterations)
        key_iterations.append(0)  # 添加初始迭代
        key_iterations.append(num_iterations-1)  # 添加最终迭代
        key_iterations = sorted(set([int(i) for i in key_iterations if 0 <= i < num_iterations]))
        
        # 为每个关键迭代点创建分布可视化
        for iteration in key_iterations:
            plt.figure(figsize=(10, 6))
            
            # 获取该迭代的概念分布
            concept_counts = defaultdict(int)
            for client_id in client_concept_history:
                if iteration < len(client_concept_history[client_id]):
                    concept_id = client_concept_history[client_id][iteration]
                    concept_counts[concept_id] += 1
            
            # 绘制饼图
            concepts = sorted(concept_counts.keys())
            counts = [concept_counts[c] for c in concepts]
            
            plt.pie(counts, labels=[f'概念 {c}' for c in concepts], autopct='%1.1f%%', 
                   startangle=90, shadow=True, explode=[0.05] * len(concepts))
            plt.title(f'迭代 {iteration} 的客户端概念分布')
            plt.savefig(os.path.join(output_dir, f'concept_distribution_iteration_{iteration}.png'), dpi=300)
            plt.close()
    
    print(f"\n分析报告和可视化已保存到: {output_dir}")
    
    # 6. 保存分析数据
    if save_plots:
        # 将概念变化数据保存为CSV
        changes_df = pd.DataFrame(concept_changes, columns=['Iteration', 'Change Rate (%)'])
        changes_df.to_csv(os.path.join(output_dir, 'concept_changes.csv'), index=False)
        
        # 保存客户端轨迹数据
        trajectory_df = pd.DataFrame({
            client_id: trajectory 
            for client_id, trajectory in client_concept_history.items()
        })
        trajectory_df.to_csv(os.path.join(output_dir, 'client_trajectories.csv'))
        
        print(f"数据文件已保存到: {output_dir}")

def export_client_concepts_to_csv(concept_config, output_path, data_dir="Cifar100_clustered/"):
    """
    将所有客户端在所有迭代中的概念分配导出为CSV文件
    
    参数:
        concept_config: 概念配置字典
        output_path: 输出CSV文件路径
        data_dir: 数据存储路径
    """
    # 确保存在客户端轨迹信息
    if 'client_concept_trajectories' not in concept_config:
        print("错误: 配置中未找到客户端概念轨迹")
        return
    
    # 准备数据
    client_concept_history = {}
    for client_id, trajectory in concept_config['client_concept_trajectories'].items():
        client_concept_history[f"客户端_{client_id}"] = trajectory
    
    # 创建数据框
    trajectory_df = pd.DataFrame(client_concept_history)
    
    # 添加漂移点标记
    if 'drift_iterations' in concept_config:
        drift_iterations = concept_config['drift_iterations']
        is_drift = []
        
        for i in range(len(trajectory_df)):
            is_drift.append("是" if i in drift_iterations else "否")
        
        trajectory_df.insert(0, "是否漂移点", is_drift)
    
    # 导出CSV
    trajectory_df.to_csv(output_path, index_label="迭代")
    print(f"客户端概念分配已导出到: {output_path}")

def analyze_concept_preferences(concept_config):
    """
    分析每个概念的类别偏好
    
    参数:
        concept_config: 概念配置字典
    """
    if 'concept_distributions' not in concept_config:
        print("错误: 配置中未找到概念分布信息")
        return
    
    print("\n======== 概念类别偏好分析 ========")
    
    for concept_id, info in concept_config['concept_distributions'].items():
        preferred_classes = info.get('preferred_classes', [])
        class_weights = info.get('class_weights', [])
        
        print(f"\n概念 {concept_id}:")
        print(f"  偏好的类别数量: {len(preferred_classes)}")
        
        # 显示权重最高的10个类别
        if preferred_classes and class_weights:
            # 转换为numpy数组以便处理
            class_weights_np = np.array(class_weights)
            # 获取权重最高的类别索引
            top_indices = np.argsort(class_weights_np)[-10:][::-1]
            
            print("  权重最高的10个类别:")
            for idx in top_indices:
                print(f"    类别 {idx}: 权重 {class_weights_np[idx]:.4f}")

if __name__ == "__main__":
    # 设置数据路径
    data_dir = "Cifar100_clustered/"
    
    # 配置中文字体支持
    setup_chinese_font()
    
    # 加载概念配置
    concept_config = load_concept_config(data_dir)
    
    # 分析概念演变
    generate_concept_evolution_report(concept_config, data_dir, save_plots=True)
    
    # 导出CSV
    csv_output_path = os.path.join(data_dir, "concept_analysis", "all_client_concepts.csv")
    export_client_concepts_to_csv(concept_config, csv_output_path, data_dir)
    
    # 分析概念偏好
    analyze_concept_preferences(concept_config)
    
    print("\n分析完成!")
