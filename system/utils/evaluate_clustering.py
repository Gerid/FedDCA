import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from matplotlib import font_manager as fm, rcParams
import matplotlib
import argparse
import sys

# 设置中文字体
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

def load_cluster_assignments(clustering_file):
    """
    加载FedDCA的聚类分配结果
    
    参数:
        clustering_file: 聚类结果文件路径
        
    返回:
        字典，客户端ID映射到集群ID
    """
    if not os.path.exists(clustering_file):
        print(f"错误: 找不到聚类文件: {clustering_file}")
        return {}
    
    try:
        with open(clustering_file, 'r', encoding='utf-8') as f:
            cluster_assignments = json.load(f)
        
        print(f"成功加载聚类结果，共 {len(cluster_assignments)} 个客户端")
        return cluster_assignments
    except Exception as e:
        print(f"加载聚类文件时出错: {e}")
        return {}

def load_true_concepts(drift_data_dir, iteration):
    """
    加载指定迭代中的真实概念分配
    
    参数:
        drift_data_dir: 数据集路径
        iteration: 迭代编号
        
    返回:
        字典，客户端ID映射到真实概念ID
    """
    config_path = os.path.join(drift_data_dir, "drift_info", "concept_config.json")
    
    if not os.path.exists(config_path):
        print(f"错误: 找不到配置文件: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            concept_config = json.load(f)
        
        client_concepts = {}
        
        # 检查是否存在客户端轨迹
        if 'client_concept_trajectories' in concept_config:
            # 获取每个客户端在指定迭代的概念
            for client_id, trajectory in concept_config['client_concept_trajectories'].items():
                if 0 <= iteration < len(trajectory):
                    client_concepts[client_id] = trajectory[iteration]
                else:
                    print(f"警告: 客户端 {client_id} 没有在迭代 {iteration} 的概念记录")
        else:
            print("警告: 配置中未找到客户端概念轨迹")
        
        print(f"成功加载迭代 {iteration} 的概念分配，共 {len(client_concepts)} 个客户端")
        return client_concepts
    except Exception as e:
        print(f"加载概念配置时出错: {e}")
        return {}

def evaluate_clustering(cluster_assignments, true_concepts):
    """
    评估聚类结果与真实概念的一致性
    
    参数:
        cluster_assignments: 字典，将客户端ID映射到算法分配的聚类
        true_concepts: 字典，将客户端ID映射到真实概念
        
    返回:
        dict: 包含各种聚类评估指标的字典
    """
    # 确保两个字典的键集合相同
    common_clients = set(cluster_assignments.keys()) & set(true_concepts.keys())
    
    if not common_clients:
        print("错误: 没有可比较的共同客户端")
        return {
            "error": "没有可比较的共同客户端",
            "num_clusters": 0,
            "num_concepts": 0
        }
    
    # 提取共同客户端的标签
    cluster_labels = [cluster_assignments[client_id] for client_id in common_clients]
    concept_labels = [true_concepts[client_id] for client_id in common_clients]
    
    # 计算调整兰德指数(ARI)
    ari = adjusted_rand_score(concept_labels, cluster_labels)
    
    # 计算标准化互信息(NMI)
    nmi = normalized_mutual_info_score(concept_labels, cluster_labels)
    
    # 计算聚类纯度(Purity)
    contingency_matrix = np.zeros((max(cluster_labels) + 1, max(concept_labels) + 1))
    for i, j in zip(cluster_labels, concept_labels):
        contingency_matrix[i, j] += 1
    
    # 每个聚类中最主要的概念数量
    cluster_purity = np.sum(np.max(contingency_matrix, axis=1)) / len(cluster_labels)
    
    # 计算每个聚类中来自不同概念的客户端分布
    cluster_concept_distribution = {}
    for i in range(len(contingency_matrix)):
        if np.sum(contingency_matrix[i]) > 0:  # 只处理非空聚类
            distribution = {}
            for j in range(len(contingency_matrix[i])):
                if contingency_matrix[i, j] > 0:
                    concept_percentage = contingency_matrix[i, j] / np.sum(contingency_matrix[i]) * 100
                    distribution[j] = concept_percentage
            cluster_concept_distribution[i] = distribution
    
    # 计算每个概念被分配到不同聚类的客户端分布
    concept_cluster_distribution = {}
    for j in range(contingency_matrix.shape[1]):
        if np.sum(contingency_matrix[:, j]) > 0:  # 只处理非空概念
            distribution = {}
            for i in range(contingency_matrix.shape[0]):
                if contingency_matrix[i, j] > 0:
                    cluster_percentage = contingency_matrix[i, j] / np.sum(contingency_matrix[:, j]) * 100
                    distribution[i] = cluster_percentage
            concept_cluster_distribution[j] = distribution
    
    # 计算聚类和概念的大小
    cluster_counts = np.zeros(max(cluster_labels) + 1)
    concept_counts = np.zeros(max(concept_labels) + 1)
    
    for label in cluster_labels:
        cluster_counts[label] += 1
    
    for label in concept_labels:
        concept_counts[label] += 1
    
    return {
        "ari": ari,  # 调整兰德指数，-1到1，1表示完全匹配
        "nmi": nmi,  # 标准化互信息，0到1，1表示完全匹配
        "purity": cluster_purity,  # 聚类纯度，0到1，1表示每个聚类只包含一个概念
        "num_clusters": len(set(cluster_labels)),  # 聚类数量
        "num_concepts": len(set(concept_labels)),  # 概念数量
        "cluster_sizes": {i: int(count) for i, count in enumerate(cluster_counts) if count > 0},  # 每个聚类的大小
        "concept_sizes": {i: int(count) for i, count in enumerate(concept_counts) if count > 0},  # 每个概念的大小
        "cluster_concept_distribution": cluster_concept_distribution,  # 每个聚类中的概念分布
        "concept_cluster_distribution": concept_cluster_distribution,  # 每个概念在聚类中的分布
        "contingency_matrix": contingency_matrix  # 列联矩阵
    }

def print_clustering_evaluation(metrics):
    """
    打印聚类评估结果
    
    参数:
        metrics: 评估指标字典
    """
    if "error" in metrics:
        print(f"错误: {metrics['error']}")
        return
    
    print("\n======== 聚类评估结果 ========")
    print(f"调整兰德指数 (ARI): {metrics['ari']:.4f}  (范围: -1 到 1, 越高越好)")
    print(f"标准化互信息 (NMI): {metrics['nmi']:.4f}  (范围: 0 到 1, 越高越好)")
    print(f"聚类纯度 (Purity): {metrics['purity']:.4f}  (范围: 0 到 1, 越高越好)")
    print(f"聚类数量: {metrics['num_clusters']}")
    print(f"概念数量: {metrics['num_concepts']}")
    
    print("\n聚类大小:")
    for cluster_id, size in metrics['cluster_sizes'].items():
        print(f"  聚类 {cluster_id}: {size} 客户端")
    
    print("\n概念大小:")
    for concept_id, size in metrics['concept_sizes'].items():
        print(f"  概念 {concept_id}: {size} 客户端")
    
    print("\n每个聚类中的概念分布:")
    for cluster_id, distribution in metrics['cluster_concept_distribution'].items():
        print(f"  聚类 {cluster_id}:")
        for concept_id, percentage in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
            print(f"    概念 {concept_id}: {percentage:.1f}%")
    
    print("\n每个概念在聚类中的分布:")
    for concept_id, distribution in metrics['concept_cluster_distribution'].items():
        print(f"  概念 {concept_id}:")
        for cluster_id, percentage in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
            print(f"    聚类 {cluster_id}: {percentage:.1f}%")
    
    # 评估聚类质量
    if metrics['ari'] > 0.8:
        quality = "极佳"
    elif metrics['ari'] > 0.6:
        quality = "良好"
    elif metrics['ari'] > 0.4:
        quality = "一般"
    elif metrics['ari'] > 0.2:
        quality = "较差"
    else:
        quality = "很差"
    
    print(f"\n聚类与真实概念的一致性: {quality}")
    
    if metrics['ari'] < 0.4:
        print("\n改进建议:")
        print("  1. 尝试调整聚类数量，使其接近真实概念数量")
        print(f"     - 当前聚类数: {metrics['num_clusters']}, 真实概念数: {metrics['num_concepts']}")
        print("  2. 优化特征提取方法，确保不同概念的特征更加可分")
        print("  3. 考虑使用更强大的聚类算法或调整现有算法的参数")
        print("  4. 为聚类算法提供更多的训练数据或迭代次数")

def visualize_confusion_matrix(metrics, output_file=None):
    """
    可视化混淆矩阵（聚类与概念的对应关系）
    
    参数:
        metrics: 评估指标字典，包含contingency_matrix
        output_file: 输出文件路径，如果为None则显示图表
    """
    # 确保有混淆矩阵
    if 'contingency_matrix' not in metrics:
        print("错误: 没有找到混淆矩阵数据")
        return
    
    # 设置中文字体
    setup_chinese_font()
    
    # 获取混淆矩阵
    cm = metrics['contingency_matrix']
    
    # 创建一个DataFrame用于显示
    clusters = [f"簇 {i}" for i in range(cm.shape[0])]
    concepts = [f"概念 {i}" for i in range(cm.shape[1])]
    df_cm = pd.DataFrame(cm, index=clusters, columns=concepts)
    
    # 计算百分比矩阵 (按行归一化)
    cm_percent = np.zeros_like(cm, dtype=float)
    for i in range(cm.shape[0]):
        row_sum = np.sum(cm[i])
        if row_sum > 0:
            cm_percent[i] = cm[i] / row_sum * 100
    
    df_cm_percent = pd.DataFrame(cm_percent, index=clusters, columns=concepts)
    
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # 绘制原始计数热力图
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu", ax=ax1)
    ax1.set_title('聚类-概念对应关系 (客户端数)')
    ax1.set_ylabel('聚类')
    ax1.set_xlabel('概念')
    
    # 绘制百分比热力图
    sns.heatmap(df_cm_percent, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax2)
    ax2.set_title('聚类-概念对应关系 (百分比%)')
    ax2.set_ylabel('聚类')
    ax2.set_xlabel('概念')
    
    plt.tight_layout()
    
    # 保存或显示图表
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存至: {output_file}")
    else:
        plt.show()
    
    plt.close()

def create_evaluation_report(cluster_assignments, true_concepts, output_dir="clustering_evaluation"):
    """
    创建完整的聚类评估报告
    
    参数:
        cluster_assignments: 字典，将客户端ID映射到算法分配的聚类 
        true_concepts: 字典，将客户端ID映射到真实概念
        output_dir: 报告输出目录
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取评估指标
    metrics = evaluate_clustering(cluster_assignments, true_concepts)
    
    # 打印评估结果
    print_clustering_evaluation(metrics)
    
    # 保存指标到JSON文件
    metrics_file = os.path.join(output_dir, "clustering_metrics.json")
    
    # 移除numpy数组，因为它们不能直接JSON序列化
    metrics_json = metrics.copy()
    if 'contingency_matrix' in metrics_json:
        metrics_json['contingency_matrix'] = metrics_json['contingency_matrix'].tolist()
    
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_json, f, indent=4)
    
    # 可视化混淆矩阵
    confusion_matrix_file = os.path.join(output_dir, "confusion_matrix.png")
    visualize_confusion_matrix(metrics, confusion_matrix_file)
    
    # 创建客户端分配CSV文件
    assignments_df = pd.DataFrame({
        'client_id': list(set(cluster_assignments.keys()) & set(true_concepts.keys())),
    })
    
    # 添加集群和概念列
    assignments_df['cluster'] = assignments_df['client_id'].apply(lambda cid: cluster_assignments[cid])
    assignments_df['concept'] = assignments_df['client_id'].apply(lambda cid: true_concepts[cid])
    
    # 添加是否正确分类列
    # 我们需要找出每个概念主要对应的聚类
    concept_to_cluster = {}
    for concept_id, distribution in metrics['concept_cluster_distribution'].items():
        concept_to_cluster[concept_id] = max(distribution, key=distribution.get)
    
    # 检查客户端是否被分配到其概念对应的主要聚类
    assignments_df['correct_cluster'] = assignments_df.apply(
        lambda row: row['cluster'] == concept_to_cluster.get(row['concept'], -1), 
        axis=1
    )
    
    # 保存到CSV
    assignments_file = os.path.join(output_dir, "client_assignments.csv")
    assignments_df.to_csv(assignments_file, index=False)
    
    # 计算并保存概念聚类映射
    concept_cluster_mapping = {
        concept: cluster
        for concept, cluster in concept_to_cluster.items()
    }
    
    mapping_file = os.path.join(output_dir, "concept_cluster_mapping.json")
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(concept_cluster_mapping, f, indent=4)
    
    # 生成额外的可视化
    
    # 1. 聚类大小与概念大小对比
    plt.figure(figsize=(12, 6))
    
    # 聚类大小
    plt.subplot(1, 2, 1)
    cluster_ids = list(metrics['cluster_sizes'].keys())
    cluster_sizes = [metrics['cluster_sizes'][i] for i in cluster_ids]
    plt.bar(cluster_ids, cluster_sizes, color='skyblue')
    plt.title('聚类大小分布')
    plt.xlabel('聚类ID')
    plt.ylabel('客户端数量')
    
    # 概念大小
    plt.subplot(1, 2, 2)
    concept_ids = list(metrics['concept_sizes'].keys())
    concept_sizes = [metrics['concept_sizes'][i] for i in concept_ids]
    plt.bar(concept_ids, concept_sizes, color='salmon')
    plt.title('概念大小分布')
    plt.xlabel('概念ID')
    plt.ylabel('客户端数量')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "size_distributions.png"), dpi=300)
    plt.close()
    
    # 2. 聚类纯度可视化
    plt.figure(figsize=(10, 6))
    
    purity_by_cluster = {}
    for cluster_id, distribution in metrics['cluster_concept_distribution'].items():
        # 获取主要概念的比例作为纯度
        max_concept_percentage = max(distribution.values())
        purity_by_cluster[cluster_id] = max_concept_percentage / 100  # 转换为0-1范围
    
    cluster_ids = list(purity_by_cluster.keys())
    purities = [purity_by_cluster[i] for i in cluster_ids]
    
    # 设置颜色根据纯度值
    colors = plt.cm.RdYlGn(np.array(purities))
    
    plt.bar(cluster_ids, purities, color=colors)
    plt.title('各聚类的纯度')
    plt.xlabel('聚类ID')
    plt.ylabel('纯度 (0-1)')
    plt.ylim(0, 1.05)
    
    # 添加水平线表示平均纯度
    plt.axhline(y=metrics['purity'], color='r', linestyle='--', label=f'平均纯度: {metrics["purity"]:.2f}')
    plt.legend()
    
    # 添加文本标签
    for i, v in enumerate(purities):
        plt.text(cluster_ids[i], v + 0.02, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cluster_purity.png"), dpi=300)
    plt.close()
    
    print(f"\n评估报告已保存至目录: {output_dir}")
    print(f"- 评估指标: {metrics_file}")
    print(f"- 混淆矩阵: {confusion_matrix_file}")
    print(f"- 客户端分配: {assignments_file}")
    print(f"- 概念聚类映射: {mapping_file}")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="评估FedDCA聚类结果与真实概念的一致性")
    parser.add_argument("--cluster_file", default="results/clustering/final_cluster_assignments.json", 
                        help="FedDCA聚类结果文件路径")
    parser.add_argument("--drift_dir", default="Cifar100_clustered/", 
                        help="概念漂移数据集路径")
    parser.add_argument("--iteration", type=int, default=0, 
                        help="要比较的迭代轮次")
    parser.add_argument("--output_dir", default="cluster_evaluation_results", 
                        help="评估报告输出目录")
    
    args = parser.parse_args()
    
    # 设置中文字体
    setup_chinese_font()
    
    # 加载数据
    print(f"从 {args.cluster_file} 加载聚类结果...")
    cluster_assignments = load_cluster_assignments(args.cluster_file)
    
    print(f"从 {args.drift_dir} 加载迭代 {args.iteration} 的真实概念...")
    true_concepts = load_true_concepts(args.drift_dir, args.iteration)
    
    if not cluster_assignments or not true_concepts:
        print("错误: 无法加载所需的数据")
        sys.exit(1)
    
    # 创建评估报告
    create_evaluation_report(cluster_assignments, true_concepts, args.output_dir)
