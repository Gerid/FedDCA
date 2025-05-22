import os
import json
import pandas as pd
import matplotlib.pyplot as plt
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

def get_client_concepts_at_iteration(data_dir="Cifar100_clustered/", iteration=0):
    """
    获取指定迭代中所有客户端的概念分类
    
    参数:
        data_dir: 数据存储路径，默认为"Cifar100_clustered/"
        iteration: 要查询的迭代编号
        
    返回:
        client_concepts: 字典，将客户端ID映射到概念ID
        是否成功读取数据
    """
    # 构建配置文件路径
    config_path = os.path.join(data_dir, "drift_info", "concept_config.json")
    
    if not os.path.exists(config_path):
        print(f"错误: 找不到配置文件: {config_path}")
        return {}, False
    
    try:
        # 加载配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            concept_config = json.load(f)
        
        # 检查是否存在客户端轨迹
        if 'client_concept_trajectories' not in concept_config:
            print("错误: 配置中未找到客户端概念轨迹")
            return {}, False
        
        # 获取所有客户端在指定迭代的概念
        client_concepts = {}
        for client_id, trajectory in concept_config['client_concept_trajectories'].items():
            if 0 <= iteration < len(trajectory):
                client_concepts[client_id] = trajectory[iteration]
            else:
                print(f"警告: 客户端 {client_id} 没有在迭代 {iteration} 的概念记录")
        
        print(f"成功读取迭代 {iteration} 的客户端概念分类")
        return client_concepts, True
    
    except Exception as e:
        print(f"读取配置文件时出错: {e}")
        return {}, False

def print_client_concepts(client_concepts, iteration):
    """打印客户端概念分类表格"""
    if not client_concepts:
        print("没有数据可显示")
        return
    
    print(f"\n===== 迭代 {iteration} 的客户端概念分类 =====")
    
    # 按照概念分组
    concept_groups = defaultdict(list)
    for client_id, concept_id in client_concepts.items():
        concept_groups[concept_id].append(client_id)
    
    # 打印表格
    print(f"{'概念ID':<10}{'客户端数量':<15}{'客户端列表'}")
    print("-" * 70)
    
    for concept_id, clients in sorted(concept_groups.items()):
        # 限制显示的客户端数量，避免输出太长
        client_display = ", ".join(clients[:10])
        if len(clients) > 10:
            client_display += f"... (共{len(clients)}个)"
        
        print(f"{concept_id:<10}{len(clients):<15}{client_display}")
    
    print(f"\n总客户端数量: {len(client_concepts)}")

def visualize_client_concepts(client_concepts, iteration, show_plot=True, save_path=None):
    """
    可视化客户端概念分类
    
    参数:
        client_concepts: 客户端概念字典
        iteration: 迭代编号
        show_plot: 是否显示图表
        save_path: 保存图表的路径
    """
    if not client_concepts:
        print("没有数据可可视化")
        return
    
    # 确保配置了中文字体
    setup_chinese_font()
    
    # 统计每个概念的客户端数量
    concept_counts = defaultdict(int)
    for concept_id in client_concepts.values():
        concept_counts[concept_id] += 1
    
    # 创建可视化
    plt.figure(figsize=(10, 6))
    
    # 饼图
    concepts = list(concept_counts.keys())
    counts = [concept_counts[c] for c in concepts]
    
    plt.pie(counts, labels=[f'概念 {c}' for c in concepts], autopct='%1.1f%%', 
           startangle=90, shadow=True, explode=[0.05] * len(concepts))
    
    plt.title(f'迭代 {iteration} 的客户端概念分布')
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"可视化已保存到: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def export_to_csv(client_concepts, output_path):
    """
    将客户端概念分类导出为CSV文件
    
    参数:
        client_concepts: 客户端概念字典
        output_path: 输出CSV文件路径
    """
    if not client_concepts:
        print("没有数据可导出")
        return
    
    # 创建数据框
    df = pd.DataFrame([
        {"客户端ID": client_id, "概念ID": concept_id}
        for client_id, concept_id in client_concepts.items()
    ])
    
    # 导出CSV
    df.to_csv(output_path, index=False)
    print(f"数据已导出到: {output_path}")

if __name__ == "__main__":
    import argparse
    
    # 配置中文字体支持
    setup_chinese_font()
    
    parser = argparse.ArgumentParser(description="读取CIFAR-100概念漂移数据集中客户端的概念分类")
    parser.add_argument("--data_dir", default="Cifar100_clustered/", help="数据集路径")
    parser.add_argument("--iteration", type=int, default=0, help="要查询的迭代编号")
    parser.add_argument("--visualize", action="store_true", help="是否生成可视化")
    parser.add_argument("--save", action="store_true", help="是否保存结果")
    parser.add_argument("--output_dir", default="concept_results", help="输出目录")
    
    args = parser.parse_args()
    
    # 读取客户端概念
    client_concepts, success = get_client_concepts_at_iteration(args.data_dir, args.iteration)
    
    if success:
        # 打印表格
        print_client_concepts(client_concepts, args.iteration)
        
        # 可视化
        if args.visualize or args.save:
            if args.save:
                # 创建输出目录
                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir)
                
                # 设置保存路径
                viz_path = os.path.join(args.output_dir, f"iteration_{args.iteration}_concepts.png")
                csv_path = os.path.join(args.output_dir, f"iteration_{args.iteration}_concepts.csv")
                
                # 保存可视化和CSV
                visualize_client_concepts(client_concepts, args.iteration, show_plot=args.visualize, save_path=viz_path)
                export_to_csv(client_concepts, csv_path)
            else:
                visualize_client_concepts(client_concepts, args.iteration)
    else:
        print("无法获取客户端概念信息，请检查数据路径和配置文件。")
