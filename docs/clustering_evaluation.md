# FedDCA聚类验证工具

## 概述

该工具集用于验证FedDCA算法的聚类结果是否与客户端的真实概念划分一致。工具提供多种评估指标和可视化方法，帮助分析聚类质量和识别改进方向。

## 主要功能

1. **聚类评估指标计算**：
   - 调整兰德指数 (ARI)：衡量两个聚类之间的相似度，范围[-1, 1]
   - 标准化互信息 (NMI)：衡量聚类与真实标签之间的互信息，范围[0, 1]
   - 聚类纯度 (Purity)：衡量每个聚类中占主导的类别比例，范围[0, 1]

2. **聚类可视化**：
   - 混淆矩阵：展示聚类与真实概念的对应关系
   - 聚类分布图：展示各聚类规模和概念分布
   - 聚类纯度图：展示每个聚类的纯度

3. **概念-聚类映射分析**：
   - 分析每个概念被划分到哪些聚类中
   - 分析每个聚类中包含哪些概念

4. **多轮次比较**：
   - 支持比较不同迭代轮次的聚类效果
   - 绘制指标随时间变化的趋势图

## 工具列表

1. `clustering_metrics.py` - 基础聚类评估工具
   - `evaluate_clustering()` - 计算聚类指标
   - `get_true_concepts_at_iteration()` - 获取指定迭代的真实概念
   - `print_clustering_evaluation()` - 打印评估结果

2. `evaluate_clustering.py` - 全面聚类评估程序
   - `create_evaluation_report()` - 生成完整评估报告
   - `visualize_confusion_matrix()` - 可视化混淆矩阵
   - 包含多种可视化图表生成功能

3. `run_clustering_evaluation.py` - 批量评估脚本
   - 支持评估多个迭代轮次
   - 自动生成评估报告和可视化图表

4. `patch_feddca_clustering_evaluation.py` - FedDCA集成评估功能
   - 在FedDCA类中添加实时评估能力
   - 定期评估聚类效果并输出报告

## 使用方法

### 1. 单独评估聚类结果

```bash
python run_clustering_evaluation.py --cluster_file results/clustering/final_cluster_assignments.json --drift_dir system/Cifar100_clustered/ --iterations 0,10,20
```

参数说明:
- `--cluster_file`: FedDCA聚类结果文件路径
- `--drift_dir`: 概念漂移数据集目录
- `--iterations`: 要评估的迭代轮次，用逗号分隔。使用'all'评估所有轮次
- `--output_dir`: 评估报告输出目录

### 2. 集成评估到FedDCA

```bash
# 修补FedDCA类
python patch_feddca_clustering_evaluation.py

# 然后正常运行FedDCA，聚类评估将自动执行
python main.py --algorithm FedDCA --use_drift_dataset True --drift_data_dir system/Cifar100_clustered/
```

### 3. 查看评估报告

评估报告将输出到指定目录（默认为`clustering_evaluation_results`），包括:
- 评估指标JSON文件
- 混淆矩阵可视化
- 聚类分布图
- 聚类纯度图
- 指标随时间变化的趋势图

## 评估指标解释

1. **调整兰德指数 (ARI)**
   - 范围: [-1, 1]
   - 解释: 值越接近1表示聚类结果越接近真实概念划分
   - 评价标准:
     - > 0.8: 极佳
     - > 0.6: 良好
     - > 0.4: 一般
     - > 0.2: 较差
     - ≤ 0.2: 很差

2. **标准化互信息 (NMI)**
   - 范围: [0, 1]
   - 解释: 值越接近1表示聚类包含的概念信息越多

3. **聚类纯度 (Purity)**
   - 范围: [0, 1]
   - 解释: 值越接近1表示每个聚类越"纯净"，即主要包含一个概念

## 改进建议

如果聚类效果不理想(ARI < 0.4)，可尝试以下改进:

1. 调整聚类数量，使其接近真实概念数量
2. 优化特征提取方法，确保不同概念的特征更加可分
3. 调整聚类算法参数，如温度系数、学习率等
4. 提供更多的训练数据或迭代次数
5. 考虑使用更强大的聚类算法

## 开发者

本工具集由FedDCA团队开发维护。
