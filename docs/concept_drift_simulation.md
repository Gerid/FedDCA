# 标准CIFAR-100上的概念漂移模拟指南

本指南介绍了如何在标准CIFAR-100数据集上模拟概念漂移，以便测试FedDCA算法的真实性能。

## 背景

FedDCA算法在修改后的CIFAR-100数据集上展现出异常高的准确率(97%)，这可能是由于数据集特性导致的，而非算法本身的优异性能。为了客观评估FedDCA的性能，我们实现了在标准CIFAR-100数据集上模拟概念漂移的功能。

## 实现方法

我们通过修改`clientdca.py`中的`load_train_data`方法，使其能够按照预设的漂移模式在每次加载训练数据时修改数据标签和样本分布。具体实现了以下几种漂移模式：

1. **标签漂移(Label Drift)** - 改变部分或全部类别的标签
2. **样本分布漂移(Prior Probability Shift)** - 改变各类别样本的分布比例
3. **协变量漂移(Covariate Shift)** - 修改输入特征分布，如添加噪声或改变亮度
4. **组合漂移** - 同时应用多种漂移模式

## 使用方法

### 1. 启用概念漂移模拟

在创建`clientDCA`实例或调用`main.py`时，设置以下参数：

```python
args.simulate_drift = True  # 启用漂移模拟
args.increment_iteration = True  # 每次调用load_train_data时递增迭代计数
```

### 2. 自定义漂移模式

可以通过修改`initialize_drift_patterns`方法来自定义漂移模式和时间表：

```python
# 修改漂移模式
client.drift_patterns['label_drift_custom'] = {
    'label_mapping': {i: (i+10)%100 for i in range(0, 30)}
}

# 修改漂移时间表
client.drift_schedule = [
    {'iterations': [0, 50], 'pattern': None},
    {'iterations': [50, 150], 'pattern': 'label_drift_custom'},
    # ...其他阶段
]
```

### 3. 运行测试脚本

我们提供了一个测试脚本`test_drift_simulation.py`，可以用来可视化漂移效果并分析模型性能：

```bash
python test_drift_simulation.py
```

## 分析步骤

### 1. 数据漂移可视化

首先运行`test_single_client_drift()`函数，查看不同迭代阶段的数据变化情况：

```python
test_single_client_drift()
```

这将生成一个可视化图像`drift_simulation_visualization.png`，展示不同漂移阶段的数据样本。

### 2. 模型性能分析

运行完整实验，对比在标准数据集上的性能：

```python
analyze_model_performance()
```

比较以下指标：
- 在标准CIFAR-100上的准确率曲线
- 在不同漂移阶段的准确率变化
- 与其他联邦学习算法的性能比较

### 3. 聚类效果分析

分析FedDCA在面对漂移时的聚类效果：

1. 观察`client_cluster_history.json`中的聚类变化
2. 查看是否在漂移点处发生显著的聚类重组
3. 评估漂移前后聚类的纯度和一致性

## 预期结果

通过在标准CIFAR-100上模拟漂移，我们预期会观察到：

1. FedDCA的准确率将明显低于在修改版CIFAR-100上的表现（可能从97%下降到40%-50%）
2. 算法在面对概念漂移时会表现出适应性，即准确率在漂移后会有所下降，但随后会逐渐恢复
3. 客户端聚类会随着漂移而变化，反映出算法的自适应能力

## 排查建议

如果遇到以下问题，可尝试：

1. **模型准确率异常低**：检查漂移模式是否过于剧烈，可尝试减少漂移强度
2. **训练不稳定**：检查批次大小和学习率，在漂移环境下可能需要较小的学习率
3. **内存错误**：减少批次大小或降低数据预处理复杂度
4. **聚类不变化**：检查漂移是否足够明显，以及聚类参数是否适当

通过这种方式，我们可以更客观地评估FedDCA算法的真实性能，并了解其在面对概念漂移时的行为模式。
