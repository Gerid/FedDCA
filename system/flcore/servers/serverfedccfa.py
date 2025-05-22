import copy
import torch
import time
import numpy as np
import os
from flcore.servers.serverbase import Server
from flcore.clients.clientfedccfa import clientFedCCFA
from threading import Thread
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import traceback


class FedCCFA(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        
        # 设置客户端
        self.set_slow_clients()
        self.set_clients(clientFedCCFA)
        
        # 分类器层的键
        self.clf_keys = None
        
        # 全局原型和性能追踪
        self.global_protos = []
        self.prev_rep_norm = 0
        self.prev_clf_norm = 0
        self.rep_norm_scale = 0
        self.clf_norm_scale = 0
        
        # 时间和性能追踪
        self.Budget = []
        self.client_data_size = {}  # 记录每个客户端的数据大小
        
        # 参数设置
        if not hasattr(args, 'eps'):
            args.eps = 0.5  # DBSCAN 聚类的 eps 参数
        
        # 原型聚类设置
        if not hasattr(args, 'clustered_protos'):
            args.clustered_protos = False  # 是否使用聚类的原型
        
        # Oracle 模式设置
        if not hasattr(args, 'oracle'):
            args.oracle = False  # 是否使用 oracle 模式合并
        
        print(f"\nFedCCFA设置完成!")
        print(f"参与率 / 总客户端数: {self.join_ratio} / {self.num_clients}")

    def get_client_data_size(self, clients):
        """
        记录每个客户端的数据大小
        
        Args:
            clients: 客户端列表
        """
        for client in clients:
            self.client_data_size[client.id] = len(client.train_samples)

    def send_params(self, selected_clients):
        """
        向选定的客户端发送模型参数
        
        Args:
            selected_clients: 选定的客户端列表
        """
        for client in selected_clients:
            client.set_parameters(self.global_model)

    def send_rep_params(self, selected_clients):
        """
        向选定的客户端发送表示层参数
        
        Args:
            selected_clients: 选定的客户端列表
        """
        # 获取表示层参数
        rep_params = [param for name, param in self.global_model.named_parameters() 
                     if name not in self.clf_keys]
        
        # 向每个客户端发送
        for client in selected_clients:
            client.set_rep_params(rep_params)

    def aggregate_rep(self, clients):
        """
        聚合所有客户端的表示层参数
        
        Args:
            clients: 客户端列表
        """
        # 获取表示层参数
        rep_params = [param for name, param in self.global_model.named_parameters() 
                     if name not in self.clf_keys]
        
        # 初始化新参数
        new_params = torch.zeros_like(parameters_to_vector(rep_params))
        total_size = 0
        
        # 加权聚合参数
        for client in clients:
            client_size = self.client_data_size[client.id]
            total_size += client_size
            
            # 获取客户端的表示层参数
            client_rep_params = [param for name, param in client.model.named_parameters() 
                                if name not in self.clf_keys]
            client_params = parameters_to_vector(client_rep_params)
            
            # 加权累加
            new_params += client_size * client_params
        
        # 计算加权平均
        if total_size > 0:
            new_params /= total_size
        
        # 应用到全局模型
        vector_to_parameters(new_params, rep_params)

    def aggregate_protos(self, clients):
        """
        根据客户端的标签分布聚合全局原型
        
        Args:
            clients: 客户端列表
        """
        # 检查聚合权重方式
        if not hasattr(self.args, 'weights'):
            self.args.weights = "label"  # 默认使用标签加权
        
        if self.args.weights == "uniform":
            # 均匀权重聚合
            aggregate_proto_dict = {}
            
            for client in clients:
                local_protos = client.local_protos
                for label in local_protos.keys():
                    if label in aggregate_proto_dict:
                        aggregate_proto_dict[label] += local_protos[label]
                    else:
                        aggregate_proto_dict[label] = local_protos[label].clone()
            
            # 计算平均值
            for label, proto in aggregate_proto_dict.items():
                aggregate_proto_dict[label] = proto / len(clients)
        else:
            # 按标签分布加权聚合
            aggregate_proto_dict = {}
            label_size_dict = {}
            
            for client in clients:
                # 获取客户端的标签分布
                label_distribution = client.label_distribution
                local_protos = client.local_protos
                
                for label in local_protos.keys():
                    if label in aggregate_proto_dict:
                        aggregate_proto_dict[label] += local_protos[label] * label_distribution[label]
                        label_size_dict[label] += label_distribution[label]
                    else:
                        aggregate_proto_dict[label] = local_protos[label] * label_distribution[label]
                        label_size_dict[label] = label_distribution[label]
            
            # 计算加权平均值
            for label, proto in aggregate_proto_dict.items():
                if label_size_dict[label] > 0:
                    aggregate_proto_dict[label] = proto / label_size_dict[label]
        
        # 更新全局原型
        self.global_protos = [aggregate_proto_dict[label] if label in aggregate_proto_dict else None 
                             for label in range(self.args.num_classes)]

    @staticmethod
    def madd(vecs):
        """
        计算向量之间的余弦相似度距离
        
        Args:
            vecs: 向量列表
            
        Returns:
            距离矩阵
        """
        def cos_sim(a, b):
            # 计算余弦相似度
            return a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        
        num = len(vecs)
        res = np.zeros((num, num))
        
        # 计算每对向量之间的距离
        for i in range(num):
            for j in range(i + 1, num):
                dist = 0.0
                for z in range(num):
                    if z == i or z == j:
                        continue
                    
                    # 计算距离差异
                    dist += np.abs(cos_sim(vecs[i], vecs[z]) - cos_sim(vecs[j], vecs[z]))
                
                # 对称矩阵
                if num > 2:  # 避免除以零
                    res[i][j] = res[j][i] = dist / (num - 2)
        
        return res

    def merge_classifiers(self, clf_params_dict):
        """
        通过测量每个标签的参数距离，合并相同概念下的分类器
        
        Args:
            clf_params_dict: 所有客户端的分类器参数。键是客户端ID，值是参数。
            
        Returns:
            每个标签合并的客户端ID。键是标签，值是客户端集群列表。
        """
        client_ids = np.array(list(clf_params_dict.keys()))
        client_clf_params = list(clf_params_dict.values())
        label_num = self.args.num_classes
        
        label_merged_dict = {}
        for label in range(label_num):
            # 计算每个标签的距离矩阵，通过分类器的参数
            params_list = []
            for clf_params in client_clf_params:
                params = [param[label].detach().cpu().numpy() for param in clf_params]
                params_list.append(np.hstack(params))
            params_list = np.array(params_list)
            
            # 计算距离矩阵
            dist = self.madd(params_list)
            
            # 使用 DBSCAN 聚类
            clustering = DBSCAN(eps=self.args.eps, min_samples=1, metric="precomputed")
            clustering.fit(dist)
            
            # 处理聚类结果
            merged_ids = []
            for i in set(clustering.labels_):
                indices = np.where(clustering.labels_ == i)[0]
                if len(indices) > 1:
                    # 至少有两个分类器被合并
                    ids = client_ids[indices]
                    ids = sorted(list(ids))  # 排序以便观察
                    merged_ids.append(ids)
            
            label_merged_dict[label] = merged_ids
        
        return label_merged_dict

    def oracle_merging(self, _round, ids):
        """
        Oracle 模式的合并策略，用于评估最佳合并结果
        
        Args:
            _round: 当前轮次
            ids: 客户端ID列表
            
        Returns:
            每个标签合并的客户端ID
        """
        # 这里实现一个简单的 Oracle 策略，可以根据实际场景修改
        if _round < 100:
            # 在轮次100之前，所有客户端合并为一个组
            return {
                label: [ids] for label in range(self.args.num_classes)
            }
        else:
            # 在轮次100之后，根据ID尾数将客户端分为3组
            group1 = [_id for _id in ids if _id % 10 < 3]
            group2 = [_id for _id in ids if 3 <= _id % 10 < 6]
            group3 = [_id for _id in ids if _id % 10 >= 6]
            
            # 对不同的标签应用不同的分组方式
            return {
                0: [ids],  # 所有客户端合并
                1: [group1, [_id for _id in ids if _id not in group1]],  # 分两组
                2: [group1, [_id for _id in ids if _id not in group1]],
                3: [group2, [_id for _id in ids if _id not in group2]],
                4: [group2, [_id for _id in ids if _id not in group2]],
                5: [group3, [_id for _id in ids if _id not in group3]],
                6: [group3, [_id for _id in ids if _id not in group3]],
                7: [ids],  # 所有客户端合并
                8: [ids],
                9: [ids]
            }

    def aggregate_label_params(self, label, clients):
        """
        聚合特定标签的参数
        
        Args:
            label: 要聚合的标签
            clients: 同一组的客户端
            
        Returns:
            聚合的参数向量
        """
        # 获取标签对应的参数
        label_params = [param[label] for name, param in self.global_model.named_parameters() 
                       if name in self.clf_keys]
        
        # 初始化聚合参数
        aggregated_params = torch.zeros_like(parameters_to_vector(label_params))
        label_size = 0
        
        # 聚合参数
        for client in clients:
            client_label_params = [param[label] for name, param in client.model.named_parameters() 
                                  if name in self.clf_keys]
            client_params = parameters_to_vector(client_label_params)
            
            # 根据权重方式选择聚合方法
            if hasattr(self.args, 'weights') and self.args.weights == "uniform":
                aggregated_params += client_params
            else:
                # 使用客户端标签分布作为权重
                client_weight = client.label_distribution[label]
                aggregated_params += client_params * client_weight
                label_size += client_weight
        
        # 计算平均值
        if hasattr(self.args, 'weights') and self.args.weights == "uniform":
            if clients:  # 确保不为空
                aggregated_params /= len(clients)
        else:
            if label_size > 0:
                aggregated_params /= label_size
        
        return aggregated_params

    def aggregate_label_protos(self, label, clients):
        """
        聚合特定标签的原型
        
        Args:
            label: 要聚合的标签
            clients: 同一组的客户端
            
        Returns:
            聚合的原型向量
        """
        # 初始化聚合原型
        aggregated_proto = None
        label_size = 0
        
        # 聚合原型
        for client in clients:
            if label in client.local_protos:
                client_proto = client.local_protos[label]
                
                # 根据权重方式选择聚合方法
                if hasattr(self.args, 'weights') and self.args.weights == "uniform":
                    if aggregated_proto is None:
                        aggregated_proto = client_proto.clone()
                    else:
                        aggregated_proto += client_proto
                else:
                    # 使用客户端标签分布作为权重
                    client_weight = client.label_distribution[label]
                    if client_weight > 0:
                        if aggregated_proto is None:
                            aggregated_proto = client_proto.clone() * client_weight
                        else:
                            aggregated_proto += client_proto * client_weight
                        label_size += client_weight
        
        # 计算平均值
        if aggregated_proto is not None:
            if hasattr(self.args, 'weights') and self.args.weights == "uniform":
                valid_clients = sum(1 for client in clients if label in client.local_protos)
                if valid_clients > 0:
                    aggregated_proto /= valid_clients
            else:
                if label_size > 0:
                    aggregated_proto /= label_size
        
        return aggregated_proto if aggregated_proto is not None else torch.zeros(1)

    def local_evaluate(self, selected_clients, _round):
        """
        评估选定客户端在本地数据上的性能
        
        Args:
            selected_clients: 选定的客户端
            _round: 当前轮次
            
        Returns:
            平均本地准确率
        """
        local_accuracies = []
        
        for client in selected_clients:
            # 测试本地模型在本地数据上的性能
            test_acc, test_num = client.test_metrics()
            if test_num > 0:
                local_accuracies.append(test_acc / test_num)
        
        # 计算平均准确率
        if local_accuracies:
            avg_accuracy = sum(local_accuracies) / len(local_accuracies)
            
            # 记录性能
            if _round % 20 == 0:
                self.rs_test_acc.append(avg_accuracy)
            
            return avg_accuracy
        else:
            return 0.0

    def global_evaluate(self, selected_clients, global_test_sets, _round):
        """
        评估选定客户端在全局测试集上的性能
        
        Args:
            selected_clients: 选定的客户端
            global_test_sets: 全局测试集
            _round: 当前轮次
            
        Returns:
            平均全局准确率
        """
        # 如果有特殊的全局测试集，则使用它进行评估
        # 对于标准 PFL-Non-IID 项目，我们使用客户端的测试数据
        global_accuracies = []
        
        for client in selected_clients:
            # 测试本地模型在测试集上的性能
            test_acc, test_num = client.test_metrics()
            if test_num > 0:
                global_accuracies.append(test_acc / test_num)
        
        # 计算平均准确率
        if global_accuracies:
            avg_accuracy = sum(global_accuracies) / len(global_accuracies)
            return avg_accuracy
        else:
            return 0.0

    def last_round_evaluate(self, clients, global_test_sets):
        """
        在最后一轮评估所有客户端
        
        Args:
            clients: 所有客户端
            global_test_sets: 全局测试集
        """
        # 本地评估
        local_accuracies = []
        for client in clients:
            test_acc, test_num = client.test_metrics()
            if test_num > 0:
                local_accuracies.append(test_acc / test_num)
        
        # 计算平均本地准确率
        if local_accuracies:
            avg_local_accuracy = sum(local_accuracies) / len(local_accuracies)
            print(f"Final average local accuracy: {avg_local_accuracy:.4f}")
        else:
            print("No valid local accuracy results.")

    def train(self):
        """训练过程的主控制流"""
        # 获取所有客户端的数据大小信息
        self.get_client_data_size(self.clients)
        
        # 设置分类器层的键
        # 确定分类器层的键，通常是最后几层
        # 这里假设最后两层是分类器层
        if self.clf_keys is None:
            self.clf_keys = list(self.global_model.state_dict().keys())[-2:]
            print(f"设置分类器层键为：{self.clf_keys}")
        
        # 将分类器层键分配给所有客户端
        for client in self.clients:
            client.clf_keys = self.clf_keys
        
        # 训练循环
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            
            # 处理数据漂移（如果有配置）
            if hasattr(self.args, 'drift_pattern'):
                if self.args.drift_pattern == "sudden" and i == 100:
                    print("模拟突然漂移在轮次 100")
                    # 在真实系统中需要实现 sudden_drift 函数
                elif self.args.drift_pattern == "recurrent" and i in [100, 150]:
                    print(f"模拟重复漂移在轮次 {i}")
                    # 在真实系统中需要实现 sudden_drift 函数
                elif self.args.drift_pattern == "incremental" and i in [100, 110, 120]:
                    print(f"模拟渐进漂移在轮次 {i}")
                    # 在真实系统中需要实现 incremental_drift 函数
            
            # 每轮选择客户端
            self.selected_clients = self.select_clients()
            
            # 向选定的客户端发送参数
            self.send_params(self.selected_clients)
            
            # 用于存储平衡训练后的分类器参数
            balanced_clf_params_dict = {}
            
            # 每个客户端进行训练
            for client in self.selected_clients:
                # 更新客户端的标签分布
                client.update_label_distribution()
                
                # 如果配置了平衡训练，执行平衡训练
                if hasattr(self.args, 'balanced_epochs') and self.args.balanced_epochs > 0:
                    client.balance_train()
                    balanced_clf_params_dict[client.id] = copy.deepcopy(client.get_clf_parameters())
                
                # 如果没有使用聚类的原型，共享全局原型
                if not hasattr(self.args, 'clustered_protos') or not self.args.clustered_protos:
                    client.global_protos = copy.deepcopy(self.global_protos)
                
                # 使用原型进行训练
                client.train_with_protos(i)
                
                # 如果没有平衡训练，使用当前分类器参数
                if not hasattr(self.args, 'balanced_epochs') or self.args.balanced_epochs == 0:
                    balanced_clf_params_dict[client.id] = copy.deepcopy(client.get_clf_parameters())
            
            # 聚合表示层参数
            self.aggregate_rep(self.selected_clients)
            
            # 聚合原型
            self.aggregate_protos(self.selected_clients)
            
            # 向客户端发送聚合后的表示层参数
            self.send_rep_params(self.selected_clients)
            
            # 基于分类器参数进行标签级别的合并
            if hasattr(self.args, 'oracle') and self.args.oracle:
                # 使用 Oracle 策略合并
                label_merged_dict = self.oracle_merging(i, [c.id for c in self.selected_clients])
            else:
                # 使用基于参数相似性的合并
                label_merged_dict = self.merge_classifiers(balanced_clf_params_dict)
            
            # 打印合并结果
            for label, merged_identities in label_merged_dict.items():
                if merged_identities:  # 如果有合并的客户端
                    print(f"标签 {label} 合并：{merged_identities}")
                    
                    for indices in merged_identities:
                        # 聚合个性化分类器参数和原型
                        clients_group = [client for client in self.selected_clients if client.id in indices]
                        if clients_group:
                            # 聚合标签级别的参数
                            aggregated_label_params = self.aggregate_label_params(label, clients_group)
                            aggregated_label_proto = self.aggregate_label_protos(label, clients_group)
                            
                            # 更新每个客户端的参数
                            for client in clients_group:
                                client_label_params = [param[label] for name, param in client.model.named_parameters()
                                                     if name in self.clf_keys]
                                vector_to_parameters(aggregated_label_params, client_label_params)
                                client.set_label_params(label, client_label_params)
                                
                                # 更新原型
                                if len(aggregated_label_proto) > 1:  # 确保有效的原型
                                    client.global_protos[label] = aggregated_label_proto.clone()
            
            # 保存客户端的分类器参数
            for client in self.selected_clients:
                client.p_clf_params = copy.deepcopy(client.get_clf_parameters())
            
            # 计算此轮的训练时间
            e_t = time.time()
            self.Budget.append(e_t - s_t)
            
            # 定期评估
            if i % 20 == 0:
                local_accuracy = self.local_evaluate(self.selected_clients, i)
                global_accuracy = self.global_evaluate(self.selected_clients, None, i)
                print(f"轮次 {i} | 本地准确率: {local_accuracy:.4f} | 全局准确率: {global_accuracy:.4f}")
            
            # 是否达到预设的准确率要求提前结束
            if self.auto_break and len(self.rs_test_acc) > 0 and self.rs_test_acc[-1] > 0.99:
                break
        
        # 最终评估
        print("\n训练完成!")
        self.last_round_evaluate(self.clients, None)
        
        # 输出耗时统计
        if len(self.Budget) > 0:
            avg_time = sum(self.Budget) / len(self.Budget)
            print(f"平均每轮耗时: {avg_time:.2f}秒")
        
        # 保存结果和模型
        self.save_results()
        self.save_models()

    def save_models(self):
        """保存全局模型"""
        if not os.path.exists("saved_models"):
            os.makedirs("saved_models")
        
        # 保存全局模型
        torch.save(self.global_model.state_dict(), 
                  os.path.join("saved_models", f"FedCCFA_global_{self.dataset}_{self.times}.pt"))
        
        print(f"模型保存完成：全局模型已保存到 saved_models/FedCCFA_global_{self.dataset}_{self.times}.pt")
    
    def save_results(self):
        """保存训练结果"""
        if not os.path.exists("results"):
            os.makedirs("results")
        
        # 保存精度和损失
        algo = self.algorithm
        result_path = f"results/{self.dataset}_{algo}_{self.goal}_{self.times}.h5"
        
        with h5py.File(result_path, 'w') as hf:
            hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
            hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
            hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
        
        print(f"结果已保存到 {result_path}")
    
    def set_clients(self, clientObj):
        """初始化客户端对象"""
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, data_dir=self.args.drift_data_dir if hasattr(self.args, 'drift_data_dir') else None)
            test_data = read_client_data(self.dataset, i, is_train=False, data_dir=self.args.drift_data_dir if hasattr(self.args, 'drift_data_dir') else None)
            
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=train_data, 
                            test_samples=test_data, 
                            train_slow=self.train_slow_clients, 
                            send_slow=self.send_slow_clients)
            self.clients.append(client)
