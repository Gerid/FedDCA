"""
执行FedDCA服务器的聚类函数
"""

def perform_clustering(self, round_idx=None):
    """
    执行客户端聚类，同时支持两种不同的聚类方法
    """
    try:
        if self.args.algorithm == 'feddca':
            # 如果是第一轮，或者应该更新聚类
            update_clusters = round_idx is None or round_idx % self.args.cluster_interval == 0
            
            if update_clusters:
                print(f"\n{'初始化' if not self.cluster_inited else '更新'}聚类...")
                
                # 确定使用的聚类算法
                if hasattr(self.args, 'clustering_method') and self.args.clustering_method == 'label_conditional':
                    # 使用基于标签条件的Wasserstein聚类
                    self.perform_label_conditional_clustering()
                else:
                    # 使用原始的VWC聚类
                    self.perform_vwc_clustering()
                
                # 更新集群模型
                self.update_cluster_models()
                
                # 更新客户端聚类历史
                self.update_cluster_history(round_idx)
                
                # 设置聚类初始化标志
                self.cluster_inited = True
            
            # 检查是否需要可视化聚类结果
            if not self.cluster_inited or (self.args.visualize_clusters and hasattr(self.args, 'vis_interval') and 
              round_idx is not None and round_idx % self.args.vis_interval == 0):
                self.visualize_clustering(round_idx)
                
            # 准备集群分布信息
            if hasattr(self.args, 'verbose') and self.args.verbose:
                self.print_cluster_distribution()
        else:
            # 如果不是FedDCA算法，跳过聚类
            pass
            
    except Exception as e:
        print(f"聚类过程中出错: {str(e)}")
        import traceback
        print(traceback.format_exc())
        # 在发生错误时，确保至少有一个默认集群
        if not self.clusters:
            print("使用默认集群分配（所有客户端分配到簇0）")
            self.clusters = {client.id: 0 for client in self.selected_clients}
            self.cluster_centroids = {0: copy.deepcopy(self.global_model)}

def perform_vwc_clustering(self):
    """
    执行基于变分Wasserstein聚类的客户端分组
    """
    try:
        print("使用原始的变分Wasserstein聚类 (VWC)...")
        
        # 收集所有客户端的特征表示
        client_features = {}
        for client in self.selected_clients:
            if hasattr(client, 'intermediate_output') and client.intermediate_output is not None:
                client_features[client.id] = client.intermediate_output
        
        if not client_features:
            print("警告: 没有可用的客户端特征进行聚类")
            return
            
        # 初始化VWC聚类器
        clusterer = VariationalWassersteinClustering(
            num_clusters=self.args.num_clusters,
            num_iterations=self.args.vwc_iterations if hasattr(self.args, 'vwc_iterations') else 100,
            device=self.device
        )
        
        # 执行聚类
        self.clusters = clusterer.fit(client_features)
        
        # 打印结果
        print(f"VWC聚类完成，形成 {len(set(self.clusters.values()))} 个集群")
        
    except Exception as e:
        print(f"VWC聚类过程中出错: {str(e)}")
        import traceback
        print(traceback.format_exc())
        # 确保在出错时有默认集群
        if not self.clusters:
            self.clusters = {client.id: 0 for client in self.selected_clients}

def perform_label_conditional_clustering(self):
    """
    执行基于标签条件分布的Wasserstein聚类
    """
    try:
        print("使用基于标签条件的Wasserstein聚类...")
        
        # 使用标签条件Wasserstein聚类函数
        verbose = hasattr(self.args, 'verbose') and self.args.verbose
        self.clusters = perform_label_conditional_clustering(
            clients=self.selected_clients,
            num_clusters=self.args.num_clusters,
            device=self.device,
            verbose=verbose
        )
        
        # 检查聚类结果
        if not self.clusters:
            print("警告: 标签条件聚类未能产生有效结果，回退到随机分配")
            # 随机分配客户端到簇
            for client in self.selected_clients:
                self.clusters[client.id] = random.randint(0, self.args.num_clusters - 1)
        else:
            print(f"标签条件聚类完成，形成 {len(set(self.clusters.values()))} 个集群")
            
    except Exception as e:
        print(f"标签条件聚类过程中出错: {str(e)}")
        import traceback
        print(traceback.format_exc())
        # 确保在出错时有默认集群
        if not self.clusters:
            self.clusters = {client.id: 0 for client in self.selected_clients}

def update_cluster_history(self, round_idx):
    """
    更新客户端的聚类历史记录，用于分析聚类稳定性
    
    Args:
        round_idx: 当前训练轮次
    """
    for client_id, cluster_id in self.clusters.items():
        if client_id not in self.client_cluster_history:
            self.client_cluster_history[client_id] = []
        self.client_cluster_history[client_id].append(cluster_id)
        
    # 维护历史记录长度，防止占用过多内存
    max_history = 50  # 最多保留50轮历史
    for client_id in self.client_cluster_history:
        if len(self.client_cluster_history[client_id]) > max_history:
            self.client_cluster_history[client_id] = self.client_cluster_history[client_id][-max_history:]

def print_cluster_distribution(self):
    """
    打印当前簇的分布情况
    """
    if not self.clusters:
        print("警告: 没有可用的聚类数据")
        return
        
    # 计算每个簇的客户端数量
    cluster_counts = {}
    for cluster_id in self.clusters.values():
        if cluster_id not in cluster_counts:
            cluster_counts[cluster_id] = 0
        cluster_counts[cluster_id] += 1
    
    # 打印分布
    print("\n集群分布:")
    for cluster_id, count in sorted(cluster_counts.items()):
        percentage = count / len(self.clusters) * 100
        print(f"  簇 {cluster_id}: {count} 个客户端 ({percentage:.1f}%)")
