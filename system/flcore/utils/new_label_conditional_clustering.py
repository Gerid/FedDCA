    def label_conditional_clustering(self):
        """使用标签条件特征分布进行客户端聚类"""
        try:
            # 收集按标签条件分组的代理数据
            label_conditional_data = self.collect_label_conditional_proxy_data()
            
            if not label_conditional_data:
                print("未能收集有效的标签条件代理数据")
                return False
                
            # 计算标签条件Wasserstein距离矩阵
            print("计算标签条件Wasserstein距离...")
            distance_matrix = compute_label_conditional_wasserstein_distance(label_conditional_data, device=self.device)
            
            # 使用鲁棒聚类方法，自动处理异常情况
            client_ids = list(label_conditional_data.keys())
            num_clusters = min(self.args.num_clusters, len(client_ids))
            if num_clusters < 2:
                num_clusters = 1
                
            print(f"执行鲁棒聚类，客户端数: {len(client_ids)}，目标集群数: {num_clusters}")
            cluster_assignments = robust_clustering(distance_matrix, num_clusters, client_ids)
            
            # 更新客户端集群分配
            for idx, client_id in enumerate(client_ids):
                cluster_id = cluster_assignments[idx]
                
                # 记录聚类历史记录
                if client_id not in self.client_cluster_history:
                    self.client_cluster_history[client_id] = []
                self.client_cluster_history[client_id].append(cluster_id)
                
                # 更新当前集群分配
                self.clusters[client_id] = cluster_id
                
            print(f"标签条件聚类完成. 生成了 {len(set(cluster_assignments))} 个集群.")
            
            # 可视化聚类结果
            visualize_clusters(distance_matrix, cluster_assignments, client_ids, 
                              save_path=f"results/clustering_round_{self.current_iteration}.png")
                              
            return True
            
        except Exception as e:
            print(f"标签条件聚类失败: {str(e)}")
            print(traceback.format_exc())
            return False
