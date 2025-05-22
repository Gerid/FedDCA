    def select_clustering_algorithm(self):
        """
        根据配置选择使用的聚类算法
        
        返回:
            str: 聚类算法名称 ('vwc', 'label_conditional' 或 'enhanced_label')
        """
        # 默认使用原始的VWC
        clustering_method = 'vwc'
        
        # 如果在args中指定了聚类方法，则使用指定的方法
        if hasattr(self.args, 'clustering_method'):
            clustering_method = self.args.clustering_method
            
        # 确保聚类方法是有效的
        valid_methods = ['vwc', 'label_conditional', 'enhanced_label']
        if clustering_method not in valid_methods:
            print(f"警告：未知的聚类方法 '{clustering_method}'，使用默认的 'vwc' 方法")
            clustering_method = 'vwc'
            
        return clustering_method
