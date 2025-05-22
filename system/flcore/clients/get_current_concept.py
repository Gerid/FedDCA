def get_current_concept(self):
    """根据当前迭代和漂移类型确定要使用的概念
    
    支持三种漂移类型:
    1. 突变漂移(sudden drift): 在漂移点立即切换到新概念
    2. 渐进漂移(gradual drift): 在漂移点周围窗口期内，逐渐从一个概念过渡到另一个
    3. 周期漂移(recurring drift): 按照周期在可用概念间循环切换
    
    返回:
        当前应用的概念或混合概念
    """
    import numpy as np
    import copy
    
    # 尝试使用外部模块的函数来获取当前概念
    if hasattr(self, 'use_shared_concepts') and self.use_shared_concepts:
        try:
            from utils.concept_drift_simulation import get_current_concept as external_get_concept
            
            # 如果没有初始化客户端概念，先初始化
            if not hasattr(self, 'client_concepts') or self.client_concepts is None:
                self.initialize_client_concepts()
            
            drift_type = getattr(self, 'drift_type', 'sudden')
            drift_points = getattr(self, 'drift_points', [40, 80, 120, 160])
            window_size = getattr(self, 'gradual_window', 10)
            period = getattr(self, 'recurring_period', 30)
            
            # 使用外部函数获取当前概念
            current_concept = external_get_concept(
                self.client_concepts, 
                drift_type, 
                self.current_iteration, 
                drift_points, 
                window_size, 
                period
            )
            
            # 记录当前概念ID (用于跟踪和分析)
            if current_concept and 'id' in current_concept:
                self.current_concept_id = current_concept['id']
                
            return current_concept
            
        except (ImportError, Exception) as e:
            print(f"无法使用外部模块获取概念，将使用内部方法: {str(e)}")
    
    # 使用内部方法
    # 如果没有初始化客户端概念，先初始化
    if not hasattr(self, 'client_concepts') or self.client_concepts is None:
        self.initialize_client_concepts()
        
    # 如果客户端没有概念，返回None
    if not hasattr(self, 'client_concepts') or len(self.client_concepts) == 0:
        return None
        
    # 获取当前迭代轮次
    current_iter = self.current_iteration
    
    # 确定漂移类型
    drift_type = getattr(self, 'drift_type', 'sudden')
    
    # 获取漂移点列表
    drift_points = getattr(self, 'drift_points', [40, 80, 120, 160])  # 默认漂移点
    
    # 获取客户端可用概念列表
    available_concepts = self.client_concepts
    num_concepts = len(available_concepts)
    
    if num_concepts == 0:
        return None
    elif num_concepts == 1:
        # 记录当前概念ID (用于跟踪和分析)
        if 'id' in available_concepts[0]:
            self.current_concept_id = available_concepts[0]['id']
        return copy.deepcopy(available_concepts[0])
        
    # 根据漂移类型确定当前概念
    if drift_type == 'sudden':
        # 突变漂移: 在漂移点立即切换概念
        concept_idx = 0
        for i, point in enumerate(drift_points):
            if current_iter >= point:
                concept_idx = (i + 1) % num_concepts
                
        # 记录当前概念ID
        if 'id' in available_concepts[concept_idx]:
            self.current_concept_id = available_concepts[concept_idx]['id']
            
        return copy.deepcopy(available_concepts[concept_idx])
        
    elif drift_type == 'gradual':
        # 渐进漂移: 在漂移点周围一段时间内，逐渐过渡到新概念
        window_size = getattr(self, 'gradual_window', 10)  # 过渡窗口大小
        
        # 确定基础概念索引
        base_idx = 0
        target_idx = 0
        transition_prob = 0.0
        in_transition = False
        
        for i, point in enumerate(drift_points):
            if current_iter >= point:
                base_idx = (i + 1) % num_concepts
                
            # 检查是否在过渡窗口内
            if point - window_size <= current_iter < point:
                base_idx = i % num_concepts
                target_idx = (i + 1) % num_concepts
                # 计算过渡概率
                transition_prob = (current_iter - (point - window_size)) / window_size
                in_transition = True
                break
        
        # 记录当前概念ID
        if 'id' in available_concepts[base_idx]:
            self.current_concept_id = available_concepts[base_idx]['id']
            
        # 如果在过渡期，创建混合概念 - 结合两个概念的标签映射
        if in_transition and transition_prob > 0:
            base_concept = available_concepts[base_idx]
            target_concept = available_concepts[target_idx]
            
            # 创建混合概念
            mixed_concept = copy.deepcopy(base_concept)
            
            # 混合标签映射
            if 'label_mapping' in base_concept and 'label_mapping' in target_concept:
                base_mapping = base_concept['label_mapping']
                target_mapping = target_concept['label_mapping']
                
                # 选择标签映射的混合策略
                if np.random.random() < transition_prob:
                    # 随着转换概率增加，更多地使用目标概念的映射
                    for label, new_label in target_mapping.items():
                        if np.random.random() < transition_prob:
                            mixed_concept['label_mapping'][label] = new_label
            
            # 混合类别权重
            if 'class_weights' in base_concept and 'class_weights' in target_concept:
                base_weights = base_concept['class_weights']
                target_weights = target_concept['class_weights']
                mixed_weights = {}
                
                # 根据过渡概率加权混合两个概念的权重
                for class_idx in range(100):  # CIFAR-100
                    base_w = base_weights.get(class_idx, 0.1)
                    target_w = target_weights.get(class_idx, 0.1)
                    mixed_weights[class_idx] = base_w * (1 - transition_prob) + target_w * transition_prob
                    
                mixed_concept['class_weights'] = mixed_weights
                
            return mixed_concept
        else:
            # 不在过渡期，使用当前概念
            return copy.deepcopy(available_concepts[base_idx])
            
    elif drift_type == 'recurring':
        # 周期漂移: 按固定周期循环使用概念
        period = getattr(self, 'recurring_period', 30)  # 默认30轮一个周期
        concept_idx = (current_iter // period) % num_concepts
        
        # 记录当前概念ID
        if 'id' in available_concepts[concept_idx]:
            self.current_concept_id = available_concepts[concept_idx]['id']
            
        return copy.deepcopy(available_concepts[concept_idx])
        
    else:
        # 默认使用第一个概念
        if 'id' in available_concepts[0]:
            self.current_concept_id = available_concepts[0]['id']
            
        return copy.deepcopy(available_concepts[0])
