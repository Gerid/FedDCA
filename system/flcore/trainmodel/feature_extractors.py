import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseFeatureExtractor(nn.Module):
    """
    为FedCCFA实现的基础特征提取器模型，用于获取中间层特征
    """
    def __init__(self, base_model):
        super(BaseFeatureExtractor, self).__init__()
        self.base_model = base_model
        self.feature_dim = self._get_feature_dim()
        
    def _get_feature_dim(self):
        """确定特征维度"""
        if hasattr(self.base_model, 'fc'):
            # 如果是ResNet等带fc层的模型
            if isinstance(self.base_model.fc, nn.Linear):
                return self.base_model.fc.in_features
        elif hasattr(self.base_model, 'classifier'):
            # 如果是VGG等带classifier层的模型
            if isinstance(self.base_model.classifier, nn.Sequential):
                for m in self.base_model.classifier:
                    if isinstance(m, nn.Linear):
                        return m.in_features
            elif isinstance(self.base_model.classifier, nn.Linear):
                return self.base_model.classifier.in_features
                
        # 默认值，如果无法确定
        return 512
    
    def extract_features(self, x, return_features=False):
        """
        提取特征并可选择返回中间表示
        
        Args:
            x: 输入数据
            return_features: 是否返回中间特征
        
        Returns:
            输出数据和可选的中间特征
        """
        # 这个方法需要在子类中实现
        raise NotImplementedError("子类必须实现extract_features方法")
    
    def forward(self, x):
        """标准前向传播"""
        return self.extract_features(x, return_features=False)


class CNNFeatureExtractor(BaseFeatureExtractor):
    """为CNNs模型实现特征提取"""
    def __init__(self, base_model):
        super(CNNFeatureExtractor, self).__init__(base_model)
        
    def extract_features(self, x, return_features=False):
        # 这里假设模型可以分为特征提取部分和分类器部分
        # 需要根据具体模型结构调整
        features = None
        if hasattr(self.base_model, 'features'):
            # VGG等模型
            features = self.base_model.features(x)
            features = features.view(features.size(0), -1)
            output = self.base_model.classifier(features)
        elif hasattr(self.base_model, 'conv1'):
            # ResNet等模型
            x = self.base_model.conv1(x)
            x = self.base_model.bn1(x) if hasattr(self.base_model, 'bn1') else x
            x = self.base_model.relu(x) if hasattr(self.base_model, 'relu') else F.relu(x)
            x = self.base_model.maxpool(x) if hasattr(self.base_model, 'maxpool') else x
            
            x = self.base_model.layer1(x) if hasattr(self.base_model, 'layer1') else x
            x = self.base_model.layer2(x) if hasattr(self.base_model, 'layer2') else x
            x = self.base_model.layer3(x) if hasattr(self.base_model, 'layer3') else x
            x = self.base_model.layer4(x) if hasattr(self.base_model, 'layer4') else x
            
            x = self.base_model.avgpool(x) if hasattr(self.base_model, 'avgpool') else F.adaptive_avg_pool2d(x, (1, 1))
            features = torch.flatten(x, 1)
            output = self.base_model.fc(features)
        else:
            # 通用模型，尝试直接获取输出
            output = self.base_model(x)
            features = output  # 如果没有更好的方法，就使用输出作为特征
        
        if return_features:
            return output, features
        else:
            return output


# 封装函数，用于将模型包装为支持特征提取的模型
def wrap_model_for_feature_extraction(model):
    """
    将模型包装为支持特征提取的模型
    
    Args:
        model: 原始模型
    
    Returns:
        支持特征提取的模型
    """
    if hasattr(model, 'extract_features'):
        # 已经支持特征提取
        return model
    
    # 根据模型类型选择合适的特征提取器
    if any(hasattr(model, attr) for attr in ['conv1', 'features']):
        return CNNFeatureExtractor(model)
    
    # 如果无法识别模型类型，使用通用特征提取器
    return BaseFeatureExtractor(model)
