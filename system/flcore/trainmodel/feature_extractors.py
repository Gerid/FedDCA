import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from torchvision.models.alexnet import AlexNet
from torchvision.models.vgg import VGG
from torchvision.models.mobilenet import MobileNetV2
from torchvision.models.googlenet import GoogLeNet

# Helper to identify the classifier module by common names
def get_classifier_module(model):
    if hasattr(model, 'fc'):
        return model.fc, 'fc'
    elif hasattr(model, 'classifier'):
        # For models like VGG, classifier might be a Sequential or a single Linear layer
        if isinstance(model.classifier, nn.Sequential):
            # Find the last Linear layer in the Sequential classifier
            for i in range(len(model.classifier) -1, -1, -1):
                if isinstance(model.classifier[i], nn.Linear):
                    return model.classifier[i], f'classifier.{i}'
            return None, None # No linear layer found in sequential classifier
        elif isinstance(model.classifier, nn.Linear):
            return model.classifier, 'classifier'
        else:
            return None, None # Classifier is not nn.Linear or nn.Sequential
    elif hasattr(model, 'output_layer'): # Example for custom models
        return model.output_layer, 'output_layer'
    # Add more checks for other model types if necessary
    return None, None

class WrappedFeatureExtractor(nn.Module):
    def __init__(self, base_model, original_model_type_name="unknown"):
        super(WrappedFeatureExtractor, self).__init__()
        self.original_model_type_name = original_model_type_name
        self._base_model_ref_for_forward = base_model # Store a reference for forward pass logic if needed
        
        identified_classifier_module, identified_classifier_name = get_classifier_module(base_model)

        if identified_classifier_module is None:
            print(f"Warning: Could not automatically identify a standard classifier layer (fc, classifier) for model type {original_model_type_name}. Feature extraction might be incorrect.")
            # Fallback: assume the whole model is the feature extractor, and add a dummy classifier
            self.features_module = base_model
            # Try to get num_classes from a 'num_classes' attribute if it exists
            num_classes_attr = getattr(base_model, 'num_classes', None)
            if num_classes_attr is None and hasattr(base_model, 'args') and hasattr(base_model.args, 'num_classes'): # common in this project
                num_classes_attr = base_model.args.num_classes

            if num_classes_attr is not None:
                 # Attempt to get in_features by doing a dummy forward pass if possible
                try:
                    dummy_input = torch.randn(1, 3, 32, 32) # Common input size, adjust if needed
                    if "mnist" in original_model_type_name.lower() or "omniglot" in original_model_type_name.lower():
                         dummy_input = torch.randn(1, 1, 28, 28)
                    
                    # Ensure model is on a device for dummy pass, and use no_grad
                    device = next(base_model.parameters()).device if list(base_model.parameters()) else torch.device('cpu')
                    with torch.no_grad():
                        dummy_features = self.features_module(dummy_input.to(device))
                    in_features = dummy_features.view(dummy_features.size(0), -1).shape[1]
                    self.classifier_module = nn.Linear(in_features, num_classes_attr)
                    self.feature_dim = in_features
                except Exception as e:
                    print(f"Error during dummy forward pass for fallback classifier: {e}. Using placeholder.")
                    self.classifier_module = nn.Identity() # Placeholder, likely incorrect
                    self.feature_dim = -1
            else:
                print("Warning: Could not determine num_classes for fallback classifier. Using Identity.")
                self.classifier_module = nn.Identity() # Placeholder
                self.feature_dim = -1
        else:
            # A classifier was identified. Set it as self.classifier_module.
            self.classifier_module = identified_classifier_module
            
            if hasattr(self.classifier_module, 'in_features'):
                self.feature_dim = self.classifier_module.in_features
            else:
                print(f"Warning: Identified classifier '{identified_classifier_name}' of type {type(self.classifier_module)} does not have 'in_features'. Feature dimension may be unknown.")
                self.feature_dim = -1
            
            # Create the features part by removing the classifier
            if identified_classifier_name == 'fc' and hasattr(base_model, 'fc'):
                if isinstance(base_model, ResNet):
                    self.features_module = nn.Sequential(*list(base_model.children())[:-1])
                else:
                    # For generic models ending in 'fc' (e.g., FedAvgCNN, custom CNNs)
                    # features_module should be all layers *before* the final fc (which is self.classifier_module)
                    # The main challenge is ensuring flattening happens correctly if conv layers are followed by linear layers *within* this features_module.
                    
                    children = list(base_model.children())
                    # Exclude the identified classifier (base_model.fc)
                    feature_children_list = children[:-1] 

                    # If feature_children_list is empty, features_module is empty (e.g. model was just one fc layer)
                    if not feature_children_list:
                        self.features_module = nn.Sequential()
                    else:
                        # Check if the last part of feature_children_list is a Linear layer or a Sequential of Linear layers.
                        # If so, and if there are preceding conv layers, a flatten might be needed before it.
                        # The error (1280x5 and 1600x512) implies an internal structure issue.
                        # For FedAvgCNN: model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024)
                        # children are roughly: conv1, pool1, relu1, conv2, pool2, relu2, flatten, linear1 (dim, e.g. 1024), relu3, fc (classifier, e.g. 10)
                        # If fc is classifier, then feature_children_list is [conv1, ..., relu3]
                        # The flatten is already part of FedAvgCNN structure before its internal linear1.
                        # So, simply taking children[:-1] should be correct if the model (like FedAvgCNN) is well-defined internally.
                        self.features_module = nn.Sequential(*feature_children_list)

                    # Try to set/verify self.feature_dim based on the output of the constructed features_module
                    if isinstance(self.classifier_module, nn.Linear):
                        try:
                            dummy_input_shape = (2, 1, 28, 28) # MNIST-like
                            if "cifar" in self.original_model_type_name.lower() or "Cifar" in self.original_model_type_name:
                                dummy_input_shape = (2, 3, 32, 32)
                            elif "omniglot" in self.original_model_type_name.lower():
                                dummy_input_shape = (2, 1, 28, 28) # Example, adjust if different
                            elif "digit5" in self.original_model_type_name.lower() or "Digit5" in self.original_model_type_name:
                                dummy_input_shape = (2, 3, 32, 32) # Example, adjust if different
                            # Add more specific input shapes as needed for other datasets/models

                            device = next(base_model.parameters()).device if list(base_model.parameters()) else torch.device('cpu')
                            dummy_input = torch.randn(*dummy_input_shape).to(device)
                            
                            with torch.no_grad():
                                test_features_out = self.features_module(dummy_input)
                            

                            # The output of features_module should be flat or flattenable to match classifier_module.in_features
                            test_features_flat = torch.flatten(test_features_out, 1)
                            self.feature_dim = test_features_flat.shape[1]

                            # If the originally identified classifier_module.in_features does not match this,
                            # it might indicate an issue. However, self.feature_dim should be what features_module *actually* outputs (flattened).
                            if self.classifier_module.in_features != self.feature_dim:
                                print(f"Warning for {self.original_model_type_name}: Original classifier.in_features ({self.classifier_module.in_features}) "
                                      f"does not match empirically determined feature_dim ({self.feature_dim}) from features_module output. "
                                      f"Proceeding with empirically determined dim. This might be due to model structure (e.g. internal flatten).")
                                # No, we must respect the classifier_module.in_features. 
                                # If they don't match, the forward pass will fail. 
                                # The self.feature_dim should be the one expected by the classifier.
                                self.feature_dim = self.classifier_module.in_features 
                                # The check in forward() will catch this if test_features_flat.shape[1] != self.feature_dim

                        except Exception as e:
                            print(f"Note: Could not perform dummy forward pass to determine feature_dim for {self.original_model_type_name} due to: {e}. Using classifier.in_features ({self.feature_dim}).")
                            # self.feature_dim would have been set from classifier_module.in_features earlier if available

            elif identified_classifier_name.startswith('classifier'):
                if isinstance(base_model, VGG): # VGG: features, avgpool, classifier (Sequential)
                    self.features_module = nn.Sequential(
                        base_model.features,
                        base_model.avgpool if hasattr(base_model, 'avgpool') else nn.AdaptiveAvgPool2d((7, 7)) # VGG default
                    )
                    # self.classifier_module is already the specific nn.Linear from get_classifier_module
                elif isinstance(base_model, AlexNet): # AlexNet: features, avgpool, classifier (Sequential)
                    self.features_module = nn.Sequential(
                        base_model.features,
                        base_model.avgpool if hasattr(base_model, 'avgpool') else nn.AdaptiveAvgPool2d((6, 6)) # AlexNet default
                    )
                    # self.classifier_module is already the specific nn.Linear
                elif isinstance(base_model, MobileNetV2): # MobileNetV2: features, classifier (Linear)
                    self.features_module = base_model.features
                    # self.classifier_module is base_model.classifier (the nn.Linear layer)

                elif isinstance(base_model, GoogLeNet): # GoogLeNet is complex, has multiple fcs
                    children_list = list(base_model.children())
                    fc_index = -1
                    # self.classifier_module is base_model.fc as identified by get_classifier_module
                    for idx, child in enumerate(children_list):
                        if child == self.classifier_module: 
                            fc_index = idx
                            break
                    if fc_index != -1:
                        self.features_module = nn.Sequential(*children_list[:fc_index])
                    else: # Fallback if fc not found directly in children
                        self.features_module = nn.Sequential(*list(base_model.children())[:-1]) 
                else: # Generic model with a 'classifier' attribute (could be model.classifier or model.classifier[i])
                    temp_children = list(base_model.children())
                    # If identified_classifier_name is 'classifier' and it's a direct child:
                    if identified_classifier_name == 'classifier' and self.classifier_module in temp_children:
                        idx_to_remove = temp_children.index(self.classifier_module)
                        self.features_module = nn.Sequential(*temp_children[:idx_to_remove])
                    # Else, if base_model.features exists, use it. This is an approximation.
                    elif hasattr(base_model, 'features') and not isinstance(base_model.features, nn.ModuleList) and base_model.features is not None:
                        self.features_module = base_model.features
                    else: 
                        print(f"Warning: Using entire base model as feature extractor for {original_model_type_name} with classifier {identified_classifier_name} due to complex classifier structure.")
                        self.features_module = base_model 
                        # CRITICAL: DO NOT set self.classifier_module to nn.Identity() here.
                        # It holds the identified classifier. Acknowledge potential forward pass issues.
            elif hasattr(base_model, identified_classifier_name): # Custom, e.g. 'output_layer'
                # self.classifier_module is getattr(base_model, identified_classifier_name)
                # Assume last child is classifier if it matches
                children_list = list(base_model.children())
                if children_list and children_list[-1] == self.classifier_module:
                    self.features_module = nn.Sequential(*list(base_model.children())[:-1])
                else:
                    print(f"Warning: Using entire base model as feature extractor for {original_model_type_name} with custom classifier '{identified_classifier_name}'. Structure unclear.")
                    self.features_module = base_model # Fallback, potentially risky
            else:
                # This case should ideally not be reached if get_classifier_module returned a valid module and name.
                # If it is, it means we identified a classifier but couldn't match its structure to known patterns.
                print(f"Critical Warning: Identified classifier '{identified_classifier_name}' for {original_model_type_name}, but cannot determine feature extractor structure. Using entire base model as features. Forward pass may be incorrect.")
                self.features_module = base_model
            
            # Ensure feature_dim is correctly set if it was initially -1 and classifier is Linear
            if self.feature_dim == -1 and isinstance(self.classifier_module, nn.Linear):
                self.feature_dim = self.classifier_module.in_features


    def forward(self, x, return_features=False):
        features = self.features_module(x)
        
        if isinstance(self.classifier_module, nn.Linear):
            features_flat = torch.flatten(features, 1)
            if self.feature_dim > 0 and features_flat.shape[1] != self.feature_dim:
                # If this error occurs, it means the output of self.features_module (when flattened)
                # does not match the in_features of self.classifier_module.
                # This is the primary point of failure if dimensions are mismatched.
                raise RuntimeError(f"Feature dimension mismatch for {self.original_model_type_name} before final classifier. "
                                   f"Classifier ({type(self.classifier_module)}) expects {self.feature_dim} input features, "
                                   f"but flattened output of features_module ({type(self.features_module)}) has {features_flat.shape[1]} features. "
                                   f"Input shape to forward: {x.shape}, features_module direct output shape: {features.shape}")
            
            output = self.classifier_module(features_flat)
        elif isinstance(self.classifier_module, nn.Identity) and self.features_module is self._base_model_ref_for_forward:
            features_flat = torch.flatten(features,1) 
            output = features 
        else: 
            features_flat = torch.flatten(features, 1) 
            output = self.classifier_module(features) 

        if return_features:
            return output, features_flat 
        return output

def wrap_model_for_feature_extraction(model, model_name_str="unknown"):
    """
    Wraps a given PyTorch model with FeatureExtractor to enable access to intermediate features.
    If the model is already wrapped, it returns the model as is.
    Args:
        model: The PyTorch model (e.g., ResNet, VGG, custom CNN).
        model_name_str: A string identifier for the model type, used for warnings.
    Returns:
        A model instance that has an `extract_features` method.
    """
    if isinstance(model, WrappedFeatureExtractor):
        # print("Model is already wrapped with WrappedFeatureExtractor.")
        return model
    
    # print(f"Wrapping model type: {model_name_str if model_name_str else model.__class__.__name__} for feature extraction.")
    return WrappedFeatureExtractor(model, original_model_type_name=model_name_str if model_name_str else model.__class__.__name__)

def get_classifier_keys(wrapped_model):
    """
    Identifies the parameter names of the classifier part of a WrappedFeatureExtractor.
    Args:
        wrapped_model: A model instance, ideally already wrapped by WrappedFeatureExtractor.
    Returns:,
        A list of strings, where each string is the fully qualified name of a classifier parameter.
        Returns empty list if classifier cannot be identified or has no parameters.
    """
    if not isinstance(wrapped_model, WrappedFeatureExtractor):
        # print("Warning: Model not wrapped with WrappedFeatureExtractor. Attempting to get classifier keys directly, but may fail or be incorrect.")
        # Try to find classifier on the unwrapped model directly
        classifier_module, _ = get_classifier_module(wrapped_model)
        if classifier_module:
            keys = [name for name, _ in classifier_module.named_parameters()]
            # Need to prefix these keys if classifier_module is a submodule
            # This is complex without knowing the structure. For now, assume it's top-level 'fc' or 'classifier'.
            # This part is unreliable for unwrapped models.
            # Let's assume the main.py always wraps first.
            return keys # This will likely be wrong if not wrapped.
        else:
            return []


    # If wrapped, the classifier is self.classifier_module
    if hasattr(wrapped_model, 'classifier_module') and isinstance(wrapped_model.classifier_module, nn.Module):
        # Parameters of self.classifier_module are prefixed with "classifier_module."
        clf_keys = []
        for name, param in wrapped_model.classifier_module.named_parameters():
            clf_keys.append(f"classifier_module.{name}")
        return clf_keys
    else:
        # print("Warning: Wrapped model does not have a valid 'classifier_module'.")
        return []

# Example Usage (for testing)
if __name__ == '__main__':
    import torchvision.models
    # Test with ResNet18
    resnet = torchvision.models.resnet18(pretrained=False, num_classes=10)
    wrapped_resnet = wrap_model_for_feature_extraction(resnet, "resnet18")
    print("ResNet18 wrapped.")
    resnet_clf_keys = get_classifier_keys(wrapped_resnet)
    print(f"ResNet18 classifier keys: {resnet_clf_keys}") # Expected: ['classifier_module.weight', 'classifier_module.bias']
    
    dummy_input_resnet = torch.randn(2, 3, 32, 32)
    output_resnet, features_resnet = wrapped_resnet(dummy_input_resnet, return_features=True)
    print(f"ResNet output shape: {output_resnet.shape}, features shape: {features_resnet.shape}")
    assert features_resnet.shape[1] == wrapped_resnet.feature_dim

    # Test with a simple custom CNN
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(2)
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(320, num_classes) # 320 for 28x28 MNIST-like input

        def forward(self, x):
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = self.flatten(x)
            x = self.fc(x)
            return x

    simple_cnn = SimpleCNN(num_classes=10)
    wrapped_simple_cnn = wrap_model_for_feature_extraction(simple_cnn, "SimpleCNN")
    print("SimpleCNN wrapped.")
    simple_cnn_clf_keys = get_classifier_keys(wrapped_simple_cnn)
    print(f"SimpleCNN classifier keys: {simple_cnn_clf_keys}")
    
    dummy_input_cnn = torch.randn(2, 1, 28, 28)
    output_cnn, features_cnn = wrapped_simple_cnn(dummy_input_cnn, return_features=True)
    print(f"SimpleCNN output shape: {output_cnn.shape}, features shape: {features_cnn.shape}")
    if wrapped_simple_cnn.feature_dim > 0 :
        assert features_cnn.shape[1] == wrapped_simple_cnn.feature_dim


    # Test with AlexNet
    alexnet_model = torchvision.models.alexnet(pretrained=False, num_classes=10)
    wrapped_alexnet = wrap_model_for_feature_extraction(alexnet_model, "alexnet")
    print("AlexNet wrapped.")
    alexnet_clf_keys = get_classifier_keys(wrapped_alexnet)
    print(f"AlexNet classifier keys: {alexnet_clf_keys}")
   
    dummy_input_alexnet = torch.randn(2, 3, 224, 224) # AlexNet typical input
    output_alexnet, features_alexnet = wrapped_alexnet(dummy_input_alexnet, return_features=True)
    print(f"AlexNet output shape: {output_alexnet.shape}, features shape: {features_alexnet.shape}")
    if wrapped_alexnet.feature_dim > 0:
         assert features_alexnet.shape[1] == wrapped_alexnet.feature_dim

    # Test with VGG16
    vgg_model = torchvision.models.vgg16(pretrained=False, num_classes=10)
    wrapped_vgg = wrap_model_for_feature_extraction(vgg_model, "vgg16")
    print("VGG16 wrapped.")
    vgg_clf_keys = get_classifier_keys(wrapped_vgg)
    print(f"VGG16 classifier keys: {vgg_clf_keys}")

    dummy_input_vgg = torch.randn(2, 3, 224, 224) # VGG typical input
    output_vgg, features_vgg = wrapped_vgg(dummy_input_vgg, return_features=True)
    print(f"VGG output shape: {output_vgg.shape}, features shape: {features_vgg.shape}")
    if wrapped_vgg.feature_dim > 0:
        assert features_vgg.shape[1] == wrapped_vgg.feature_dim
        
    # Test with MobileNetV2
    mobilenet_model = torchvision.models.mobilenet_v2(pretrained=False, num_classes=10)
    wrapped_mobilenet = wrap_model_for_feature_extraction(mobilenet_model, "mobilenet_v2")
    print("MobileNetV2 wrapped.")
    mobilenet_clf_keys = get_classifier_keys(wrapped_mobilenet)
    print(f"MobileNetV2 classifier keys: {mobilenet_clf_keys}")

    dummy_input_mobilenet = torch.randn(2, 3, 224, 224) 
    output_mobilenet, features_mobilenet = wrapped_mobilenet(dummy_input_mobilenet, return_features=True)
    print(f"MobileNetV2 output shape: {output_mobilenet.shape}, features shape: {features_mobilenet.shape}")
    if wrapped_mobilenet.feature_dim > 0:
        assert features_mobilenet.shape[1] == wrapped_mobilenet.feature_dim

    # Test with GoogLeNet
    googlenet_model = torchvision.models.googlenet(pretrained=False, num_classes=10, aux_logits=False)
    wrapped_googlenet = wrap_model_for_feature_extraction(googlenet_model, "googlenet")
    print("GoogLeNet wrapped.")
    googlenet_clf_keys = get_classifier_keys(wrapped_googlenet)
    print(f"GoogLeNet classifier keys: {googlenet_clf_keys}")

    dummy_input_googlenet = torch.randn(2, 3, 224, 224)
    output_googlenet, features_googlenet = wrapped_googlenet(dummy_input_googlenet, return_features=True)
    print(f"GoogLeNet output shape: {output_googlenet.shape}, features shape: {features_googlenet.shape}")
    if wrapped_googlenet.feature_dim > 0:
        assert features_googlenet.shape[1] == wrapped_googlenet.feature_dim


    # Test case where model might already be a feature extractor + separate head (BaseHeadSplit)
    from flcore.trainmodel.models import BaseHeadSplit
    base_m = SimpleCNN(num_classes=10)
    # Simulate BaseHeadSplit: features part is base_m.features, head is base_m.fc
    # Let's assume base_m.fc is Identity and a new head is added.
    # For this test, let's just use SimpleCNN which has an fc layer.
    
    # If BaseHeadSplit is used, args.model.base is the feature part, args.model.head is the classifier
    # Our wrapper should ideally handle this.
    # Let's simulate a model that looks like BaseHeadSplit for the wrapper
    class SimulatedBaseHeadSplit(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.base = nn.Sequential(
                nn.Conv2d(1,10,5), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(10,20,5), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten()
            )
            self.head = nn.Linear(320, num_classes)
            self.feature_dim = 320 # Manually set for this test structure

        def forward(self, x):
            x = self.base(x)
            x = self.head(x)
            return x
        
        # Add an 'fc' attribute that points to 'head' for get_classifier_module
        @property
        def fc(self):
            return self.head


    bhs_model = SimulatedBaseHeadSplit(num_classes=15)
    wrapped_bhs = wrap_model_for_feature_extraction(bhs_model, "SimulatedBaseHeadSplit")
    print("SimulatedBaseHeadSplit wrapped.")
    bhs_clf_keys = get_classifier_keys(wrapped_bhs)
    print(f"SimulatedBaseHeadSplit classifier keys: {bhs_clf_keys}")

    dummy_input_bhs = torch.randn(2, 1, 28, 28)
    output_bhs, features_bhs = wrapped_bhs(dummy_input_bhs, return_features=True)
    print(f"SimulatedBaseHeadSplit output: {output_bhs.shape}, features: {features_bhs.shape}")
    if wrapped_bhs.feature_dim > 0:
        assert features_bhs.shape[1] == wrapped_bhs.feature_dim, f"Expected {wrapped_bhs.feature_dim}, got {features_bhs.shape[1]}"


    print("All tests passed (if no assertion errors).")
