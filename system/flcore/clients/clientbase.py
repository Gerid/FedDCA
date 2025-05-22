import copy
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        # 添加对概念漂移数据集的支持
        self.current_iteration = 0
        self.drift_data_dir = args.drift_data_dir if hasattr(args, 'drift_data_dir') else None
        self.use_drift_dataset = args.use_drift_dataset if hasattr(args, 'use_drift_dataset') else False
        self.max_iterations = args.max_iterations if hasattr(args, 'max_iterations') else 200

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs["train_slow"]
        self.send_slow = kwargs["send_slow"]
        self.train_time_cost = {"num_rounds": 0, "total_cost": 0.0}
        self.send_time_cost = {"num_rounds": 0, "total_cost": 0.0}

        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay

    def update_iteration(self, new_iteration):
        """更新当前迭代计数器，用于概念漂移数据集"""
        if hasattr(self, 'current_iteration'):
            self.current_iteration = new_iteration

    def load_train_data(self, batch_size=None):
        """加载训练数据，支持概念漂移数据集"""
        if batch_size == None:
            batch_size = self.batch_size
        
        if hasattr(self, 'use_drift_dataset') and self.use_drift_dataset and hasattr(self, 'drift_data_dir') and self.drift_data_dir:
            return self.load_drift_data(is_train=True, batch_size=batch_size)
        else:
            train_data = read_client_data(self.dataset, self.id, is_train=True)
            return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        """加载测试数据，支持概念漂移数据集"""
        if batch_size == None:
            batch_size = self.batch_size
            
        if hasattr(self, 'use_drift_dataset') and self.use_drift_dataset and hasattr(self, 'drift_data_dir') and self.drift_data_dir:
            return self.load_drift_data(is_train=False, batch_size=batch_size)
        else:
            test_data = read_client_data(self.dataset, self.id, is_train=False)
            return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
            
    def load_drift_data(self, is_train=True, batch_size=None):
        """
        从概念漂移数据集加载特定迭代的数据
        
        参数:
            is_train: 是否加载训练集，False则加载测试集
            batch_size: 批次大小，如果为None则使用默认批次大小
        """
        import os
        import json
        import numpy as np
        from torch.utils.data import DataLoader, TensorDataset
        
        if batch_size is None:
            batch_size = self.batch_size
        
        try:
            # 构建当前迭代的路径
            iter_path = os.path.join(self.drift_data_dir, f"iteration_{self.current_iteration}")
            
            # 确保迭代不超出范围
            if self.current_iteration >= self.max_iterations:
                print(f"Warning: Iteration {self.current_iteration} exceeds max {self.max_iterations}, resetting to 0")
                self.current_iteration = 0
                iter_path = os.path.join(self.drift_data_dir, f"iteration_{self.current_iteration}")
            
            # 构建数据文件路径
            data_type = "train" if is_train else "test"
            json_path = os.path.join(iter_path, data_type, f"{self.id}.json")
            npz_path = os.path.join(iter_path, data_type, f"{self.id}.npz")
            
            # 优先尝试加载 JSON，其次尝试 NPZ
            data = None
            if os.path.exists(json_path):
                # 从 JSON 加载
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif os.path.exists(npz_path):
                # 尝试从NPZ加载并检查必需键
                try:
                    loaded = np.load(npz_path)
                    if 'x' in loaded.files and 'y' in loaded.files:
                        data = {'x': loaded['x'], 'y': loaded['y']}
                    else:
                        raise KeyError("'x' or 'y' not found in NPZ archive")
                except KeyError as e:
                    print(f"Warning: NPZ data missing for client {self.id}: {str(e)}, falling back to default loader")
                    # 如果NPZ加载失败，回退到默认加载方式
                    if is_train:
                        train_data = read_client_data(self.dataset, self.id, is_train=True)
                        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)
                    else:
                        test_data = read_client_data(self.dataset, self.id, is_train=False)
                        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
            else:
                print(f"Warning: Data file not found for client {self.id}, paths: {json_path} or {npz_path}")
                # 回退到默认数据加载
                if is_train:
                    train_data = read_client_data(self.dataset, self.id, is_train=True)
                    return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)
                else:
                    test_data = read_client_data(self.dataset, self.id, is_train=False)
                    return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
            
            if data is not None:
                # 加载数据后的统一处理
                import torch
                x = torch.tensor(data['x'], dtype=torch.float32)
                y = torch.tensor(data['y'], dtype=torch.long)
                
                # 创建数据集和数据加载器
                dataset = TensorDataset(x, y)
                drop_last = True if is_train else False
                dataloader = DataLoader(
                    dataset, 
                    batch_size=batch_size, 
                    shuffle=True,
                    drop_last=drop_last
                )
                
                return dataloader
            
        except Exception as e:
            print(f"Error loading drift data for client {self.id}: {str(e)}")
            # 回退到默认数据加载
            if is_train:
                train_data = read_client_data(self.dataset, self.id, is_train=True)
                return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)
            else:
                test_data = read_client_data(self.dataset, self.id, is_train=False)
                return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average="micro")

        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y

    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(
            item,
            os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"),
        )

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(
            os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt")
        )

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
