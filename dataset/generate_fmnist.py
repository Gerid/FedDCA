import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file


random.seed(1)
np.random.seed(1)
# num_clients = 20 # Will be set in main
# num_classes = 10 # Remains 10 for FashionMNIST
# dir_path = "fmnist/" # Will be set in main


# Allocate data to users
def generate_fmnist(dir_path, num_clients, num_classes, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    # if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    # Get FashionMNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.FashionMNIST(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)


if __name__ == "__main__":
    # Default settings
    num_clients_default = 100
    num_classes_default = 10 # FashionMNIST has 10 classes
    niid_default = True
    balance_default = False # Default to imbalanced
    partition_default = "dir" # Default partition type

    # Parse command line arguments
    # Expected order: [script_name] [niid_status (noniid/iid)] [balance_status (balanced/imbalanced)] [partition_type (e.g., dir)] [num_clients]
    
    is_niid = niid_default
    if len(sys.argv) > 1:
        arg1_lower = sys.argv[1].lower()
        if arg1_lower == "noniid":
            is_niid = True
        elif arg1_lower == "iid":
            is_niid = False

    is_balanced = balance_default
    if len(sys.argv) > 2:
        arg2_lower = sys.argv[2].lower()
        if arg2_lower == "balanced":
            is_balanced = True
        elif arg2_lower == "imbalanced":
            is_balanced = False
            
    partition_type = partition_default
    if len(sys.argv) > 3 and sys.argv[3] != "-":
        partition_type = sys.argv[3]

    num_clients_to_generate = num_clients_default
    if len(sys.argv) > 4 and sys.argv[4].isdigit():
        num_clients_to_generate = int(sys.argv[4])

    dynamic_dir_path = f"fmnist_{num_clients_to_generate}_clients/"

    print(f"--- Generating FashionMNIST dataset with settings ---")
    print(f"Data directory: {dynamic_dir_path}")
    print(f"Client count: {num_clients_to_generate}")
    print(f"Class count: {num_classes_default}")
    print(f"NIID: {is_niid}")
    print(f"Balanced: {is_balanced}")
    print(f"Partition type: {partition_type}")
    print(f"---------------------------------------------------")

    generate_fmnist(dynamic_dir_path, num_clients_to_generate, num_classes_default, is_niid, is_balanced, partition_type)