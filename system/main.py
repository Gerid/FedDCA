#!/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
import yaml # Added YAML import

import wandb # Added for Weights & Biases

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverpFedMe import pFedMe
from flcore.servers.serverperavg import PerAvg
from flcore.servers.serverprox import FedProx
from flcore.servers.serverfomo import FedFomo
from flcore.servers.serveramp import FedAMP
from flcore.servers.servermtl import FedMTL
from flcore.servers.serverlocal import Local
from flcore.servers.serverper import FedPer
from flcore.servers.serverapfl import APFL
from flcore.servers.serverditto import Ditto
from flcore.servers.serverrep import FedRep
from flcore.servers.serverphp import FedPHP
from flcore.servers.serverbn import FedBN
from flcore.servers.serverrod import FedROD
from flcore.servers.serverproto import FedProto
from flcore.servers.serverdyn import FedDyn
from flcore.servers.servermoon import MOON
from flcore.servers.serverbabu import FedBABU
from flcore.servers.serverapple import APPLE
from flcore.servers.servergen import FedGen
from flcore.servers.serverscaffold import SCAFFOLD
from flcore.servers.serverdistill import FedDistill
from flcore.servers.serverala import FedALA
#from flcore.servers.serverpac import FedPAC
from flcore.servers.serverlg import LG_FedAvg
from flcore.servers.servergc import FedGC
from flcore.servers.serverfml import FML
from flcore.servers.serverkd import FedKD
from flcore.servers.serverpcl import FedPCL
from flcore.servers.servercp import FedCP
from flcore.servers.servergpfl import GPFL
from flcore.servers.serverdca import FedDCA
from flcore.servers.serverifca import FedIFCA
from flcore.servers.serverfedccfa import FedCCFA
from flcore.servers.serverfeddrift import FedDrift
from flcore.servers.serverflash import Flash
from flcore.servers.serverfedrc import serverFedRC # Add FedRC server import

from flcore.trainmodel.models import *

from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.transformer import *

from utils.result_utils import average_data
from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)

# hyper-params for Text tasks
vocab_size = 98635
max_len=200
emb_dim=32

def run(args):

    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Initialize Weights & Biases if enabled
        if args.use_wandb:
            run_name = f"{args.wandb_run_name_prefix}_{args.algorithm}_{args.dataset}_run{i}"
            if args.use_drift_dataset:
                run_name += "_drift"
            
            wandb_config = vars(args).copy()
            wandb.login(key=args.wandb_api_key)  # Ensure API key is set before initializing
            # Remove sensitive or non-serializable args if necessary
            if 'wandb_api_key' in wandb_config:
                del wandb_config['wandb_api_key']

            try:
                wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=run_name,
                    config=wandb_config,
                    settings=wandb.Settings(
                        start_method="thread",
                        anonymous="never"  # Explicitly set anonymous policy
                    )
                )
                print(f"Wandb initialized for run: {run_name}")
            except Exception as e:
                print(f"Error initializing wandb: {e}")
                print("Proceeding without wandb.")
                args.use_wandb = False # Disable wandb if init fails


        # Generate args.model
        if model_str == "mlr": # convex
            if "mnist" in args.dataset:
                args.model = Mclr_Logistic(1*28*28, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = Mclr_Logistic(3*32*32, num_classes=args.num_classes).to(args.device)
            else:
                args.model = Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)

        elif model_str == "cnn": # non-convex
            if "mnist" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "omniglot" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
                # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
            elif "Digit5" in args.dataset:
                args.model = Digit5CNN().to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

        elif model_str == "dnn": # non-convex
            if "mnist" in args.dataset:
                args.model = DNN(1*28*28, 100, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = DNN(3*32*32, 100, num_classes=args.num_classes).to(args.device)
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)
        
        elif model_str == "resnet":
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)
            
            # args.model = torchvision.models.resnet18(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
            
            # args.model = resnet18(num_classes=args.num_classes, has_bn=True, bn_block_num=4).to(args.device)

        elif model_str == "alexnet":
            args.model = alexnet(pretrained=False, num_classes=args.num_classes).to(args.device)
            
            # args.model = alexnet(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
            
        elif model_str == "googlenet":
            args.model = torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes).to(args.device)
            
            # args.model = torchvision.models.googlenet(pretrained=True, aux_logits=False).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "mobilenet_v2":
            args.model = mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)
            
            # args.model = mobilenet_v2(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
            
        elif model_str == "lstm":
            args.model = LSTMNet(hidden_dim=emb_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(args.device)

        elif model_str == "bilstm":
            args.model = BiLSTM_TextClassification(input_size=vocab_size, hidden_size=emb_dim, output_size=args.num_classes, 
                        num_layers=1, embedding_dropout=0, lstm_dropout=0, attention_dropout=0, 
                        embedding_length=emb_dim).to(args.device)

        elif model_str == "fastText":
            args.model = fastText(hidden_dim=emb_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(args.device)

        elif model_str == "TextCNN":
            args.model = TextCNN(hidden_dim=emb_dim, max_len=max_len, vocab_size=vocab_size, 
                            num_classes=args.num_classes).to(args.device)

        elif model_str == "Transformer":
            args.model = TransformerModel(ntoken=vocab_size, d_model=emb_dim, nhead=8, d_hid=emb_dim, nlayers=2, 
                            num_classes=args.num_classes).to(args.device)
        
        elif model_str == "AmazonMLP":
            args.model = AmazonMLP().to(args.device)

        elif model_str == "harcnn":
            if args.dataset == 'har':
                args.model = HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, conv_kernel_size=(1, 9), pool_kernel_size=(1, 2)).to(args.device)
            elif args.dataset == 'pamap':
                args.model = HARCNN(9, dim_hidden=3712, num_classes=args.num_classes, conv_kernel_size=(1, 9), pool_kernel_size=(1, 2)).to(args.device)

        else:
            raise NotImplementedError

        print(args.model)

        # select algorithm
        if args.algorithm == "FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i)

        elif args.algorithm == "Local":
            server = Local(args, i)

        elif args.algorithm == "FedMTL":
            server = FedMTL(args, i)

        elif args.algorithm == "PerAvg":
            server = PerAvg(args, i)

        elif args.algorithm == "pFedMe":
            server = pFedMe(args, i)

        elif args.algorithm == "FedProx":
            server = FedProx(args, i)

        elif args.algorithm == "FedFomo":
            server = FedFomo(args, i)

        elif args.algorithm == "FedAMP":
            server = FedAMP(args, i)

        elif args.algorithm == "APFL":
            server = APFL(args, i)

        elif args.algorithm == "FedPer":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPer(args, i)

        elif args.algorithm == "Ditto":
            server = Ditto(args, i)

        elif args.algorithm == "FedRep":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedRep(args, i)

        elif args.algorithm == "FedPHP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPHP(args, i)

        elif args.algorithm == "FedBN":
            server = FedBN(args, i)

        elif args.algorithm == "FedROD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedROD(args, i)

        elif args.algorithm == "FedProto":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedProto(args, i)

        elif args.algorithm == "FedDyn":
            server = FedDyn(args, i)

        elif args.algorithm == "MOON":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = MOON(args, i)

        elif args.algorithm == "FedBABU":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedBABU(args, i)

        elif args.algorithm == "APPLE":
            server = APPLE(args, i)

        elif args.algorithm == "FedGen":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGen(args, i)

        elif args.algorithm == "SCAFFOLD":
            server = SCAFFOLD(args, i)

        elif args.algorithm == "FedDistill":
            server = FedDistill(args, i)

        elif args.algorithm == "FedALA":
            server = FedALA(args, i)

        #elif args.algorithm == "FedPAC":
            #args.head = copy.deepcopy(args.model.fc)
            #args.model.fc = nn.Identity()
            #args.model = BaseHeadSplit(args.model, args.head)
            #server = FedPAC(args, i)

        elif args.algorithm == "LG-FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = LG_FedAvg(args, i)

        elif args.algorithm == "FedGC":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGC(args, i)

        elif args.algorithm == "FML":
            server = FML(args, i)

        elif args.algorithm == "FedKD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedKD(args, i)

        elif args.algorithm == "FedPCL":
            args.model.fc = nn.Identity()
            server = FedPCL(args, i)        
        elif args.algorithm == "FedCP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedCP(args, i)
            
        elif args.algorithm == "GPFL":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = GPFL(args, i)
            
        elif args.algorithm == "FedDCA":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head) # model is now BaseHeadSplit
            server = FedDCA(args, i)

            # Determine classifier keys from the 'head' part of BaseHeadSplit
            clf_keys = []
            if hasattr(args.model, 'head') and args.model.head is not None:
                for key, _ in args.model.head.named_parameters():
                    # Parameters in the state_dict of the full model (args.model)
                    # are prefixed with 'head.' if head is an nn.Module attribute of BaseHeadSplit.
                    clf_keys.append('head.' + key)
            
            if not clf_keys:
                raise ValueError("Classifier keys (clf_keys) could not be determined for FedDCA. Ensure model.head is properly defined in BaseHeadSplit and has parameters.")

            server.set_clf_keys(clf_keys) # Add this method to FedDCA server
            
        elif args.algorithm == "FedIFCA":
            args.head = copy.deepcopy(args.model.fc)            
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedIFCA(args, i)
        elif args.algorithm == "FedCCFA":
            # Default FedCCFA specific parameters are now mostly set in the FedCCFA server __init__
            # or have been set via argparse with defaults from FedCCFA.yaml.
            # We still need to ensure the model is wrapped for feature extraction
            # and that clf_keys are determined and set for both server and clients.

            # Determine classifier keys (e.g., ['classifier.weight', 'classifier.bias'])
            # This needs to be done AFTER the model is potentially wrapped.
            clf_keys = list(args.model.state_dict().keys())[-2:]

            if not clf_keys:
                raise ValueError("Could not determine classifier keys for the model. Ensure model has a 'classifier' attribute.")

            # Instantiate the server
            server = FedCCFA(args, i)
            
            # Set classifier keys in the server
            server.set_clf_keys(clf_keys)

            # Set classifier keys for each client (after clients are created by the server)
            # The server's __init__ calls self.set_clients(), so clients are available now.
            for client in server.clients:
                if hasattr(client, 'set_clf_keys'):
                    client.set_clf_keys(clf_keys)
                else:
                    # This is a fallback, ideally clientFedCCFA should have this method
                    # print(f"Warning: Client {client.id} does not have set_clf_keys method. Setting clf_keys directly.")
                    client.clf_keys = clf_keys
            
            print(f"FedCCFA server and clients configured with clf_keys: {clf_keys}")

        elif args.algorithm == "FedDrift":
            # 为FedDrift设置特定参数
            if not hasattr(args, 'detection_threshold'):
                args.detection_threshold = 0.5  # 概念漂移检测阈值
            if not hasattr(args, 'visualize_clusters'):
                args.visualize_clusters = False  # 是否可视化集群
                
            server = FedDrift(args, i)
        elif args.algorithm == "Flash":
            # 为Flash设置特定参数
            if not hasattr(args, 'loss_decrement'):
                args.loss_decrement = 0.01  # 早停损失下降阈值
            if not hasattr(args, 'beta1'):
                args.beta1 = 0.9  # 一阶动量系数
            if not hasattr(args, 'beta2'):
                args.beta2 = 0.99  # 二阶动量系数
            if not hasattr(args, 'ftau'):
                args.ftau = 1e-8  # 数值稳定常数
            if not hasattr(args, 'server_learning_rate'):
                args.server_learning_rate = 0.01 # 服务器学习率
            if not hasattr(args, 'verbose'):
                args.verbose = 1.0  # 服务器学习率
                
            server = Flash(args, i)

        elif args.algorithm == "FedRC": # Add FedRC algorithm block
            # FedRC specific parameters will be loaded from YAML or command line
            # Ensure num_clusters, cluster_update_frequency, fedrc_lambda are in args
            if not hasattr(args, 'num_clusters'): # Default if not in YAML/CLI
                args.num_clusters = 3 
            if not hasattr(args, 'cluster_update_frequency'):
                args.cluster_update_frequency = 5
            if not hasattr(args, 'fedrc_lambda'):
                args.fedrc_lambda = 0.1
            server = serverFedRC(args, i)
            
        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time()-start)

        if args.use_wandb:
            try:
                wandb.finish()
                print(f"Wandb run {run_name} finished.")
            except Exception as e:
                print(f"Error finishing wandb run: {e}")


    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    

    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # Add an argument for the config file
    parser.add_argument('-cfg', "--config_file", type=str, default="../config.yaml", 
                        help="Path to the YAML configuration file")
    # general
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="mnist")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=200)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=2,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-worker', "--num_workers", type=int, default=0,
                        help="Number of workers for DataLoader")
    parser.add_argument('-pm', "--pin_memory", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable pin_memory for DataLoader")
    parser.add_argument('--use_amp', action=argparse.BooleanOptionalAction, default=False,
                        help="Enable Automatic Mixed Precision training")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-dp', "--privacy", type=bool, default=False,
                        help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-fte', "--fine_tuning_epoch", type=int, default=0)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP
    parser.add_argument('-bt', "--beta", type=float, default=0.0,
                        help="Average moving parameter for pFedMe, Second learning rate of Per-FedAvg, \
                        or L1 regularization weight of FedTransfer")
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0,
                        help="Proximal rate for FedProx")
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=5,
                        help="Server only sends M client models to one client at each round")
    # FedMTL
    parser.add_argument('-itk', "--itk", type=int, default=4000,
                        help="The iterations for solving quadratic subproblems")
    # FedAMP
    parser.add_argument('-alk', "--alphaK", type=float, default=1.0, 
                        help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    # APFL
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_steps", type=int, default=1)
    # MOON
    parser.add_argument('-tau', "--tau", type=float, default=1.0)
    # FedBABU
    parser.add_argument('-fts', "--fine_tuning_steps", type=int, default=10)
    # APPLE
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)
    # FedGen
    parser.add_argument('-nd', "--noise_dim", type=int, default=512)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
    parser.add_argument('-se', "--server_epochs", type=int, default=1000)
    parser.add_argument('-lf', "--localize_feature_extractor", type=bool, default=False)
    # SCAFFOLD
    # parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)
    # FedALA
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                        help="More fine-graind than its original paper.")
    # FedKD
    parser.add_argument('-mlr', "--mentee_learning_rate", type=float, default=0.005)
    parser.add_argument('-Ts', "--T_start", type=float, default=0.95)
    parser.add_argument('-Te', "--T_end", type=float, default=0.98)
    # GPFL
    parser.add_argument('-lamr', "--lamda_reg", type=float, default=0.0)    # FedDCA
    parser.add_argument('-encpath', "--autoencoder_model_path", type=str, default="/enc_path")
    parser.add_argument('-ncs', "--num_clusters", type=int, default=5)
    parser.add_argument('-st', "--split_threshold", type=float, default=0.3)
    parser.add_argument('-mt', "--merge_threshold", type=float, default=0.05)
    parser.add_argument('-gs', "--gmm_samples", type=int, default=100)
    parser.add_argument('-cm', "--clustering_method", type=str, default="enhanced_label",
                        choices=["vwc", "label_conditional", "enhanced_label"],
                       help="聚类方法: vwc (原始变分Wasserstein聚类) 或 label_conditional (基于标签的条件Wasserstein聚类)")
    # FedCCFA
    parser.add_argument('-cle', "--clf_epochs", type=int, default=1, 
                        help="FedCCFA分类器训练轮数") # YAML: 1
    parser.add_argument('-rpe', "--rep_epochs", type=int, default=5, 
                        help="FedCCFA表示层训练轮数") # YAML: 5
    parser.add_argument('-be', "--balanced_epochs", type=int, default=5, 
                        help="FedCCFA平衡训练轮数") # YAML: 5
    parser.add_argument('-lp', "--lambda_proto", type=float, default=0.0, 
                        help="FedCCFA原型损失权重") # YAML: 0.0
    parser.add_argument('-eps', "--eps", type=float, default=0.1, 
                        help="FedCCFA DBSCAN聚类的eps参数") # YAML: 0.1
    parser.add_argument('-wts', "--weights", type=str, default="uniform",
                        choices=["uniform", "label"], 
                        help="FedCCFA聚合权重方式 (uniform或label)") # YAML: uniform
    parser.add_argument('-balr', "--balanced_clf_lr", type=float, default=0.1,
                        help="FedCCFA clf learning rate ") # YAML: uniform
    parser.add_argument('-pnz', "--penalize", type=str, default="contrastive",
                        choices=["L2", "contrastive"], 
                        help="FedCCFA原型损失类型 (L2或contrastive)") # YAML: CL (contrastive)
    parser.add_argument('-tmp', "--temperature", type=float, default=0.1,
                        help="FedCCFA对比学习温度参数") # YAML: 0.1
    parser.add_argument('-gm', "--gamma", type=float, default=20.0,
                        help="FedCCFA自适应原型权重参数,0表示使用固定权重") # YAML: 20.0
    parser.add_argument('-orc', "--oracle", action='store_true', default=False,
                        help="FedCCFA使用Oracle合并策略") # YAML: false
    parser.add_argument('-cp', "--clustered_protos", action='store_true', default=True,
                        help="FedCCFA使用聚类原型") # YAML: true
    parser.add_argument('-ci', "--cluster_interval", type=int, default=5, 
                       help="执行聚类的轮次间隔")
    # parser.add_argument('-vc', "--visualize_clusters", type=bool, default=True, # This was a bool, action='store_true' is better for flags
    #                    help="是否可视化聚类结果")
    parser.add_argument("--visualize_clusters", action='store_true',# This was a bool, action='store_true' is better for flags
                       help="是否可视化聚类结果")

    parser.add_argument('-vi', "--vis_interval", type=int, default=10,
                       help="聚类可视化的轮次间隔")
    parser.add_argument('-vb', "--verbose", type=bool, default=True, # This was a bool, action='store_true' is better for flags
                       help="是否输出详细信息")    # 概念漂移数据集参数
    parser.add_argument('--use_drift_dataset', action='store_true', 
                        help='使用概念漂移数据集')

    parser.add_argument('--simulate_drift', action='store_true', 
                        help='使用标准数据集，但是进行模拟漂移')
    parser.add_argument('--drift_data_dir', type=str, default='../dataset/Cifar100_clustered/', 
                        help='概念漂移数据集目录')
    parser.add_argument('--max_iterations', type=int, default=200, 
                        help='概念漂移数据集的最大迭代数')
    parser.add_argument('--drift_round', type=int, default=100,
                        help='概念漂移发生的轮次，默认为第100轮')
                        
    # Flash算法特有参数
    parser.add_argument('--loss_decrement', type=float, default=0.01,
                        help='Flash早停损失下降阈值')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Flash一阶动量系数')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Flash二阶动量系数')
    parser.add_argument('--ftau', type=float, default=1e-8,
                        help='Flash数值稳定常数')

    # Weights & Biases arguments
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default="FedDCA", help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default="Gerid", help='Wandb entity (username or team)') # User should set this
    parser.add_argument('--wandb_api_key', type=str, default="47b2a9d806ce037a46ec07b05d6f211af19728a3",
                        help='Wandb API key (optional, can be set as env var)')
    parser.add_argument('--wandb_run_name_prefix', type=str, default="exp", help='Prefix for wandb run names')
    parser.add_argument('--save_global_model_to_wandb', action='store_true', help='Save global model to Wandb Artifacts')
    parser.add_argument('--save_results_to_wandb', action='store_true', help='Save H5 results to Wandb Artifacts')

    # FedDCA specific arguments
    parser.add_argument('-client_profile_noise_stddev', "--profile_noise_stddev", type=float, default=0.01, help="Stddev of Gaussian noise for client profiles")

    # Ablation study parameters for FedDCA
    parser.add_argument('--ablation_no_lp', action='store_true', help="FedDCA-NoLP: Disable label profiles. Clients send no LPs, server performs no LP-based drift detection or clustering.")
    parser.add_argument('--ablation_no_drift_detect', action='store_true', help="FedDCA-NoDriftDetect: Disable explicit drift detection mechanism. Clustering might still occur based on LPs if available, but not dynamically triggered by detected drift.")
    parser.add_argument('--ablation_no_clustering', action='store_true', help="FedDCA-NoCluster: Disable client clustering. A single global classifier is maintained.")
    parser.add_argument('--ablation_lp_type', type=str, default="feature_based", choices=['feature_based', 'class_counts'], help="FedDCA-SimpleLP: Type of label profile to use ('feature_based' or 'class_counts').")
    # Add other ablation flags as needed, e.g.:
    # parser.add_argument('--ablation_global_fe_only', action='store_true', help="Use only a global feature extractor, no personalization even if clustered.")


    args = parser.parse_args()

    # Load YAML config
    config_path = args.config_file
    if not os.path.isabs(config_path):
        # Assuming main.py is in system/ and config.yaml is in the root PFL-Non-IID/
        # Adjust if your structure is different or if you run main.py from the root.
        # If main.py is run from d:\repos\PFL-Non-IID\system\, then ../config.yaml is correct.
        # If main.py is run from d:\repos\PFL-Non-IID\, then config.yaml would be correct.
        # For now, let's assume it's relative to the script's directory's parent.
        script_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(os.path.dirname(script_dir), args.config_file) 
        # This makes it robust to where you run python from, assuming config is in root
        # If config.yaml is in the same directory as main.py, use:
        # config_path = os.path.join(script_dir, args.config_file)

    try:
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: YAML config file not found at {config_path}. Using default argparse values.")
        yaml_config = {}
    except Exception as e:
        print(f"Error loading YAML config file: {e}. Using default argparse values.")
        yaml_config = {}

    # Merge YAML config with argparse defaults
    # Argparse values take precedence if provided explicitly on the command line
    # For common parameters, load from YAML first, then let argparse override if specified
    
    # Create a new namespace or update args directly.
    # We will update args, letting command-line args override YAML.
    
    # Load common config
    common_config = yaml_config.get('common', {})
    for key, value in common_config.items():
        if not hasattr(args, key) or getattr(args, key) == parser.get_default(key):
            setattr(args, key, value)

    # Load algorithm-specific config
    # The args.algorithm is from argparse (command line or its default)
    algo_config = yaml_config.get(args.algorithm.lower(), {})
    for key, value in algo_config.items():
        if not hasattr(args, key) or getattr(args, key) == parser.get_default(key):
            setattr(args, key, value)

    # Override with command-line arguments if they were changed from their defaults
    # This is implicitly handled because argparse already parsed them into args.
    # If a command-line arg was given, it's already in 'args' and won't be overwritten by YAML
    # unless the YAML loading logic above is changed to always prefer YAML.
    # The current logic: YAML sets if argparse is default. If argparse is not default, it stays.

    # wandb specific args from YAML (if not overridden by command line)
    wandb_yaml_config = yaml_config.get('wandb', {})
    for key, value in wandb_yaml_config.items():
        if not hasattr(args, key) or getattr(args, key) == parser.get_default(key):
            setattr(args, key, value)


    # Ensure device is set correctly
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU.")
        args.device = "cpu"
    elif args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
        print(f"Using CUDA device: {args.device_id}")
    else:
        print("Using CPU.")

    print("Hyperparameters:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    run(args)


    print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")
