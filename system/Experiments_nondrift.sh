# 创建一个唯一的目录来存放输出文件
OUTPUT_DIR="feddca_direct_commands_output_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
echo "输出文件将保存在: $OUTPUT_DIR"
echo "请确保 main.py 接受所有指定的 FedDCA 特定参数和 Wandb 参数。"
echo "脚本将限制并发运行的 Python 实验数量。"
echo ""

# --- 并发控制 ---
MAX_CONCURRENT_JOBS=2
# Function to get current number of running python main.py jobs started by this script
get_running_jobs_count() {
    # pgrep -f "python -u main.py" -P $$ # This would be more precise if pgrep was guaranteed
    # Using `jobs -p` is simpler and usually sufficient if this script only launches these python jobs
    local count=$(jobs -p | wc -l)
    echo "$count"
}

# --- Wandb 设置 ---
WANDB_PROJECT="FedDCA_Impact_Studies_Cifar100"
# 如果不想使用 Wandb，设置为空字符串 ""
# USE_WANDB_FLAG="--use_wandb" # 启用 Wandb
USE_WANDB_FLAG="" # 禁用 Wandb (示例，根据需要修改)

# --- Helper 函数用于运行实验 ---
# 参数:
# 1. wandb_group_name: 用于在 wandb 中对运行进行分组
# 2. run_specific_name: 运行名称和输出文件名的唯一部分
# 3. param_overrides: 参数覆盖数组 (例如, ("--num_profile_samples" "10" "--dca_ot_reg" "0.05"))
run_experiment() {
    local wandb_group_name=$1
    local run_specific_name=$2
    shift 2 # 移除前两个参数
    local param_overrides=("$@")

    # 基础参数和 FedDCA 特定参数的默认值
    # 对于 action='store_true' 标志:
    # PRESENCE_ONLY_TRUE 表示包含该标志 (使其为 True)
    # PRESENCE_ONLY_FALSE 表示不包含该标志 (使其为 False, 即 argparse 的默认值)
    declare -A args_map=(
        ["-data"]="Cifar100"
        ["-m"]="cnn"
        ["-algo"]="FedDCA"
        ["-gr"]="200"
        ["-ls"]="1"
        ["-lbs"]="128"
        ["-lr"]="0.05"
        ["-did"]="0"
        ["-nb"]="100"
        ["-eg"]="10"
        ["-nc"]="20"
        ["-worker"]="0" # 示例：为 DataLoader 设置 num_workers
        ["-pm"]="PRESENCE_ONLY_TRUE" # pin_memory 默认为 True (BooleanOptionalAction)

        # FedDCA 特有参数的默认值
        ["--num_profile_samples"]="30"
        # ["--ablation_lp_type"]="feature_based" # 'feature_based' 或 'class_counts' # Commented out as it seems specific to an ablation not directly used in complex drift base config
        # ["--ablation_no_lp"]="PRESENCE_ONLY_FALSE" # 默认: ablation_no_lp is False # Commented out
        ["--dca_ot_reg"]="0.1"
        ["--dca_vwc_reg"]="0.1"
        ["--dca_target_num_clusters"]="3"
        ["--dca_vwc_K_t"]="5" # Default for VWC Kt
        ["--clustering_method"]="label_conditional"
        ["-go"]="feddca_study" # 实验目标/标签

        # --- Complex Concept Drift 参数默认值 ---
        # 这些通常会在调用 run_experiment 时被覆盖，但可以设置一个基础默认值（例如，不漂移）
        ["--complex_drift_scenario"]="None" # 默认不使用复杂漂移
        ["--drift_base_epoch"]="100"
        ["--drift_stagger_interval"]="10"
        ["--drift_partial_percentage"]="0.1"
        ["--superclass_map_path"]="../dataset/cifar100_superclass_map.json" # 假设的默认路径
        # --- End Complex Concept Drift 参数默认值 ---

        # 其他 action='store_true' 标志的默认状态 (False)
        ["--use_drift_dataset"]="PRESENCE_ONLY_FALSE"
        # --use_wandb 将由 USE_WANDB_FLAG 控制，这里不需要
    )

    # 应用参数覆盖
    local i=0
    while [ $i -lt ${#param_overrides[@]} ]; do
        local key="${param_overrides[$i]}"
        local value="${param_overrides[$((i+1))]}"
        args_map["$key"]="$value"
        i=$((i+2))
    done

    # 构建命令行参数
    local cmd_args=""
    for key in "${!args_map[@]}"; do
        local value="${args_map[$key]}"
        if [[ "$value" == "PRESENCE_ONLY_TRUE" ]]; then
            cmd_args="$cmd_args $key"
        elif [[ "$value" != "PRESENCE_ONLY_FALSE" ]]; then # 如果值为 PRESENCE_ONLY_FALSE 则跳过
             cmd_args="$cmd_args $key $value"
        fi
    done

    local full_run_name="${wandb_group_name}_${run_specific_name}"
    full_run_name=$(echo "$full_run_name" | sed 's/[^a-zA-Z0-9_.-]/_/g')
    
    local output_file="$OUTPUT_DIR/${full_run_name}.out"

    # 如果启用了 wandb，则添加 wandb 参数
    if [[ -n "$USE_WANDB_FLAG" ]]; then # $USE_WANDB_FLAG 包含 "--use_wandb"
        cmd_args="$cmd_args $USE_WANDB_FLAG --wandb_project $WANDB_PROJECT --wandb_group $wandb_group_name --wandb_run_name $full_run_name"
    fi

    local final_cmd="python -u main.py $cmd_args"

    # 并发控制：等待直到有可用的槽位
    while true; do
        local running_jobs
        running_jobs=$(get_running_jobs_count)
        if [ "$running_jobs" -lt "$MAX_CONCURRENT_JOBS" ]; then
            break
        else
            echo "已达到最大并发作业数 ($MAX_CONCURRENT_JOBS)。等待一个作业完成..."
            wait -n # 等待任何一个后台作业完成
        fi
    done

    echo "启动运行: $full_run_name"
    echo "命令: nohup $final_cmd > $output_file 2>&1 &"
    nohup $final_cmd > "$output_file" 2>&1 &
    echo "输出文件: $output_file, PID: $!"
    echo "-----------------------------------------------------"
    # sleep 1 # 短暂休眠以确保作业已注册，如果需要更精确的计数
}

# --- 无漂移基线实验 ---
echo "开始无漂移基线实验"

DATASETS_CONFIG=(
    "fmnist cnn 200 10"
    "Cifar10 cnn 200 10"
    "Cifar100 cnn 200 10"
    "har harcnn 500 6"
)
ALGORITHMS=("FedDCA" "FedIFCA" "FedCCFA" "FedRC" "FedDrift" "FedAvg" "Flash" "FedALA")

# 这些基线实验的通用参数
# 注意: -jr (join_ratio) 和 --save_..._to_wandb 标志是新添加的或需要确保存在的
COMMON_PARAMS=(
    "-lbs" "16"
    "-nc" "20"
    "-jr" "1" 
    "--use_drift_dataset" "PRESENCE_ONLY_FALSE" # 明确无漂移数据集
    "--complex_drift_scenario" "None"         # 明确无复杂漂移
    "--save_global_model_to_wandb" "PRESENCE_ONLY_TRUE" # 如果启用了 wandb，则保存模型
    "--save_results_to_wandb" "PRESENCE_ONLY_TRUE"    # 如果启用了 wandb，则保存结果
    # -eg 10 (run_experiment 默认)
    # -did 0 (run_experiment 默认)
    # -ls 1 (run_experiment 默认)
    # -lr 0.05 (run_experiment 默认)
)

for config_line in "${DATASETS_CONFIG[@]}"; do
    read -r dataset_name model_type gr_value nb_value<<< "$config_line"

    WANDB_GROUP_NO_DRIFT="NoDrift_${dataset_name}"

    for algo_name in "${ALGORITHMS[@]}"; do
        RUN_SPECIFIC_NAME="${algo_name}"
        # 为 wandb 和输出文件名创建一个描述性的 -go 参数值
        GO_PARAM_VALUE="NoDrift_${dataset_name}_${algo_name}"

        declare -a current_run_params=()
        current_run_params+=("-data" "$dataset_name")
        current_run_params+=("-m" "$model_type")
        current_run_params+=("-gr" "$gr_value")
        current_run_params+=("-nb" "$nb_value")
        current_run_params+=("-algo" "$algo_name")
        current_run_params+=("${COMMON_PARAMS[@]}")
        current_run_params+=("-go" "$GO_PARAM_VALUE") # 添加 -go 参数

        # FedDCA 的特定参数 (已在 run_experiment 中默认，但显式指定也可以)
        if [ "$algo_name" == "FedDCA" ]; then
            current_run_params+=("--clustering_method" "label_conditional")
        fi
        
        run_experiment "$WANDB_GROUP_NO_DRIFT" "$RUN_SPECIFIC_NAME" "${current_run_params[@]}"
    done
done

echo "完成无漂移基线实验"
echo "====================================================="


echo "所有研究脚本已启动。请监控 '$OUTPUT_DIR' 目录下的 .out 文件和 Wandb 仪表板 (如果启用)。"
echo "等待所有后台作业完成..."
wait
echo "所有作业已完成。"
