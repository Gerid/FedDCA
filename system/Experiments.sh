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
        ["--ablation_lp_type"]="feature_based" # 'feature_based' 或 'class_counts'
        ["--ablation_no_lp"]="PRESENCE_ONLY_FALSE" # 默认: ablation_no_lp is False
        ["--dca_ot_reg"]="0.1"
        ["--dca_vwc_reg"]="0.1"
        ["--dca_target_num_clusters"]="3"
        ["--clustering_method"]="label_conditional"
        ["-go"]="feddca_study" # 实验目标/标签

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

# --- 消融研究: 标签概要 (LP) ---
echo "开始消融研究: 标签概要 (LP)"
WANDB_GROUP_LP_ABLATION="LP_Ablation"
# 1. 基线 (基于特征的 LP) - ablation_no_lp is False
run_experiment "$WANDB_GROUP_LP_ABLATION" "FeatureBasedLP" \
    "--ablation_no_lp" "PRESENCE_ONLY_FALSE" \
    "--ablation_lp_type" "feature_based" \
    "-go" "LP_Ablation_FB"

# 2. 无标签概要 - ablation_no_lp is True
run_experiment "$WANDB_GROUP_LP_ABLATION" "NoLP" \
    "--ablation_no_lp" "PRESENCE_ONLY_TRUE" \
    "-go" "LP_Ablation_NoLP"

# 3. 基于类别计数的标签概要 - ablation_no_lp is False
run_experiment "$WANDB_GROUP_LP_ABLATION" "ClassCountLP" \
    "--ablation_no_lp" "PRESENCE_ONLY_FALSE" \
    "--ablation_lp_type" "class_counts" \
    "-go" "LP_Ablation_CC"
echo "完成消融研究: 标签概要 (LP)"
echo "====================================================="

# --- 参数影响研究: num_profile_samples ---
echo "开始参数影响研究: num_profile_samples"
WANDB_GROUP_NPS="Impact_NumProfileSamples"
for nps_val in 10 20 30 50; do
    run_experiment "$WANDB_GROUP_NPS" "NPS_${nps_val}" \
        "--num_profile_samples" "$nps_val" \
        "-go" "ImpactNPS_${nps_val}"
done
echo "完成参数影响研究: num_profile_samples"
echo "====================================================="

# --- 参数影响研究: dca_ot_reg (OT 正则化) ---
echo "开始参数影响研究: dca_ot_reg"
WANDB_GROUP_OT_REG="Impact_OTReg"
for ot_reg_val in 0.01 0.05 0.1 0.5; do
    run_experiment "$WANDB_GROUP_OT_REG" "OTReg_${ot_reg_val//./p}" \
        "--dca_ot_reg" "$ot_reg_val" \
        "-go" "ImpactOTReg_${ot_reg_val//./p}"
done
echo "完成参数影响研究: dca_ot_reg"
echo "====================================================="

# --- 参数影响研究: dca_vwc_reg (VWC 正则化) ---
echo "开始参数影响研究: dca_vwc_reg"
WANDB_GROUP_VWC_REG="Impact_VWCReg"
for vwc_reg_val in 0.01 0.05 0.1 0.2; do
    run_experiment "$WANDB_GROUP_VWC_REG" "VWCReg_${vwc_reg_val//./p}" \
        "--dca_vwc_reg" "$vwc_reg_val" \
        "-go" "ImpactVWCReg_${vwc_reg_val//./p}"
done
echo "完成参数影响研究: dca_vwc_reg"
echo "====================================================="

# --- 参数影响研究: dca_target_num_clusters ---
echo "开始参数影响研究: dca_target_num_clusters"
WANDB_GROUP_NUM_CLUSTERS="Impact_NumClusters"
NUM_CLIENTS_VAL=20 

NUM_CLUSTERS_VALUES=(2 3 5) 
for num_clusters_val in "${NUM_CLUSTERS_VALUES[@]}"; do 
    if [ "$num_clusters_val" -gt "$NUM_CLIENTS_VAL" ] && [ "$NUM_CLIENTS_VAL" -gt 0 ]; then
        echo "跳过 num_clusters=$num_clusters_val 因为它大于客户端总数 NUM_CLIENTS=$NUM_CLIENTS_VAL"
        continue
    fi
    run_experiment "$WANDB_GROUP_NUM_CLUSTERS" "Clusters_${num_clusters_val}" \
        "--dca_target_num_clusters" "$num_clusters_val" \
        "-go" "ImpactNumClusters_${num_clusters_val}"
done
echo "完成参数影响研究: dca_target_num_clusters"
echo "====================================================="

# --- 运行用户提供的示例配置 (稍作调整以适应脚本) ---
echo "运行用户示例配置"
WANDB_GROUP_USER_EXAMPLE="UserExample_FromCmd"
run_experiment "$WANDB_GROUP_USER_EXAMPLE" "Cifar100_DriftDataset_Cmd" \
    "-data" "Cifar100" \
    "-m" "cnn" \
    "-algo" "FedDCA" \
    "-gr" "200" \
    "-ls" "1" \
    "-lbs" "128" \
    "-lr" "0.05" \
    "-did" "0" \
    "-nb" "100" \
    "-eg" "10" \
    "-nc" "20" \
    "--clustering_method" "label_conditional" \
    "--use_drift_dataset" "PRESENCE_ONLY_TRUE" \
    "-go" "cnn_drift_example" # 修改了go参数以更具描述性
echo "完成用户示例配置"
echo "====================================================="

echo "所有研究脚本已启动。请监控 '$OUTPUT_DIR' 目录下的 .out 文件和 Wandb 仪表板 (如果启用)。"
echo "等待所有后台作业完成..."
wait
echo "所有作业已完成。"
