env_id=$1
env_config_id=$2
device_id=$3

OMP_NUM_THREADS=1 PYTHONPATH=. python src/train_td3.py \
    --env_id $env_id \
    --env_config_id $env_config_id \
    --device_id $device_id \
    --n_contexts 10000 \
    --len_history 0 \
    --context_objective none \
    --exp_name oracle \
    --use_gt_context