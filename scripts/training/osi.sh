env_id=$1
device_id=$2

OMP_NUM_THREADS=1 PYTHONPATH=. python src/train_td3.py \
    --env_id $env_id \
    --env_config_id train \
    --device_id $device_id \
    --n_contexts 1000 \
    --test_single_context_id -1 \
    --len_history 5 \
    --context_objective osi \
    --exp_name osi