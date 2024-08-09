env_id=$1
train_env_config_id=$2
test_env_config_id=$3
device_id=$4
training_seed=1

OMP_NUM_THREADS=1 PYTHONPATH=. python src/test_td3.py \
    --env_id $env_id \
    --env_config_id $test_env_config_id \
    --device_id $device_id \
    --n_contexts 10000 \
    --len_history 5 \
    --context_objective dm \
    --checkpoint_dir "runs/training/seed_"$training_seed"/"$env_id"/"$train_env_config_id"/dm" \
    --exp_name $train_env_config_id"/dm"