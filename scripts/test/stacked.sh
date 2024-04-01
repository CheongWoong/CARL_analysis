env_id=$1
test_single_context_id=$2
device_id=$3
training_seed=1

OMP_NUM_THREADS=1 PYTHONPATH=. python src/test_td3.py \
    --env_id $env_id \
    --env_config_id test \
    --device_id $device_id \
    --n_contexts 1 \
    --test_single_context_id $test_single_context_id \
    --len_history 5 \
    --context_objective none \
    --checkpoint_dir "runs/training/seed_"$training_seed"/"$env_id"/train/stacked" \
    --exp_name stacked