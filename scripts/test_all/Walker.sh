env_id=CARLDmcWalkerEnv
method=$1
train_env_config_id=$2
checkpoint_idx=$3
device_id=$4

for test_idx in {1..25}
do
    bash "scripts/test/"$method".sh" $env_id $train_env_config_id "test_"$test_idx $checkpoint_idx $device_id
done