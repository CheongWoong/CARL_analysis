env_id=CARLPendulum
method=$1
train_env_config_id=$2
device_id=$3

for test_idx in {1..100}
do
    bash "scripts/test/"$method".sh" $env_id $train_env_config_id "test_"$test_idx $device_id
done