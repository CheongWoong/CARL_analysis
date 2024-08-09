env_id=CARLPendulum
method=stacked
device_id=$1

for test_idx in {1..25}
do
    bash "scripts/test/"$method".sh" $env_id "test_"$test_idx "test_"$test_idx $device_id
done