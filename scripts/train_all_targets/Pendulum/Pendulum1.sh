env_id=CARLPendulum
method=stacked
device_id=0

for test_idx in {1..6}
do
    bash "scripts/training/"$method".sh" $env_id "test_"$test_idx $device_id
done