env_id=CARLDmcWalkerEnv
method=stacked
device_id=1

for test_idx in {7..12}
do
    bash "scripts/training/"$method".sh" $env_id "test_"$test_idx $device_id
done