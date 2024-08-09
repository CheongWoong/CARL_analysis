env_id=CARLDmcWalkerEnv
method=stacked
device_id=3

for test_idx in {19..25}
do
    bash "scripts/training/"$method".sh" $env_id "test_"$test_idx $device_id
done