env_id=CARLDmcWalkerEnv
method=stacked
device_id=2

for test_idx in {13..18}
do
    bash "scripts/training/"$method".sh" $env_id "test_"$test_idx $device_id
done