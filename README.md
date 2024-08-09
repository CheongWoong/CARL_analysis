# CARL_analysis

## Training
Execute the commands below to run the training scripts.
```
# (Example) bash scripts/training/stacked.sh CARLPendulum train 0
bash scripts/training/{method_name}.sh {env_id} {env_config_id} {gpu_id}
```

## Test
Execute the commands below to run the evaluation script for a single test env configuration.
```
# (Example) bash scripts/test/stacked.sh CARLPendulum train test_1 50000 0
bash scripts/test/{method_name}.sh {env_id} {train_env_config_id} {test_env_config_id} {checkpoint_idx} {gpu_id}
```
Or, you can run evaluation on all test env configurations.
```
# (Example) bash scripts/test_all/Pendulum.sh stacked train 50000 0
bash scripts/test_all/{env_id}.sh {method_name} {train_env_config_id} {checkpoint_idx} {gpu_id}
```

## Analysis
For analysis, refer to ipython notebooks [here](https://github.com/CheongWoong/CARL_analysis/tree/main/analysis/paper).