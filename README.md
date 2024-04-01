# CARL_analysis

## Training
Execute the commands below to run the training scripts.
```
# (Example) bash scripts/training/vanilla.sh CARLPendulum 0
bash scripts/training/{method_name}.sh {env_id} {gpu_id}
```

## Test
Execute the commands below to run the evaluation scripts.
```
# (Example) bash scripts/test/vanilla.sh CARLPendulum 0 0
bash scripts/test/{method_name}.sh {env_id} {test_single_context_id} {gpu_id}
```
