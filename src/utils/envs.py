import gymnasium as gym

from carl.envs import CARLMountainCarContinuous, CARLPendulum, \
    CARLBipedalWalker, \
    CARLBraxAnt, CARLBraxHalfcheetah, CARLBraxHopper, CARLBraxHumanoid, CARLBraxHumanoidStandup, \
    CARLBraxInvertedDoublePendulum, CARLBraxInvertedPendulum, CARLBraxPusher, CARLBraxReacher, CARLBraxWalker2d, \
    CARLDmcFingerEnv, CARLDmcFishEnv, CARLDmcQuadrupedEnv, CARLDmcWalkerEnv

from carl.context.context_space import UniformFloatContextFeature, CategoricalContextFeature
from carl.context.sampler import ContextSampler

from src.utils.history_wrapper import HistoryWrapperObsDict


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


def make_carl_env(env_id, seed, idx, capture_video, run_name, context_configs, n_contexts, len_history):
    def thunk():
        contexts, obs_context_features = None, None
        if context_configs is not None and n_contexts > 0:
            context_distributions = []
            obs_context_features = []
            for context_config in context_configs:
                if context_config['context_space_type'] == 'uniform': # name, lower, upper
                    context_distribution = UniformFloatContextFeature(**context_config["context_arguments"])
                elif context_config['context_space_type'] == 'categorical': # name, choices
                    context_distribution = CategoricalContextFeature(**context_config["context_arguments"])
                else:
                    raise NotImplementedError
                context_distributions.append(context_distribution)
                obs_context_features.append(context_config["context_arguments"]["name"])

            context_sampler = ContextSampler(
                context_distributions=context_distributions,
                context_space=eval(env_id).get_context_space(),
                seed=42,
            )
            contexts = context_sampler.sample_contexts(n_contexts=n_contexts)

            temp_contexts = {}
            temp_values = {}
            for context_id, context in contexts.items():
                if str(context) not in temp_values:
                    key = len(temp_contexts)
                    temp_contexts[key] = context
                    temp_values[str(context)] = 0

            contexts = temp_contexts
            print(len(temp_contexts))

            print("Training contexts are (showing up to 10):")
            print(list(contexts.items())[:10])

        if capture_video and idx == 0:
            env = gym.make(f"carl/{env_id}-v0", render_mode="rgb_array", contexts=contexts, obs_context_features=obs_context_features, obs_context_as_dict=False, disable_env_checker=True)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(f"carl/{env_id}-v0", contexts=contexts, obs_context_features=obs_context_features, obs_context_as_dict=False, disable_env_checker=True)

        if len_history > 0:
            env = HistoryWrapperObsDict(env, horizon=len_history)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    
    return thunk