# reference: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/wrappers.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class HistoryWrapperObsDict(gym.Wrapper):
    """
    History Wrapper for dict observation.

    :param env:
    :param horizon: Number of steps to keep in the history.
    """

    def __init__(self, env: gym.Env, horizon: int = 2):
        super().__init__(env)

        assert isinstance(env.observation_space, spaces.Dict)
        assert isinstance(env.observation_space.spaces["obs"], spaces.Box)
        assert isinstance(env.action_space, spaces.Box)

        wrapped_obs_space = env.observation_space.spaces["obs"]
        wrapped_action_space = env.action_space

        low_obs = np.tile(wrapped_obs_space.low, horizon)
        high_obs = np.tile(wrapped_obs_space.high, horizon)

        low_action = np.tile(wrapped_action_space.low, horizon)
        high_action = np.tile(wrapped_action_space.high, horizon)

        low = np.concatenate((low_obs, low_action))
        high = np.concatenate((high_obs, high_action))

        # Overwrite the observation space
        env.observation_space.spaces["history"] = spaces.Box(
            low=low,
            high=high,
            dtype=wrapped_obs_space.dtype,  # type: ignore[arg-type]
        )
        self.observation_space = env.observation_space

        self.horizon = horizon
        self.low_action, self.high_action = low_action, high_action
        self.low_obs, self.high_obs = low_obs, high_obs
        self.low, self.high = low, high
        self.obs_history = np.zeros(low_obs.shape, low_obs.dtype)
        self.action_history = np.zeros(low_action.shape, low_action.dtype)

    def _create_obs_from_history(self) -> np.ndarray:
        return np.concatenate((self.obs_history, self.action_history))

    def reset(self, **kwargs):
        # Flush the history
        self.obs_history[...] = 0
        self.action_history[...] = 0

        obs_dict, info = super().reset(**kwargs)
        obs = obs_dict["obs"]
        self.obs_history[..., -obs.shape[-1] :] = obs

        obs_dict.update({"history": self._create_obs_from_history()})

        return obs_dict, info

    def step(self, action):
        obs_dict, reward, terminated, truncated, info = super().step(action)
        obs = obs_dict["obs"]
        last_ax_size = obs.shape[-1]

        self.obs_history = np.roll(self.obs_history, shift=-last_ax_size, axis=-1)
        self.obs_history[..., -obs.shape[-1] :] = obs

        self.action_history = np.roll(self.action_history, shift=-action.shape[-1], axis=-1)
        self.action_history[..., -action.shape[-1] :] = action

        obs_dict.update({"history": self._create_obs_from_history()})

        return obs_dict, reward, terminated, truncated, info