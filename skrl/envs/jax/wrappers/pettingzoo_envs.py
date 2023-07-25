from typing import Any, Mapping, Sequence, Tuple, Union

import collections
import gymnasium

import jax
import numpy as np

from skrl.envs.jax.wrappers.base import MultiAgentEnvWrapper


class PettingZooWrapper(MultiAgentEnvWrapper):
    def __init__(self, env: Any) -> None:
        """PettingZoo (parallel) environment wrapper

        :param env: The environment to wrap
        :type env: Any supported PettingZoo (parallel) environment
        """
        super().__init__(env)

        self.possible_agents = self._env.possible_agents
        self._shared_observation_space = self._compute_shared_observation_space(self._env.observation_spaces)

    def _compute_shared_observation_space(self, observation_spaces):
        space = next(iter(observation_spaces.values()))
        shape = (len(self.possible_agents),) + space.shape
        return gymnasium.spaces.Box(low=np.stack([space.low for _ in self.possible_agents], axis=0),
                                    high=np.stack([space.high for _ in self.possible_agents], axis=0),
                                    dtype=space.dtype,
                                    shape=shape)

    @property
    def num_agents(self) -> int:
        """Number of agents
        """
        return len(self.possible_agents)

    @property
    def agents(self) -> Sequence[str]:
        """Names of all current agents

        These may be changed as an environment progresses (i.e. agents can be added or removed)
        """
        return self._env.agents

    @property
    def observation_spaces(self) -> Mapping[str, gymnasium.Space]:
        """Observation spaces
        """
        return {uid: self._env.observation_space(uid) for uid in self.possible_agents}

    @property
    def action_spaces(self) -> Mapping[str, gymnasium.Space]:
        """Action spaces
        """
        return {uid: self._env.action_space(uid) for uid in self.possible_agents}

    @property
    def shared_observation_spaces(self) -> Mapping[str, gymnasium.Space]:
        """Shared observation spaces
        """
        return {uid: self._shared_observation_space for uid in self.possible_agents}

    def _observation_to_tensor(self, observation: Any, space: gymnasium.Space) -> np.ndarray:
        """Convert the Gymnasium observation to a flat tensor

        :param observation: The Gymnasium observation to convert to a tensor
        :type observation: Any supported Gymnasium observation space

        :raises: ValueError if the observation space type is not supported

        :return: The observation as a flat tensor
        :rtype: np.ndarray
        """
        if isinstance(observation, int):
            return np.array(observation, dtype=np.int32).view(self.num_envs, -1)
        elif isinstance(observation, np.ndarray):
            return observation.reshape(self.num_envs, -1).astype(np.float32)
        elif isinstance(space, gymnasium.spaces.Discrete):
            return np.array(observation, dtype=np.float32).reshape(self.num_envs, -1)
        elif isinstance(space, gymnasium.spaces.Box):
            return observation.reshape(self.num_envs, -1).astype(np.float32)
        elif isinstance(space, gymnasium.spaces.Dict):
            tmp = np.concatenate([self._observation_to_tensor(observation[k], space[k]) \
                for k in sorted(space.keys())], axis=-1).view(self.num_envs, -1)
            return tmp
        else:
            raise ValueError(f"Observation space type {type(space)} not supported. Please report this issue")

    def _tensor_to_action(self, actions: np.ndarray, space: gymnasium.Space) -> Any:
        """Convert the action to the Gymnasium expected format

        :param actions: The actions to perform
        :type actions: np.ndarray

        :raise ValueError: If the action space type is not supported

        :return: The action in the Gymnasium format
        :rtype: Any supported Gymnasium action space
        """
        if isinstance(space, gymnasium.spaces.Discrete):
            return actions.item()
        elif isinstance(space, gymnasium.spaces.Box):
            return actions.astype(space.dtype).reshape(space.shape)
        raise ValueError(f"Action space type {type(space)} not supported. Please report this issue")

    def step(self, actions: Mapping[str, Union[np.ndarray, jax.Array]]) -> \
        Tuple[Mapping[str, Union[np.ndarray, jax.Array]], Mapping[str, Union[np.ndarray, jax.Array]],
              Mapping[str, Union[np.ndarray, jax.Array]], Mapping[str, Union[np.ndarray, jax.Array]],
              Mapping[str, Any]]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: dict of np.ndarray or jax.Array

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of dict of np.ndarray or jax.Array and any other info
        """
        if self._jax:
            actions = jax.device_get(actions)
        actions = {uid: self._tensor_to_action(action, self._env.action_space(uid)) for uid, action in actions.items()}
        observations, rewards, terminated, truncated, infos = self._env.step(actions)

        # build shared observation
        shared_observations = np.stack([observations[uid] for uid in self.possible_agents], axis=0)
        shared_observations = self._observation_to_tensor(shared_observations, self._shared_observation_space)
        infos["shared_states"] = {uid: shared_observations for uid in self.possible_agents}

        # convert response to numpy or jax
        observations = {uid: self._observation_to_tensor(value, self._env.observation_space(uid)) for uid, value in observations.items()}
        rewards = {uid: np.array(value, dtype=np.float32).reshape(self.num_envs, -1) for uid, value in rewards.items()}
        terminated = {uid: np.array(value, dtype=np.int8).reshape(self.num_envs, -1) for uid, value in terminated.items()}
        truncated = {uid: np.array(value, dtype=np.int8).reshape(self.num_envs, -1) for uid, value in truncated.items()}
        return observations, rewards, terminated, truncated, infos

    def reset(self) -> Tuple[Mapping[str, Union[np.ndarray, jax.Array]], Mapping[str, Any]]:
        """Reset the environment

        :return: Observation, info
        :rtype: tuple of dict of np.ndarray or jax.Array and any other info
        """
        outputs = self._env.reset()
        if isinstance(outputs, collections.abc.Mapping):
            observations = outputs
            infos = {uid: {} for uid in self.possible_agents}
        else:
            observations, infos = outputs

        # build shared observation
        shared_observations = np.stack([observations[uid] for uid in self.possible_agents], axis=0)
        shared_observations = self._observation_to_tensor(shared_observations, self._shared_observation_space)
        infos["shared_states"] = {uid: shared_observations for uid in self.possible_agents}

        # convert response to numpy or jax
        observations = {uid: self._observation_to_tensor(observation, self._env.observation_space(uid)) for uid, observation in observations.items()}
        return observations, infos

    def render(self, *args, **kwargs) -> None:
        """Render the environment
        """
        self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment
        """
        self._env.close()
