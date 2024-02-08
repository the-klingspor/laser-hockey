import os
import sys
from copy import deepcopy
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from numpy import ndarray
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

sys.path.append("..")

from agent import RainbowAgent

from common import logger
from common.algorithm import Algorithm
from common.evaluation import evaluate_against_many
from common.loading import load_agents
from common.opponent import sample_opponent_uniform
from common.replay_buffer import ReplayBuffer
from common.utils import get_linear_schedule, polyak_update
from environment.hockey_env import BasicOpponent, HockeyEnv


class RainbowAlgorithm(Algorithm):
    def __init__(
        self,
        env: HockeyEnv,
        agent: RainbowAgent,
        replay_buffer: ReplayBuffer,
        algorithm_config: dict[str, Any],
        optimizer_config: dict[str, Any],
        gamma: float,
    ) -> None:
        super().__init__(
            env,
            agent,
            replay_buffer,
            algorithm_config,
            optimizer_config,
            gamma,
        )
        optimizer_config = optimizer_config.copy()
        optimizer_class_str = optimizer_config.pop("optimizer_class")
        scheduler_steps = optimizer_config.pop("scheduler_steps")
        scheduler_factor = optimizer_config.pop("scheduler_factor")

        if optimizer_class_str == "adam":
            optimizer_class = Adam
        else:
            raise NotImplementedError

        self.optimizer = optimizer_class(
            params=self.agent.Q.parameters(), **optimizer_config
        )

        self.scheduler = MultiStepLR(
            optimizer=self.optimizer, milestones=scheduler_steps, gamma=scheduler_factor
        )

        if not self.algorithm_config["is_noisy"]:
            self.epsilon_schedule = get_linear_schedule(
                start=self.algorithm_config["exploration_initial_epsilon"],
                end=self.algorithm_config["exploration_final_epsilon"],
                end_fraction=self.algorithm_config["exploration_fraction"],
            )

        if self.algorithm_config["is_per"]:
            self.beta_schedule = get_linear_schedule(
                start=self.algorithm_config["beta_start"],
                end=self.algorithm_config["beta_end"],
                end_fraction=1,
            )

        self.multistep_buffer = []

    @staticmethod
    def train_agent_steps(
        agent: RainbowAgent,
        replay_buffer: ReplayBuffer,
        optimizer: Optimizer,
        scheduler: MultiStepLR,
        batch_size: int,
        gradient_steps: int,
        max_grad_norm: float,
        is_per: bool,
        is_distributional: bool,
        is_ddqn: bool,
        loss_buffer: list,
        gamma: float,
    ) -> None:
        device = agent.device
        for _ in range(gradient_steps):
            # Sample replay buffer
            if is_per:
                replay_data, weights, indices = replay_buffer.sample(batch_size)
                weights = weights.to(device)
            else:
                replay_data = replay_buffer.sample(batch_size)
            observations = replay_data.observations.to(device)
            next_observations = replay_data.next_observations.to(device)
            actions = replay_data.actions.to(device)
            rewards = replay_data.rewards.to(device)
            dones = replay_data.dones.byte().to(device)

            if is_distributional:
                elementwise_loss = RainbowAlgorithm.distributional_loss(
                    agent,
                    observations,
                    next_observations,
                    actions,
                    rewards,
                    dones,
                    gamma,
                )
            else:
                elementwise_loss = RainbowAlgorithm.td_loss(
                    agent,
                    observations,
                    next_observations,
                    actions,
                    rewards,
                    dones,
                    gamma,
                    is_ddqn,
                )

            if is_per:
                loss = (elementwise_loss * weights).mean()

                # Update PER priorities
                loss_for_priorities = elementwise_loss.detach().cpu()
                new_priorities = loss_for_priorities + 1e-6
                replay_buffer.update_priorities(indices, new_priorities)
            else:
                loss = elementwise_loss.mean()

            loss_buffer.append(loss.item())

            # Optimize policy
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(agent.Q.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

    @staticmethod
    def calculate_next_distributions(
        agent: RainbowAgent, next_observations: Tensor, batch_size: int
    ) -> Tensor:
        # Double DQN
        next_actions = agent.Q(next_observations).argmax(dim=-1)
        next_distributions = agent.Q_target.distribution(next_observations)  # [B, A, D]
        next_distributions = next_distributions[
            range(batch_size), next_actions
        ]  # [B, D]
        return next_distributions

    @staticmethod
    def calculate_projected_distributions(
        agent: RainbowAgent,
        next_distributions: Tensor,
        rewards: Tensor,
        dones: Tensor,
        batch_size: int,
        gamma: float,
    ) -> Tensor:
        atom_size = agent.Q.atom_size
        v_min = agent.Q.v_min
        v_max = agent.Q.v_max
        support = agent.Q.support
        delta_z = (v_max - v_min) / (atom_size - 1)

        t_z = rewards + (1 - dones) * gamma * support
        t_z = t_z.clamp(min=v_min, max=v_max)
        b = (t_z - v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        offset = (
            torch.linspace(0, (batch_size - 1) * atom_size, batch_size)
            .long()
            .unsqueeze(1)
            .expand(batch_size, atom_size)
            .to(agent.device)
        )

        proj_distributions = torch.zeros(next_distributions.size(), device=agent.device)
        proj_distributions.flatten().index_add_(
            dim=0,
            index=(l + offset).flatten(),
            source=(next_distributions * (u.float() - b)).flatten(),
        )
        proj_distributions.flatten().index_add_(
            dim=0,
            index=(u + offset).flatten(),
            source=(next_distributions * (b - l.float())).flatten(),
        )
        return proj_distributions

    @staticmethod
    def distributional_loss(
        agent: RainbowAgent,
        observations: Tensor,
        next_observations: Tensor,
        actions: Tensor,
        rewards: Tensor,
        dones: Tensor,
        gamma: float,
    ) -> Tensor:
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)
        batch_size = next_observations.shape[0]

        with torch.no_grad():
            next_distributions = RainbowAlgorithm.calculate_next_distributions(
                agent, next_observations, batch_size
            )
            proj_distributions = RainbowAlgorithm.calculate_projected_distributions(
                agent, next_distributions, rewards, dones, batch_size, gamma
            )

        distributions = agent.Q.distribution(observations)
        log_p = (distributions[range(batch_size), actions.long()]).log()
        elementwise_loss = -(proj_distributions * log_p).sum(dim=1)

        return elementwise_loss

    @staticmethod
    def td_loss(
        agent: RainbowAgent,
        observations: Tensor,
        next_observations: Tensor,
        actions: Tensor,
        rewards: Tensor,
        dones: Tensor,
        gamma: float,
        is_ddqn: bool,
    ) -> Tensor:
        with torch.no_grad():
            # Compute the next Q-values using the target network
            next_Q_values = agent.Q_target(next_observations)

            if is_ddqn:
                # Compute the next Q-value maximizers using the normal network
                max_actions = agent.Q(next_observations).argmax(dim=-1, keepdim=True)

                # Evaluate `max_actions` using the target network
                next_Q_values = next_Q_values.gather(
                    dim=-1, index=max_actions
                ).squeeze()
            else:
                next_Q_values = next_Q_values.max(dim=-1)[0]

            # n-step TD target
            target_Q_values = rewards + (1 - dones) * gamma * next_Q_values

        # Get current Q-value estimates
        current_Q_values = agent.Q(observations)

        # Get Q-values for the actions from the replay buffer
        current_Q_values = current_Q_values.gather(
            dim=-1, index=actions.long().unsqueeze(1)
        ).squeeze()

        # Huber loss
        elementwise_loss = F.smooth_l1_loss(
            current_Q_values, target_Q_values, reduction="none"
        )

        return elementwise_loss

    @staticmethod
    def collect_data(
        last_observation_1: ndarray,
        last_observation_2: ndarray,
        env: HockeyEnv,
        agent: RainbowAgent,
        opponent: RainbowAgent | BasicOpponent,
        epsilon: float,
        num_timesteps_in_episode: int,
        max_num_timesteps_in_episode: int,
    ) -> tuple[np.ndarray, np.ndarray, tuple]:
        observation_agent = last_observation_1
        observation_opponent = last_observation_2

        # Act with agent
        action_1, action_int = agent.act(
            observation=observation_agent,
            epsilon=epsilon,
            return_int=True,
        )

        # Act with opponent
        action_2 = opponent.act(observation_opponent)

        # Step with environment
        new_observation_1, reward, done, _, info = env.step(
            np.hstack([action_1, action_2])
        )
        new_observation_2 = env.obs_agent_two()

        new_observation_agent = new_observation_1

        reward = agent.reward.get(reward, info)

        observation_agent_aug_t = torch.cat(agent.last_observations, dim=0)

        new_observation_agent_aug = agent.observation.augment(new_observation_agent)
        new_observation_agent_aug_t = torch.from_numpy(
            new_observation_agent_aug.astype(np.float32)
        )
        new_observation_agent_aug_t = torch.cat(
            agent.last_observations[1:] + [new_observation_agent_aug_t], dim=0
        )

        action_t = torch.tensor(action_int, dtype=torch.uint8)
        reward_t = torch.tensor(reward, dtype=torch.float32)

        done = done or num_timesteps_in_episode + 1 == max_num_timesteps_in_episode

        if done:
            last_observation_1, _ = env.reset()
            last_observation_2 = env.obs_agent_two()
            agent.reset_observations()
        else:
            last_observation_1 = new_observation_1
            last_observation_2 = new_observation_2

        return (
            last_observation_1,
            last_observation_2,
            (
                observation_agent_aug_t,
                new_observation_agent_aug_t,
                action_t,
                reward_t,
                done,
            ),
        )

    @staticmethod
    def extract_multistep_tuple(
        multistep_buffer: list[tuple], gamma: float
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, bool]:
        (
            last_observation_t,
            new_observation_t,
            action_t,
            reward_t,
            done,
        ) = multistep_buffer.pop(0)

        gamma_pow = gamma

        for element in multistep_buffer:
            # Update target observation
            new_observation_t = element[1]

            # Add discounted reward
            reward_t += gamma_pow * element[3]

            # Update done flag
            done = element[-1]

            gamma_pow *= gamma

        if done:
            multistep_buffer.clear()

        return last_observation_t, new_observation_t, action_t, reward_t, done

    @staticmethod
    def get_epsilon(
        num_timesteps: int,
        num_total_timesteps: int,
        learning_starts: int,
        epsilon_schedule: Callable,
    ) -> float:
        num_learning_timesteps = max(0, num_timesteps - learning_starts)
        total_learning_timesteps = num_total_timesteps - learning_starts
        progress = num_learning_timesteps / total_learning_timesteps

        return epsilon_schedule(progress)

    @staticmethod
    def get_beta(
        num_timesteps: int, num_total_timesteps: int, beta_schedule: Callable
    ) -> float:
        progress = num_timesteps / num_total_timesteps

        return beta_schedule(progress)

    @staticmethod
    def update_buffers(
        replay_buffer: ReplayBuffer,
        multistep_buffer: list[tuple],
        data: tuple,
        num_multistep: int,
        gamma: float,
    ) -> None:
        multistep_buffer.append(data)

        if len(multistep_buffer) == num_multistep:
            replay_buffer.add(
                *RainbowAlgorithm.extract_multistep_tuple(multistep_buffer, gamma)
            )

    def train_agent(self, num_total_timesteps: int) -> None:
        self.agent.train()
        learning_starts = self.algorithm_config["learning_starts"]
        gradient_steps = self.algorithm_config["gradient_steps"]
        num_multistep = self.algorithm_config["num_multistep"]
        target_frequency = self.algorithm_config["target_frequency"]
        batch_size = self.algorithm_config["batch_size"]
        max_grad_norm = self.algorithm_config["max_grad_norm"]
        is_noisy = self.algorithm_config["is_noisy"]
        is_per = self.algorithm_config["is_per"]
        is_distributional = self.algorithm_config["is_distributional"]
        is_ddqn = self.algorithm_config["is_ddqn"]
        log = self.algorithm_config["log"]
        max_num_timesteps_in_episode = self.algorithm_config[
            "max_num_timesteps_in_episode"
        ]
        self.num_eval_episodes = self.algorithm_config["num_eval_episodes"]
        self.num_eval_episodes_final = self.algorithm_config["num_eval_episodes_final"]
        start_pretrained_delta = self.algorithm_config["start_pretrained_delta"]
        start_self_play_threshold = self.algorithm_config["start_self_play_threshold"]
        is_self_training = self.algorithm_config["is_self_training"]
        train_frequency = self.algorithm_config["train_frequency"]
        max_len_buffer = 100
        evaluation_frequency = 100_000
        best_win_percentage = 0
        best_win_percentage_strong = 0
        self.best_agent = deepcopy(self.agent)

        num_timesteps_in_episode = 0
        num_timesteps = 0
        num_grad_steps_since_last_update = 0
        pretrained_opponent_counter = 0
        pretrained_opponents_added = False

        # Collect observations
        last_observation_1, _ = self.env.reset()
        last_observation_2 = self.env.obs_agent_two()
        self.agent.reset_observations()

        # Instantiate opponents to train and evaluate against
        self.weak_opponent = BasicOpponent(weak=True)
        self.weak_opponent.agent_config = {"type": "basic", "name": "weak"}
        self.strong_opponent = BasicOpponent(weak=False)
        self.strong_opponent.agent_config = {"type": "basic", "name": "strong"}

        opponents = [self.weak_opponent]
        self_opponents = []
        self.eval_opponents = [self.weak_opponent, self.strong_opponent]
        train_opponents = []

        # Just 'opponents' -> used for training and evaluation
        if "opponents" in self.algorithm_config:
            opponents_temp = load_agents(self.algorithm_config["opponents"])
            self.eval_opponents.extend(opponents_temp)
            train_opponents.extend(opponents_temp)
        # 'opponents_train' -> used for training, but not evaluation
        if "opponents_train" in self.algorithm_config:
            train_opponents.extend(
                load_agents(self.algorithm_config["opponents_train"])
            )
        # 'opponents_eval' -> used for evaluation, but not training
        if "opponents_eval" in self.algorithm_config:
            self.eval_opponents.extend(
                load_agents(self.algorithm_config["opponents_eval"])
            )

        pbar = tqdm(total=num_total_timesteps)
        loss_buffer = []

        opponent = sample_opponent_uniform(opponents)

        while num_timesteps < num_total_timesteps:
            for _ in range(train_frequency):
                # Turn of epsilon-greedy behavior when we use noisy linear layers
                # as they take care of exploration
                if is_noisy:
                    self.agent.Q.reset_noise()
                    self.agent.Q_target.reset_noise()
                    epsilon = 0
                else:
                    epsilon = RainbowAlgorithm.get_epsilon(
                        num_timesteps,
                        num_total_timesteps,
                        learning_starts,
                        self.epsilon_schedule,
                    )

                if is_per:
                    self.replay_buffer.beta = RainbowAlgorithm.get_beta(
                        num_timesteps, num_total_timesteps, self.beta_schedule
                    )

                return_tuple = RainbowAlgorithm.collect_data(
                    last_observation_1,
                    last_observation_2,
                    self.env,
                    self.agent,
                    opponent,
                    epsilon,
                    num_timesteps_in_episode,
                    max_num_timesteps_in_episode,
                )
                last_observation_1, last_observation_2, data = return_tuple

                RainbowAlgorithm.update_buffers(
                    self.replay_buffer,
                    self.multistep_buffer,
                    data,
                    num_multistep,
                    self.gamma,
                )

                num_timesteps += 1
                pbar.update()
                done = data[-1]

                if done:
                    num_timesteps_in_episode = 0
                    opponent = sample_opponent_uniform(opponents)
                else:
                    num_timesteps_in_episode += 1

                if num_timesteps % evaluation_frequency == 0:
                    with torch.no_grad():
                        self.agent.eval()

                        results = evaluate_against_many(
                            self.env,
                            self.agent,
                            self.eval_opponents,
                            self.num_eval_episodes,
                            appendix="",
                        )
                        logger.log(results, log=log, step=num_timesteps)

                        if results["avg_win_percentage"] > best_win_percentage:
                            best_win_percentage = results["avg_win_percentage"]
                            self.best_agent = deepcopy(self.agent)

                        if (
                            results["win_percentage_strong"]
                            > best_win_percentage_strong
                        ):
                            best_win_percentage_strong = results[
                                "win_percentage_strong"
                            ]

                        if (
                            is_self_training
                            and best_win_percentage_strong > start_self_play_threshold
                        ):
                            new_self_opponent = deepcopy(self.agent)
                            opponents.append(new_self_opponent)
                            self_opponents.append(new_self_opponent)
                            if len(self_opponents) > 15:
                                del_opponent = self_opponents.pop(0)
                                opponents.remove(del_opponent)

                            if (
                                results["win_percentage_strong"]
                                < start_self_play_threshold
                            ):
                                opponents.append(self.strong_opponent)

                        print("Opponents length:", len(opponents))

                        self.agent.train()
                    last_observation_1, _ = self.env.reset()
                    last_observation_2 = self.env.obs_agent_two()
                    self.agent.reset_observations()

            if num_timesteps > learning_starts:
                if self.strong_opponent not in opponents:
                    opponents.append(self.strong_opponent)

                gamma = self.gamma**num_multistep
                RainbowAlgorithm.train_agent_steps(
                    self.agent,
                    self.replay_buffer,
                    self.optimizer,
                    self.scheduler,
                    batch_size,
                    gradient_steps,
                    max_grad_norm,
                    is_per,
                    is_distributional,
                    is_ddqn,
                    loss_buffer,
                    gamma,
                )

                num_grad_steps_since_last_update += gradient_steps
                if num_grad_steps_since_last_update >= target_frequency:
                    polyak_update(
                        self.agent.Q.parameters(),
                        self.agent.Q_target.parameters(),
                        self.algorithm_config["tau"],
                    )

                    num_grad_steps_since_last_update = 0

                if len(loss_buffer) > max_len_buffer:
                    logger.log(
                        {"avg_loss": np.mean(loss_buffer).item()},
                        log=log,
                        step=num_timesteps,
                    )
                    loss_buffer = []

            if not pretrained_opponents_added:
                if best_win_percentage_strong > start_self_play_threshold:
                    pretrained_opponent_counter += train_frequency

                if pretrained_opponent_counter >= start_pretrained_delta:
                    opponents.extend(train_opponents)
                    pretrained_opponents_added = True

        pbar.close()

    def train_agent_tournament(self, num_total_timesteps: int, path: str) -> None:
        self.agent.train()
        learning_starts = self.algorithm_config["learning_starts"]
        gradient_steps = self.algorithm_config["gradient_steps"]
        num_multistep = self.algorithm_config["num_multistep"]
        target_frequency = self.algorithm_config["target_frequency"]
        batch_size = self.algorithm_config["batch_size"]
        max_grad_norm = self.algorithm_config["max_grad_norm"]
        is_noisy = self.algorithm_config["is_noisy"]
        is_per = self.algorithm_config["is_per"]
        is_distributional = self.algorithm_config["is_distributional"]
        is_ddqn = self.algorithm_config["is_ddqn"]
        log = self.algorithm_config["log"]
        max_num_timesteps_in_episode = self.algorithm_config[
            "max_num_timesteps_in_episode"
        ]
        self.num_eval_episodes = self.algorithm_config["num_eval_episodes"]
        self.num_eval_episodes_final = self.algorithm_config["num_eval_episodes_final"]
        start_pretrained_delta = self.algorithm_config["start_pretrained_delta"]
        start_self_play_threshold = self.algorithm_config["start_self_play_threshold"]
        is_self_training = self.algorithm_config["is_self_training"]
        train_frequency = self.algorithm_config["train_frequency"]
        max_len_buffer = 100
        evaluation_frequency = 100_000
        best_win_percentage = 0
        best_win_percentage_strong = 0
        self.best_agent = deepcopy(self.agent)

        num_timesteps_in_episode = 0
        num_timesteps = 0
        num_grad_steps_since_last_update = 0
        pretrained_opponent_counter = 0
        pretrained_opponents_added = False

        # Collect observations
        last_observation_1, _ = self.env.reset()
        last_observation_2 = self.env.obs_agent_two()
        self.agent.reset_observations()

        # Instantiate opponents to train and evaluate against
        self.weak_opponent = BasicOpponent(weak=True)
        self.weak_opponent.agent_config = {"type": "basic", "name": "weak"}
        self.strong_opponent = BasicOpponent(weak=False)
        self.strong_opponent.agent_config = {"type": "basic", "name": "strong"}

        opponents = [self.weak_opponent]
        self_opponents = []
        self.eval_opponents = [self.weak_opponent, self.strong_opponent]
        train_opponents = []

        # Just 'opponents' -> used for training and evaluation
        if "opponents" in self.algorithm_config:
            opponents_temp = load_agents(self.algorithm_config["opponents"])
            self.eval_opponents.extend(opponents_temp)
            train_opponents.extend(opponents_temp)
        # 'opponents_train' -> used for training, but not evaluation
        if "opponents_train" in self.algorithm_config:

            if not self.algorithm_config["opponents_train"]:
                print("Warning: Training opponent list is empty. "
                      "This run is only for demonstration purposes."
                )

            train_opponents.extend(
                load_agents(self.algorithm_config["opponents_train"])
            )
        # 'opponents_eval' -> used for evaluation, but not training
        if "opponents_eval" in self.algorithm_config:
            self.eval_opponents.extend(
                load_agents(self.algorithm_config["opponents_eval"])
            )

        pbar = tqdm(total=num_total_timesteps)
        loss_buffer = []

        opponent = sample_opponent_uniform(opponents)

        while num_timesteps < num_total_timesteps:
            for _ in range(train_frequency):
                # Turn of epsilon-greedy behavior when we use noisy linear layers
                # as they take care of exploration
                if is_noisy:
                    self.agent.Q.reset_noise()
                    self.agent.Q_target.reset_noise()
                    epsilon = 0
                else:
                    epsilon = RainbowAlgorithm.get_epsilon(
                        num_timesteps,
                        num_total_timesteps,
                        learning_starts,
                        self.epsilon_schedule,
                    )

                if is_per:
                    self.replay_buffer.beta = RainbowAlgorithm.get_beta(
                        num_timesteps, num_total_timesteps, self.beta_schedule
                    )

                return_tuple = RainbowAlgorithm.collect_data(
                    last_observation_1,
                    last_observation_2,
                    self.env,
                    self.agent,
                    opponent,
                    epsilon,
                    num_timesteps_in_episode,
                    max_num_timesteps_in_episode,
                )
                last_observation_1, last_observation_2, data = return_tuple

                RainbowAlgorithm.update_buffers(
                    self.replay_buffer,
                    self.multistep_buffer,
                    data,
                    num_multistep,
                    self.gamma,
                )

                num_timesteps += 1
                pbar.update()
                done = data[-1]

                if done:
                    num_timesteps_in_episode = 0
                    opponent = sample_opponent_uniform(opponents)
                else:
                    num_timesteps_in_episode += 1

                if num_timesteps % evaluation_frequency == 0:
                    self.replay_buffer.poll_folders(
                        self.algorithm_config["folders"],
                        self.agent.observation,
                        self.agent.reward,
                        self.agent.action_converter.action2index,
                        num_stacked_observations=self.agent.agent_config[
                            "num_stacked_observations"
                        ],
                    )

                    with torch.no_grad():
                        self.agent.eval()

                        results = evaluate_against_many(
                            self.env,
                            self.agent,
                            self.eval_opponents,
                            self.num_eval_episodes,
                            appendix="",
                        )
                        logger.log(results, log=log, step=num_timesteps)
                        save_path = os.path.join(path, f"agent_{num_timesteps}")
                        self.agent.save(save_path)
                        logger.save_file(save_path, log=log)

                        if results["avg_win_percentage"] > best_win_percentage:
                            best_win_percentage = results["avg_win_percentage"]
                            self.best_agent = deepcopy(self.agent)

                        if (
                            results["win_percentage_strong"]
                            > best_win_percentage_strong
                        ):
                            best_win_percentage_strong = results[
                                "win_percentage_strong"
                            ]

                        if (
                            is_self_training
                            and best_win_percentage_strong > start_self_play_threshold
                        ):
                            new_self_opponent = deepcopy(self.agent)
                            opponents.append(new_self_opponent)
                            self_opponents.append(new_self_opponent)
                            if len(self_opponents) > 15:
                                del_opponent = self_opponents.pop(0)
                                opponents.remove(del_opponent)

                            if (
                                results["win_percentage_strong"]
                                < start_self_play_threshold
                            ):
                                opponents.append(self.strong_opponent)

                        print("Opponents length:", len(opponents))

                        self.agent.train()
                    last_observation_1, _ = self.env.reset()
                    last_observation_2 = self.env.obs_agent_two()
                    self.agent.reset_observations()

            if num_timesteps > learning_starts:
                if self.strong_opponent not in opponents:
                    opponents.append(self.strong_opponent)

                gamma = self.gamma**num_multistep
                RainbowAlgorithm.train_agent_steps(
                    self.agent,
                    self.replay_buffer,
                    self.optimizer,
                    self.scheduler,
                    batch_size,
                    gradient_steps,
                    max_grad_norm,
                    is_per,
                    is_distributional,
                    is_ddqn,
                    loss_buffer,
                    gamma,
                )

                num_grad_steps_since_last_update += gradient_steps
                if num_grad_steps_since_last_update >= target_frequency:
                    polyak_update(
                        self.agent.Q.parameters(),
                        self.agent.Q_target.parameters(),
                        self.algorithm_config["tau"],
                    )

                    num_grad_steps_since_last_update = 0

                if len(loss_buffer) > max_len_buffer:
                    logger.log(
                        {"avg_loss": np.mean(loss_buffer).item()},
                        log=log,
                        step=num_timesteps,
                    )
                    loss_buffer = []

            if not pretrained_opponents_added:
                if best_win_percentage_strong > start_self_play_threshold:
                    pretrained_opponent_counter += train_frequency

                if pretrained_opponent_counter >= start_pretrained_delta:
                    opponents.extend(train_opponents)
                    pretrained_opponents_added = True

        pbar.close()
