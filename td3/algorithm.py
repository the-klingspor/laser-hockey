import sys
from copy import deepcopy
from itertools import chain
from typing import Any, Callable

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam
from tqdm import tqdm

sys.path.append("..")

from agent import TD3Agent

import common.logger as logger
from common.algorithm import Algorithm
from common.evaluation import evaluate_against_many
from common.loading import load_agents
from common.noise import sample_gaussian_noise, sample_pink_noise
from common.opponent import sample_opponent_uniform
from common.replay_buffer import ReplayBuffer
from common.utils import get_linear_schedule, polyak_update
from environment.hockey_env import BasicOpponent, HockeyEnv


class TD3Algorithm(Algorithm):
    def __init__(
        self,
        env: HockeyEnv,
        agent: TD3Agent,
        replay_buffer: ReplayBuffer,
        algorithm_config: dict[str, Any],
        optimizer_config: tuple[dict[str, Any]],
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

        optimizer_actor_class_str = optimizer_config[0].pop("optimizer_class")
        if optimizer_actor_class_str == "adam":
            optimizer_actor_class = Adam
        else:
            raise NotImplementedError

        self.optimizer_actor = optimizer_actor_class(
            params=self.agent.actor.parameters(), **(optimizer_config[0])
        )

        optimizer_critic_class_str = optimizer_config[1].pop("optimizer_class")
        if optimizer_critic_class_str == "adam":
            optimizer_critic_class = Adam
        else:
            raise NotImplementedError

        self.optimizer_critics = optimizer_critic_class(
            params=chain(
                self.agent.critic_1.parameters(), self.agent.critic_2.parameters()
            ),
            **(optimizer_config[1]),
        )
        self.learning_starts = self.algorithm_config["learning_starts"]
        self.n_eval = self.algorithm_config["n_eval_episodes"]
        self.is_per = self.algorithm_config["is_per"]
        self.sample_opponent = sample_opponent_uniform

        self.num_timesteps = 1
        self.last_observation_1 = None
        self.last_observation_2 = None
        self.best_win_percentage = 0.0
        self.best_win_percentage_strong = 0.0
        self.best_agent = self.agent
        self.wins = []
        self.rewards = []

        if self.is_per:
            self.beta_schedule = get_linear_schedule(
                start=self.algorithm_config["beta_start"],
                end=self.algorithm_config["beta_end"],
                end_fraction=1,
            )

    def train_agent(self, total_timesteps: int) -> None:
        self.log = self.algorithm_config["log"]

        num_collected_observations = 0

        self.total_timesteps = total_timesteps
        self.start_pretrained = self.algorithm_config["start_pretrained"]
        self.pretrained_countdown = 0
        self.self_training_threshold = self.algorithm_config["self-training_threshold"]
        self.pretrained_added = False

        # Collect and augment observations
        last_observation_1, _ = self.env.reset()
        self.agent.reset_observations()
        self.last_observation_1 = self.agent.observation.augment(last_observation_1)
        self.last_observation_2 = self.env.obs_agent_two()

        # Instantiate opponents to train and evaluate against
        self.weak_opponent = BasicOpponent(weak=True)
        self.weak_opponent.agent_config = {"type": "basic", "name": "weak"}
        self.strong_opponent = BasicOpponent(weak=False)
        self.strong_opponent.agent_config = {"type": "basic", "name": "strong"}

        self.opponents = [self.weak_opponent]
        self.self_training_opponents = []
        self.eval_opponents = [self.weak_opponent, self.strong_opponent]
        self.train_opponents = []

        # Just 'opponents' -> used for training and evaluation
        if "opponents" in self.algorithm_config:
            opponents_temp = load_agents(self.algorithm_config["opponents"])
            self.eval_opponents.extend(opponents_temp)
            self.train_opponents.extend(opponents_temp)
        # 'opponents_train' -> used for training, but not evaluation
        if "opponents_train" in self.algorithm_config:
            self.train_opponents.extend(
                load_agents(self.algorithm_config["opponents_train"])
            )
        # 'opponents_eval' -> used for evaluation, but not training
        if "opponents_eval" in self.algorithm_config:
            self.eval_opponents.extend(
                load_agents(self.algorithm_config["opponents_eval"])
            )

        pbar = tqdm(total=self.total_timesteps)
        while self.num_timesteps < self.total_timesteps:
            n_gradient_steps = self.collect_rollout()
            num_collected_observations += n_gradient_steps

            # Do an episode of training after each episode of exploration
            if num_collected_observations > self.learning_starts:
                self.train_agent_steps(gradient_steps=n_gradient_steps)
                pbar.update(n_gradient_steps)

                # Add strong opponent after exploration
                if self.strong_opponent not in self.opponents:
                    self.opponents.append(self.strong_opponent)

            # Append pretrained agents once and enough steps of self-training
            if not self.pretrained_added:
                if self.best_win_percentage_strong > self.self_training_threshold:
                    self.pretrained_countdown += n_gradient_steps

                if self.pretrained_countdown >= self.start_pretrained:
                    for agent in self.train_opponents:
                        self.opponents.append(agent)

                    self.pretrained_added = True

        pbar.close()

        # Save agents and stats
        self.wins = np.array(self.wins)
        self.rewards = np.array(self.rewards)

    def train_agent_steps(self, gradient_steps: int) -> None:
        self.agent.train()

        # Load settings from config
        batch_size = self.algorithm_config["batch_size"]
        tau = self.algorithm_config["tau"]
        policy_delay = self.algorithm_config["policy_delay"]
        target_policy_noise = self.algorithm_config["target_policy_noise"]
        target_noise_clip = self.algorithm_config["target_noise_clip"]
        evaluate_every = self.algorithm_config["evaluate_every"]
        self_training = self.algorithm_config["self-training"]
        num_self_training = self.algorithm_config["num_self_training"]

        losses_actor = [0]
        losses_critics = [0]
        for step in range(gradient_steps):
            # Sample replay buffer
            if self.is_per:
                (
                    replay_data,
                    weights,
                    indices,
                ) = self.replay_buffer.sample(batch_size)
                weights = weights.to(self.device)
            else:
                replay_data = self.replay_buffer.sample(batch_size)

            # Send all elements to device
            observations = replay_data.observations.to(self.device)
            next_observations = replay_data.next_observations.to(self.device)
            actions = replay_data.actions.to(self.device)
            rewards = replay_data.rewards.unsqueeze(dim=-1).to(self.device)
            dones = replay_data.dones.byte().unsqueeze(dim=-1).to(self.device)

            with torch.no_grad():
                # Sample and clip target noise
                noise = torch.zeros_like(actions).normal_(0, target_policy_noise)
                noise = noise.clamp(-target_noise_clip, target_noise_clip)

                # Select noisy target action
                next_actions = self.agent.actor_target(next_observations) + noise
                next_actions = next_actions.clamp(-1, 1)

                # Compute TD target based on minimum value prediction of both critics
                next_observations_actions = torch.cat(
                    [next_observations, next_actions], dim=1
                )
                next_q_values = torch.cat(
                    [
                        self.agent.critic_target_1(next_observations_actions),
                        self.agent.critic_target_2(next_observations_actions),
                    ],
                    dim=1,
                )
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                td_target = rewards + (1 - dones) * self.gamma * next_q_values

            # Current Q value for each critic
            observations_actions = torch.cat([observations, actions], dim=1)
            q_values_critic_1 = self.agent.critic_1(observations_actions)
            q_values_critic_2 = self.agent.critic_2(observations_actions)

            # Compute MSE loss for all critics
            loss_critics_1 = F.mse_loss(q_values_critic_1, td_target, reduction="none")
            loss_critics_2 = F.mse_loss(q_values_critic_2, td_target, reduction="none")

            elementwise_loss_critics = loss_critics_1 + loss_critics_2

            # Update priorities if PER is used
            if self.is_per:
                loss_critics = (elementwise_loss_critics * weights).mean()

                loss_critics_for_prio = elementwise_loss_critics.detach().cpu()
                new_priorities = loss_critics_for_prio + 1e-6
                self.replay_buffer.update_priorities(indices, new_priorities)
            else:
                loss_critics = elementwise_loss_critics.mean()

            losses_critics.append(loss_critics.item())

            # Optimization step on critics
            self.optimizer_critics.zero_grad()
            loss_critics.backward()
            self.optimizer_critics.step()

            # Delayed actor update
            if self.num_timesteps % policy_delay == 0:
                # Actor loss is negative expected value
                pred_actions = self.agent.actor(observations)
                loss_actor = -self.agent.critic_1(
                    torch.cat([observations, pred_actions], dim=1)
                ).mean()
                losses_actor.append(loss_actor.item())

                # Optimization step on actor
                self.optimizer_actor.zero_grad()
                loss_actor.backward()
                self.optimizer_actor.step()

                # Update target networks with EMA
                self.polyak_update_step(tau)

            # Evaluate the agent against all opponents and save the currently best
            if self.num_timesteps % evaluate_every == 0:
                self.agent.eval()

                results = evaluate_against_many(
                    self.env,
                    self.agent,
                    self.eval_opponents,
                    self.n_eval,
                    appendix="",
                )

                logger.log(results, self.log)

                self.rewards.append(results["mean_total_reward_strong"])
                self.wins.append(results["win_percentage_strong"])

                if results["avg_win_percentage"] >= self.best_win_percentage:
                    self.best_win_percentage = results["avg_win_percentage"]
                    self.best_agent = deepcopy(self.agent)

                if results["win_percentage_strong"] >= self.best_win_percentage_strong:
                    self.best_win_percentage_strong = results["win_percentage_strong"]

                # Start self-training if results against the strong BasicOpponent are good enough
                if (
                    self_training
                    and self.best_win_percentage_strong > self.self_training_threshold
                ):
                    self.self_training_opponents.append(deepcopy(self.agent))
                    if len(self.self_training_opponents) > num_self_training:
                        self.self_training_opponents.pop(0)
                    if results["win_percentage_strong"] < self.self_training_threshold:
                        self.opponents.append(self.strong_opponent)

                print(f"Number of opponents: {len(self.opponents)}")

                self.agent.train()

            # Increase update counter
            self.num_timesteps += 1

    def collect_rollout(self) -> int:
        self.agent.eval()

        # Load settings
        num_actions = self.agent.agent_config["num_actions"]
        max_steps = self.algorithm_config["max_steps"]
        if self.algorithm_config["noise"] == "gaussian":
            sample_noise = sample_gaussian_noise
        elif self.algorithm_config["noise"] == "pink":
            sample_noise = sample_pink_noise
        else:
            raise NotImplementedError
        action_noise = self.algorithm_config["action_noise"]

        episode_noise = sample_noise(shape=(max_steps, num_actions))
        opponent = self.sample_opponent(self.opponents + self.self_training_opponents)
        opponent_noise_std = self.get_opponent_noise()

        i_steps = 0
        while i_steps < max_steps:
            # Select noisy action
            noise = action_noise * episode_noise[i_steps]
            action = self.agent.act(self.last_observation_1, noise)

            # Select (noisy) opponent action
            action_opponent = opponent.act(self.last_observation_2)
            opponent_noise = np.random.normal(
                0, opponent_noise_std, action_opponent.shape
            )
            action_opponent = action_opponent + opponent_noise
            action_opponent = np.clip(action_opponent, -1.0, 1.0)

            actions = np.hstack([action, action_opponent])

            new_observation, sparse_reward, done, _, info = self.env.step(actions)
            info["done"] = done

            # Augment state and reward
            reward = self.agent.reward.get(sparse_reward, info)

            new_observation_aug = self.agent.observation.augment(new_observation)

            new_observation_t = torch.from_numpy(new_observation_aug.astype(np.float32))
            new_observation_t = torch.cat(
                self.agent.last_observations[1:] + [new_observation_t], dim=0
            )
            action_t = torch.from_numpy(action.astype(np.float32))

            last_observation_t = torch.cat(self.agent.last_observations, dim=0)

            # Update beta of PER if needed
            if self.is_per:
                self.replay_buffer.beta = TD3Algorithm.get_beta(
                    self.num_timesteps, self.total_timesteps, self.beta_schedule
                )

            # Save newly observed data
            self.replay_buffer.add(
                last_observation_t, new_observation_t, action_t, reward, done
            )

            i_steps += 1

            # Save new state
            if done or i_steps == max_steps:
                self.last_observation_1, _ = self.env.reset()
                self.agent.reset_observations()
                self.last_observation_2 = self.env.obs_agent_two()

                break
            else:
                self.last_observation_1 = new_observation
                self.last_observation_2 = self.env.obs_agent_two()

        return i_steps

    def polyak_update_step(self, tau):
        polyak_update(
            self.agent.actor.parameters(), self.agent.actor_target.parameters(), tau
        )
        polyak_update(
            self.agent.critic_1.parameters(),
            self.agent.critic_target_1.parameters(),
            tau,
        )
        polyak_update(
            self.agent.critic_2.parameters(),
            self.agent.critic_target_2.parameters(),
            tau,
        )

    def update_beta(self) -> None:
        current_fraction = 1 - self.num_timesteps / self.total_timesteps
        beta = self.beta_schedule(current_fraction)
        self.replay_buffer.beta = beta

    @staticmethod
    def get_beta(
        num_timesteps: int, num_total_timesteps: int, beta_schedule: Callable
    ) -> float:
        progress = num_timesteps / num_total_timesteps

        return beta_schedule(progress)

    def get_opponent_noise(self) -> float:
        opponent_noise_type = self.algorithm_config["opponent_noise_type"]
        if opponent_noise_type == "none":
            opponent_noise_std = 0.0
        elif opponent_noise_type == "gaussian":
            opponent_noise_std = self.algorithm_config["opponent_noise"]
        elif opponent_noise_type == "exp":
            beta = self.algorithm_config["opponent_noise"]
            opponent_noise_std = np.random.exponential(beta)
        else:
            raise NotImplementedError()

        return opponent_noise_std
