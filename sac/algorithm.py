import os
import sys 
import torch
import numpy as np
from copy import deepcopy

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import common.opponent
from common.loading import load_agents
from common.evaluation import evaluate_against_many
from environment.hockey_env import HockeyEnv, BasicOpponent


class SACTrainer:
 

    def __init__(self, agent, env, logger, config, replay_buffer) -> None:
        self.logger = logger
        self.agent = agent 
        self.best_agent = self.agent
        self._config = config

        self.weak_opponent = BasicOpponent(weak=True)
        self.weak_opponent.agent_config = {"type": "basic", "name": "weak"}
        self.strong_opponent = BasicOpponent(weak=False)
        self.strong_opponent.agent_config = {"type": "basic", "name": "strong"}

        self.env = env 
        self.eval_env = deepcopy(env)

        self.buffer = replay_buffer
        self.wins = []
        self.rewards = []
        self.best_win_percentage = 0.0
        self.best_win_percentage_strong = 0
        
        
    def store_transition(self, transition):
        obs = torch.FloatTensor(transition[0]).to(self._config["device"])
        act = torch.FloatTensor(transition[1]).to(self._config["device"])
        rew = transition[2]
        next_obs = torch.FloatTensor(transition[3]).to(self._config["device"])
        d = transition[4]

        self.buffer.add(
            observation=obs, next_observation=next_obs, 
            action=act, reward=rew, done=d)
        
        

    def evaluate_in_training(self, opponents, appendix=""):
        with torch.no_grad():
            self.agent.eval()

            results = evaluate_against_many(env=self.eval_env, agent=self.agent, opponents=opponents,
                                    n_eval_episodes=self._config["eval_episodes"], appendix=appendix)

            # store the best agent 
            if results["avg_win_percentage"] > self.best_win_percentage:
                self.best_win_percentage = results["avg_win_percentage"]
                self.best_agent = deepcopy(self.agent)
                self.agent.save(os.path.join(self._config["best_agent_save_path"], "best_agent"))

            # if results["win_percentage_strong"] > self.best_win_percentage_strong:
            #     self.best_win_percentage_strong = results["win_percentage_strong"]

            if self._config["selfplay"] and self.best_win_percentage_strong > self._config["start_self_play_threshold"]:

                new_self_opponent = deepcopy(self.agent)
                # opponents.append(new_self_opponent)
                self.self_opponents.append(new_self_opponent)

                if len(self.self_opponents) > 15:
                    del_opponent = self.self_opponents.pop(0)
                    # opponents.remove(del_opponent)

                # if results["win_percentage_strong"] < self._config["start_self_play_threshold"]:
                #     opponents.append(self.strong_opponent)

            # print("Opponents length:", len(opponents))


            self.logger.log(results, self._config["log"])

            self.rewards.append(results["mean_total_reward_weak"])
            self.wins.append(results["win_percentage_weak"])

            if results["win_percentage_weak"] >= self.best_win_percentage:
                self.best_win_percentage = results["win_percentage_weak"]
                os.makedirs(self._config["best_agent_save_path"], exist_ok=True)
                

            self.agent.train()

    def evaluate(self, appendix=""):
        # self.eval_opponents = [self.weak_opponent, self.strong_opponent]
        self.eval_opponents = [self.weak_opponent]
        self.agent.eval()
        total_step_counter = 0 

        while 1:

            ob, _ = self.env.reset()
            obs_agent2 = self.env.obs_agent_two()
            total_reward = 0
            opponent = common.opponent.sample_opponent_uniform(self.eval_opponents)

            for step in range(self._config['max_steps']):
                
                action_1 = self.agent.act(ob)
                action_2 = self.agent.act(obs_agent2)
                # action_2 = opponent.act(obs_agent2)
                
                actions = np.hstack([action_1, action_2])

                next_state, sparse_reward, done, _, info = self.env.step(actions)
                self.env.render()

                # Augment state and reward
                reward = self.agent.reward.get(sparse_reward, info)
                total_reward += reward

                info["done"] = done

                # if self._config["augment_state"]:
                #     self.agent.store_transition((self.agent.observation.augment(ob), action_1, 
                #                                  reward, self.agent.observation.augment(next_state), done))
                # else:
                #     self.agent.store_transition((ob, action_1, reward, next_state, done))

                ob = next_state
                obs_agent2 = self.env.obs_agent_two() 

                if done:
                    break



                total_step_counter += 1


            # self.logger.print_episode_info(self.env.winner, total_step_counter, step, total_reward)
          

    def train_agent(self):
        
        err_dict_vanilla = {"total_loss": [], "loss_actor": [], "loss_critic1": [],
                        "loss_valuenet": []}
        err_dict_v1 = {"total_loss": [], "loss_actor": [], "loss_critic1": [],
                        "loss_critic2": []}
        
        episode_counter, total_step_counter = 0, 0

        # Flag for adding pretrained agent to opponents
        self.pretrained_added = False

        self.self_opponents = []
        opponents = [self.weak_opponent, self.strong_opponent]
        # this list contains the previous versions of the agent
        self.eval_opponents = [self.weak_opponent, self.strong_opponent]
        train_opponents = [self.weak_opponent, self.strong_opponent]

        # Just 'opponents' -> used for training and evaluation
        # if "opponents" in self._config:
        #     opponents_temp = load_agents(self._config["opponents"])
        #     self.eval_opponents.extend(opponents_temp)
        #     train_opponents.extend(opponents_temp)
        # # 'opponents_train' -> used for training, but not evaluation
        # if "opponents_train" in self._config:
        #     train_opponents.extend(
        #         load_agents(self._config["opponents_train"]))
        # # 'opponents_eval' -> used for evaluation, but not training
        # if "opponents_eval" in self._config:
        #     self.eval_opponents.extend(
        #         load_agents(self._config["opponents_eval"]))

        self.eval_opponents = [self.weak_opponent, self.strong_opponent]
        opponents = [self.weak_opponent, self.strong_opponent]


        while total_step_counter <= self._config['max_timesteps']:

            ob, _ = self.env.reset()
            obs_agent2 = self.env.obs_agent_two()

            total_reward, episode_step = 0, 0

            opponent = common.opponent.sample_opponent_uniform(opponents)

            for step in range(self._config['max_steps']):
                
                action_1 = self.agent.act(ob)
                action_2 = opponent.act(obs_agent2)
                
                actions = np.hstack([action_1, action_2])

                next_state, sparse_reward, done, _, info = self.env.step(actions)

                # Augment state and reward
                reward = self.agent.reward.get(sparse_reward, info)
                total_reward += reward

                info["done"] = done

                if self._config["augment_state"]:
                    self.store_transition((self.agent.observation.augment(ob), action_1, 
                                                 reward, self.agent.observation.augment(next_state), done))
                else:
                    self.store_transition((ob, action_1, reward, next_state, done))

                ob = next_state
                obs_agent2 = self.env.obs_agent_two() 

                if done or step == self._config['max_steps'] - 1:
                    break


                # self.env.render()

                # reset the networks if the version is sr-sac
                if self._config["version"] == "sr" and total_step_counter % self._config["network_reset"]:
                    self.agent.init_weights()
 
                total_step_counter += 1
                episode_step += 1

                # evaluation time 
                if (total_step_counter % self._config['evaluation_timestep'] == 0) and (not self._config["eval_only"]):  
                    self.evaluate_in_training(opponents, appendix="")

            if self.buffer.pos < self._config['batch_size']:
                continue

            update_dict = deepcopy(err_dict_vanilla) if self._config["version"] == "vanilla" else deepcopy(err_dict_v1)
            # update after episode ends 
            for _ in range(self._config["replay_ratio"] * episode_step):
                data = self.buffer.sample(self._config['batch_size'])
                update_dict_ = self.agent.update(data, total_step_counter)

                for k, v in update_dict.items():
                    v.append(update_dict_[k])
            
            update_dict = {k: np.mean(v) for k, v in update_dict.items()}
            
            self.logger.log(update_dict, self._config["log"])
            self.logger.print_episode_info(self.env.winner, total_step_counter, step, total_reward)
            episode_counter += 1