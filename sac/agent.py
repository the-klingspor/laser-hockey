import os 
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.distributions
from torch.nn import functional as F
from torch.distributions import Normal

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
 
from common.mlp import MLP
from common.agent import Agent
from common.utils import hard_update, polyak_update
from common.reward import SimpleReward, SparseReward
from common.observation import DistHockeyObservation, HockeyObservation

 

class Actor(MLP):
    def __init__(self, input_dim, act_func, hidden_sizes):
        
        # initialize MLP instance
        super().__init__(input_dim=input_dim,
                        hidden_layers=hidden_sizes,
                        output_dim=1,
                        activation_function=act_func,
                        use_noisy_linear=False)
        
        n_actions = 4
        self.mu = nn.Linear(hidden_sizes[-1], n_actions)
        self.log_sigma = nn.Linear(hidden_sizes[-1], n_actions)

    
    def forward(self, state):
        actor_feature = state
 
        for layer in self.layers[:-1]:
            actor_feature = F.relu(layer(actor_feature))

        mu = self.mu(actor_feature)
        log_sigma = self.log_sigma(actor_feature)
        
        # clip log_sigma for vanishind or exploding values
        log_sigma = torch.clamp(log_sigma, min=-10, max=10)

        return mu, log_sigma

class SACAgent(Agent):
    def __init__(self, agent_config, alpha_milestones=[100, 200, 300]):
        super().__init__(agent_config)

        self.agent_config = agent_config
   
 
        self.device = agent_config['device']
        self.alpha = agent_config['alpha_temp']
        self.tune_alpha_temp = agent_config['tune_alpha_temp']
 
        self.eval_mode = False

        reward_str = agent_config["reward"]

        if reward_str == "sparse":
            self.reward = SparseReward
        elif reward_str == "simple":
            self.reward = SimpleReward
        else:
            raise NotImplementedError

        observation_str = agent_config["observation"]

        if observation_str == "hockey":
            self.observation = HockeyObservation
        elif observation_str == "dist":
            self.observation = DistHockeyObservation
        else:
            raise NotImplementedError

 
        self.act_coef = torch.tensor(1.).to(self.device)
        self.act_bias = torch.tensor(0.).to(self.device)
        self.actor_eps = 1e-6

        lr_milestones = [int(x) for x in (agent_config['lr_milestones'][0]).split(' ')]

        self.agent_version = agent_config['version']
        observation_shape = 18 if not agent_config["augment_state"] else agent_config["observation_shape"]
        
        self.actor = Actor(input_dim=observation_shape,
                                    hidden_sizes=agent_config["actor_hidden_layers"],
                                    act_func=agent_config['activation_function'])

        self.critic1 = MLP(input_dim=observation_shape + agent_config["num_actions"],
                            hidden_layers=agent_config["critic_hidden_layers"],
                            output_dim=1,
                            activation_function=agent_config["activation_function"],
                            squash=False,
                            use_noisy_linear=False)
        self.critic1_target = MLP(input_dim=observation_shape + agent_config["num_actions"],
                            hidden_layers=agent_config["critic_hidden_layers"],
                            output_dim=1,
                            activation_function=agent_config["activation_function"],
                            squash=False,
                            use_noisy_linear=False)
        self.critic2 = MLP(input_dim=observation_shape + agent_config["num_actions"],
                            hidden_layers=agent_config["critic_hidden_layers"],
                            output_dim=1,
                            activation_function=agent_config["activation_function"],
                            squash=False,
                            use_noisy_linear=False)
        self.critic2_target = MLP(input_dim=observation_shape + agent_config["num_actions"],
                            hidden_layers=agent_config["critic_hidden_layers"],
                            output_dim=1,
                            activation_function=agent_config["activation_function"],
                            squash=False,
                            use_noisy_linear=False)
        self.value_net = MLP(input_dim=observation_shape,
                            hidden_layers=agent_config["critic_hidden_layers"],
                            output_dim=1,
                            activation_function=agent_config["activation_function"],
                            squash=False,
                            use_noisy_linear=False)
        self.value_net_target = MLP(input_dim=observation_shape,
                            hidden_layers=agent_config["critic_hidden_layers"],
                            output_dim=1,
                            activation_function=agent_config["activation_function"],
                            squash=False,
                            use_noisy_linear=False)
        
        self.init_weights()
        
        self.update_version_sr = self.update_version_v1
        
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=agent_config['lr'], eps=1e-6)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=agent_config['lr'], eps=1e-6)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=agent_config['lr'], eps=0.000001)
        self.value_net_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=agent_config['lr'], eps=1e-6)
        
        self.critic1_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.critic1_optimizer, 
                milestones=lr_milestones, gamma=agent_config['lr'])
        self.critic2_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.critic2_optimizer, 
                milestones=lr_milestones, gamma=agent_config['lr'])
        self.actor_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.actor_optimizer, 
                milestones=lr_milestones, gamma=agent_config['lr'])
        
        hard_update(target=self.critic1_target, source=self.critic1)
        hard_update(target=self.critic2_target, source=self.critic2)
        hard_update(self.value_net_target, self.value_net)
        

        if self.tune_alpha_temp:
            self.target_ent = -torch.tensor(agent_config["num_actions"]).to(self.device)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.agent_config['alpha_lr'])
            self.alpha_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.alpha_optim, milestones=alpha_milestones, gamma=0.5)
 
 
        device = agent_config["device"]
        if device == "cuda" and not torch.cuda.is_available():
            raise Exception("CUDA is not available, but was required")
        self.set_device(device)

 
    def save(self, path):
        torch.save({"actor_state_dict": self.actor.state_dict(),
                    "critic_1_state_dict": self.critic1.state_dict(),
                    "critic_2_state_dict": self.critic2.state_dict(),
                    "value_net_state_dict": self.value_net.state_dict()}, path,)

    def load(self, fpath):
        checkpoint = torch.load(fpath, map_location=torch.device(self.agent_config["device"]))
        
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic1.load_state_dict(checkpoint["critic_1_state_dict"])
        self.critic2.load_state_dict(checkpoint["critic_2_state_dict"])
        self.value_net.load_state_dict(checkpoint["value_net_state_dict"])
        
        hard_update(target=self.critic1_target, source=self.critic1)
        hard_update(target=self.critic2_target, source=self.critic2)
        hard_update(target=self.value_net_target, source=self.value_net)
    
    
    def init_weights(self):
        
        for _layer_ in self.critic1.layers:   
            if isinstance(_layer_, nn.Linear):
                torch.nn.init.xavier_uniform_(_layer_.weight)
        for _layer_ in self.critic2.layers:   
            if isinstance(_layer_, nn.Linear):
                torch.nn.init.xavier_uniform_(_layer_.weight)
        for _layer_ in self.actor.layers:   
            if isinstance(_layer_, nn.Linear):
                torch.nn.init.xavier_uniform_(_layer_.weight)
        for _layer_ in self.value_net.layers:
            if isinstance(_layer_, nn.Linear):
                torch.nn.init.xavier_uniform_(_layer_.weight)
                
        hard_update(target=self.critic1_target, source=self.critic1)
        hard_update(target=self.critic2_target, source=self.critic2)
        hard_update(self.value_net_target, self.value_net)
    
        
 
    def set_device(self, device: torch.device) -> None:
        self.device = device
        if hasattr(self, "critic"):
            self.critic.to(device)
            
        if hasattr(self, "critic_target"):
            self.critic_target.to(device)

        if hasattr(self, "critic1"):
            self.critic1.to(device)
            
        if hasattr(self, "critic2"):
            self.critic2.to(device)
            
        if hasattr(self, "critic1_target"):
            self.critic1_target.to(device)
            
        if hasattr(self, "critic2_target"):
            self.critic2_target.to(device)
        
        if hasattr(self, "actor"):
            self.actor.to(device)
        
        if hasattr(self, "value_net"):
            self.value_net.to(device)
        
        if hasattr(self, "value_net_target"):
            self.value_net_target.to(device)

    def eval(self):
        self.eval_mode = True

    def train(self):
        self.eval_mode = False

    def act(self, obs):        
        if self.agent_config["augment_state"]:
            obs = self.observation.augment(obs)
        
        obs = (torch.from_numpy(obs.astype(np.float32)).unsqueeze(dim=0).to(self.device))
        
        if self.eval_mode:
            _, _, action, _ = self.actor_sample(obs)
        else:
            action, _, _, _ = self.actor_sample(obs)
        return action.detach().cpu().numpy()[0]
    
    
    def actor_sample(self, obs):
        mu, log_sigma = self.actor(obs)
        sigma = log_sigma.exp()
        normal = Normal(mu, sigma)

        sampl = normal.rsample()
        sampl_tanh = torch.tanh(sampl)

        # Reparametrization
        action = sampl_tanh * self.act_coef + self.act_bias

        log_prob = normal.log_prob(sampl)

        log_prob -= torch.log(self.act_coef * (1 - sampl_tanh.pow(2)) + self.actor_eps)
        log_prob = log_prob.sum(axis=1, keepdim=True)
        mu = torch.tanh(mu) * self.act_coef + self.act_bias
        return action, log_prob, mu, sigma 


    def schedulers_step(self):
        self.critic.lr_scheduler.step()
        self.actor.lr_scheduler.step()
 
    def update_version_vanilla(self, data, total_step):
      
        state = data.observations.to(self.device)
        next_state = data.next_observations.to(self.device)
        action = data.actions.to(self.device)
        reward = data.rewards.to(self.device)
        not_done = (~data.dones).to(self.device)
        
        with torch.no_grad():
 
            V_next = self.value_net_target(next_state)
            Q_next = reward + not_done * self.agent_config['gamma'] * (V_next).squeeze()
            
        Q_cur = self.critic1(torch.cat((state, action), dim=1))
        
        Q_loss = F.mse_loss(Q_cur.squeeze(), Q_next)
        self.critic1_optimizer.zero_grad()
        Q_loss.backward()
        self.critic1_optimizer.step()
        
        pi, log_pi, _, _ = self.actor_sample(state) 
        
        V_cur = self.value_net(state)
        V_loss = F.mse_loss(V_cur.squeeze(), (Q_cur - self.alpha * log_pi).squeeze().detach())
        
        self.value_net_optimizer.zero_grad()
        V_loss.backward()
        self.value_net_optimizer.step()
        
        Q_pi = self.critic1(torch.cat((state, pi), dim=1))    
        # in vanilla SAC, we use critic1 to evaluate the next state-action pair
        policy_loss = ((self.alpha * log_pi) - Q_pi).mean(axis=0) 
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        if total_step % self.agent_config['update_target_every'] == 0:
            polyak_update(parameters=self.critic1.parameters(), 
                          target_parameters=self.critic1_target.parameters(),
                        tau=self.agent_config['tau'])
            polyak_update(parameters=self.value_net.parameters(), 
                            target_parameters=self.value_net_target.parameters(),
                            tau=self.agent_config['tau'])
             
        dict_to_log = {"total_loss": (Q_loss+V_loss).detach().cpu().numpy(),
                    "loss_actor": policy_loss.detach().cpu().numpy(),
                    "loss_critic1": Q_loss.detach().cpu().numpy(),
                    "loss_valuenet": V_loss.detach().cpu().numpy()}     
             
        return dict_to_log
    
    # select the update according to the agent version
    def update(self, data, total_step):
        return eval(f"self.update_version_{self.agent_version}")(data, total_step)
  
 
    def update_version_v1(self, data, total_step):
        
        state = data.observations.to(self.device)
        next_state = data.next_observations.to(self.device)
        action = data.actions.to(self.device)
        reward = data.rewards.to(self.device)
        not_done = (~data.dones).to(self.device)

        with torch.no_grad():
            
            next_state_action, next_state_log_pi, _, _ = self.actor_sample(next_state)
 
            q1_target_val = self.critic1_target(torch.cat((next_state, next_state_action), dim=1))
            q2_target_val = self.critic2_target(torch.cat((next_state, next_state_action), dim=1))

            q_target = torch.min(q1_target_val, q2_target_val) - self.alpha * next_state_log_pi
            next_q_value = reward + not_done * self.agent_config['gamma'] * (q_target).squeeze()
        
        q1_val = self.critic1(torch.cat((state, action), dim=1))
        q2_val = self.critic2(torch.cat((state, action), dim=1))
        
        q1_loss = F.mse_loss(q1_val.squeeze(), next_q_value)
        q2_loss = F.mse_loss(q2_val.squeeze(), next_q_value)
        qtotal_loss = q1_loss + q2_loss

        self.critic1_optimizer.zero_grad()
        q1_loss.backward()
        self.critic1_optimizer.step()
        # self.critic1_scheduler.step()
        
        self.critic2_optimizer.zero_grad()
        q2_loss.backward()
        self.critic2_optimizer.step()
        # self.critic2_scheduler.step()
        
        pi, log_pi, _, _ = self.actor_sample(state)

        qval1_pi = self.critic1(torch.cat((state, pi), dim=1))
        qval2_pi = self.critic2(torch.cat((state, pi), dim=1))
        min_qval_pi = torch.min(qval1_pi, qval2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qval_pi).mean(axis=0)

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        dict_to_log = {"total_loss": qtotal_loss.detach().cpu().numpy(),
                    "loss_actor": policy_loss.detach().cpu().numpy(),
                    "loss_critic1": q1_loss.detach().cpu().numpy(),
                    "loss_critic2": q2_loss.detach().cpu().numpy()}

        if self.tune_alpha_temp:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_ent).detach()).mean()
 

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha_lr_scheduler.step()
            self.alpha = self.log_alpha.exp().detach().item()
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)

        if total_step % self.agent_config['update_target_every'] == 0:
            polyak_update(parameters=self.critic1.parameters(), 
                          target_parameters=self.critic1_target.parameters(),
                        tau=self.agent_config['tau'])
            polyak_update(parameters=self.critic2.parameters(), 
                          target_parameters=self.critic2_target.parameters(),
                        tau=self.agent_config['tau'])
            
       
        return dict_to_log
 