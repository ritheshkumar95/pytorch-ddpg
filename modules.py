import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from memory import ReplayBuffer


class Actor(nn.Module):
    def __init__(self, cf):
        super(Actor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(cf.state_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, cf.action_dim),
            nn.Tanh()
            )

    def forward(self, state):
        return self.model(state)


class Critic(nn.Module):
    def __init__(self, cf):
        super(Critic, self).__init__()
        self.transform_state = nn.Sequential(
            nn.Linear(cf.state_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
            )

        self.transform_action = nn.Sequential(
            nn.Linear(cf.state_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
            )

        self.transform_both = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
            )

    def forward(self, state, action):
        state = self.transform_state(state)
        action = self.transform_action(action)
        both = torch.cat([state, action], 1)
        return self.transform_both(both)


class DDPG(nn.Module):
    def __init__(self, cf):
        super(DDPG, self).__init__()
        self.cf = cf

        self.actor = Actor(cf).cuda()
        self.actor_optimizer = optim.Adam(self.actor.parameters(),
                                          cf.learning_rate)
        self.target_actor = Actor(cf).cuda()

        self.critic = Critic(cf).cuda()
        self.actor_optimizer = optim.Adam(self.actor.parameters(),
                                          cf.learning_rate)
        self.target_critic = Critic(cf).cuda()

        self.buffer = ReplayBuffer(cf).cuda()

    def update_targets(self):
        for actor, actor_target in zip(self.actor.parameters(),
                                       self.actor_target.parameters()):
            actor_target.data.copy_(
                self.cf.tau*actor.data + (1-self.cf.tau)*actor_target.data
                )

        for critic, critic_target in zip(self.critic.parameters(),
                                         self.critic_target.parameters()):
            critic_target.data.copy_(
                self.cf.tau*critic.data + (1-self.cf.tau)*critic_target.data
                )

    def train_batch(self):
        s_t, a_t, r_t, s_tp1 = self.buffer.sample(self.cf.batch_size)

        # The below 2 operations need to be detached since we only
        # update critic and not targets
        a_tp1 = self.target_actor(s_tp1).detach()
        q_value = self.target_critic(s_tp1, a_tp1).detach()
        y_target = r_t + self.cf.gamma*q_value
        y_predicted = self.critic(s_t, a_t)

        critic_loss = F.smooth_l1_loss(y_predicted, y_target)
        critic_loss.backward()
        self.critic_optimizer.step()

        a_t_pred = self.actor(s_t)
        q_pred = self.critic(s_t, a_t_pred)
        actor_loss = -1*torch.sum(q_pred)  # because we want to maximize q_pred
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_targets()
        return critic_loss, actor_loss
