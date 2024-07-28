import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data.sampler import *
import dgl
from dgl.nn.pytorch import GraphConv, HeteroGraphConv


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


class Actor_MLP(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_MLP, self).__init__()
        self.fc1 = nn.Linear(actor_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, actor_input):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size, episode_limit, N, actor_input_dim), prob.shape(mini_batch_size, episode_limit, N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        x = self.activate_func(self.fc2(x))
        prob = torch.softmax(self.fc3(x), dim=-1)
        return prob


class Critic_MLP(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_MLP, self).__init__()
        self.fc1 = nn.Linear(critic_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size, episode_limit, N, critic_input_dim), value.shape=(mini_batch_size, episode_limit, N, 1)
        x = self.activate_func(self.fc1(critic_input))
        x = self.activate_func(self.fc2(x))
        value = self.fc3(x)
        return value


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        # 实例化HeteroGraphConv，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregate是聚合函数的类型
        self.conv1 = HeteroGraphConv({
            rel: GraphConv(in_feats[i], hid_feats)
            for i, rel in enumerate(rel_names)}, aggregate='sum')
        self.conv2 = HeteroGraphConv({
            rel: GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv3 = HeteroGraphConv({
            rel: GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # 输入是节点的特征字典
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv3(graph, h)
        return h['driver']


class ActorGNNMLP(nn.Module):
    def __init__(self, args):
        super(ActorGNNMLP, self).__init__()
        self.gnn = RGCN(in_feats=[4, args.agent_n + 25, args.agent_n + 25, 4, 6, 4], hid_feats=args.gnn_hidden_dim, out_feats=args.gnn_embedding_dim,
                        rel_names=['region2region', 'driver2region', 'driver2driver', 'region2driver', 'order2region', 'POI2region'])
        self.mlp = Actor_MLP(args, args.gnn_embedding_dim)

    def forward(self, graph, gnn_input):
        gnn_input_shape = gnn_input['driver'].shape
        driver = gnn_input['driver']
        if len(gnn_input_shape) == 4:
            gnn_output = self.gnn(graph, gnn_input)
            # gnn_output_ = torch.cat((gnn_output, driver), dim=-1)
            actor_output = self.mlp(gnn_output).permute(1, 2, 0, 3)
            return actor_output, gnn_output
        elif len(gnn_input_shape) == 2:
            gnn_output = self.gnn(graph, gnn_input)
            # gnn_output_ = torch.cat((gnn_output, driver), dim=-1)
            actor_output = self.mlp(gnn_output)
            return actor_output, gnn_output

class MAPPO:
    def __init__(self, args):
        self.N = args.N
        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.episode_limit = args.episode_limit
        self.rnn_hidden_dim = args.rnn_hidden_dim

        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.add_agent_id = args.add_agent_id
        self.use_value_clip = args.use_value_clip

        # get the input dimension of actor and critic
        self.actor_input_dim = args.obs_dim
        self.critic_input_dim = args.state_dim
        if self.add_agent_id:
            print("------add agent id------")
            self.actor_input_dim += args.N
            self.critic_input_dim += args.N

        self.device = args.device

        self.actor = ActorGNNMLP(args).to(self.device)
        self.critic = Critic_MLP(args, self.critic_input_dim).to(self.device)

        self.ac_parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        if self.set_adam_eps:
            print("------set adam eps------")
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr, eps=1e-5)
        else:
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr)

    def choose_action(self, obs_n, obs_order, obs_POI, obs_region, graph, evaluate):
        with torch.no_grad():
            actor_inputs = []
            obs_n = torch.tensor(obs_n, dtype=torch.float32)  # obs_n.shape=(N，obs_dim)
            actor_inputs.append(obs_n)
            if self.add_agent_id:
                """
                    Add an one-hot vector to represent the agent_id
                    For example, if N=3
                    [obs of agent_1]+[1,0,0]
                    [obs of agent_2]+[0,1,0]
                    [obs of agent_3]+[0,0,1]
                    So, we need to concatenate a N*N unit matrix(torch.eye(N))
                """
                actor_inputs.append(torch.eye(self.N))

            actor_inputs = torch.cat([x for x in actor_inputs], dim=-1).to(self.device) # actor_input.shape=(N, actor_input_dim)
            obs_order = torch.tensor(obs_order, dtype=torch.float32).to(self.device)
            obs_POI = torch.tensor(obs_POI, dtype=torch.float32).to(self.device)
            obs_region = torch.tensor(obs_region, dtype=torch.float32).to(self.device)
            gnn_input = {'driver': actor_inputs, 'region': obs_region, 'order': obs_order, 'POI': obs_POI}
            graph_ = graph
            graph_ = graph_.to(self.device)
            graph_.nodes['driver'].data['feature'] = actor_inputs
            graph_.nodes['region'].data['feature'] = obs_region
            graph_.nodes['order'].data['feature'] = obs_order
            graph_.nodes['POI'].data['feature'] = obs_POI
            prob, gnn_output = self.actor(graph_, gnn_input)  # prob.shape=(N,action_dim)
            if evaluate:  # When evaluating the policy, we select the action with the highest probability
                a_n = prob.argmax(dim=-1)
                return a_n.cpu().numpy(), None, gnn_output
            else:
                dist = Categorical(probs=prob)
                a_n = dist.sample()
                a_logprob_n = dist.log_prob(a_n)
                return a_n.cpu().numpy(), a_logprob_n.cpu().numpy(), gnn_output

    def random_action(self):
        """
        随机执行动作
        :return:
        """
        random_action = np.random.randint(0, self.action_dim, self.N)
        return random_action

    def get_value(self, s):
        with torch.no_grad():
            critic_inputs = []
            # Because each agent has the same global state, we need to repeat the global state 'N' times.
            s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).repeat(self.N, 1)  # (state_dim,)-->(N,state_dim)
            critic_inputs.append(s)
            if self.add_agent_id:  # Add an one-hot vector to represent the agent_id
                critic_inputs.append(torch.eye(self.N))
            critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)  # critic_input.shape=(N, critic_input_dim)
            v_n = self.critic(critic_inputs.to(self.device))  # v_n.shape(N,1)
            return v_n.cpu().numpy().flatten()

    def train(self, replay_buffer, total_steps, graph):
        batch = replay_buffer.get_training_data()  # get training data

        # Calculate the advantage using GAE
        adv = []
        gae = 0
        with torch.no_grad():  # adv and td_target have no gradient
            deltas = batch['r_n'] + self.gamma * batch['v_n'][:, 1:] * (1 - batch['done_n']) - batch['v_n'][:, :-1]  # deltas.shape=(batch_size,episode_limit,N)
            for t in reversed(range(self.episode_limit)):
                gae = deltas[:, t] + self.gamma * self.lamda * gae
                adv.insert(0, gae)
            adv = torch.stack(adv, dim=1)  # adv.shape(batch_size,episode_limit,N)
            v_target = adv + batch['v_n'][:, :-1]  # v_target.shape(batch_size,episode_limit,N)
            if self.use_adv_norm:  # Trick 1: advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        """
            Get actor_inputs and critic_inputs
            actor_inputs.shape=(batch_size, max_episode_len, N, actor_input_dim)
            critic_inputs.shape=(batch_size, max_episode_len, N, critic_input_dim)
        """
        actor_inputs, order_inputs, POI_inputs, region_inputs, critic_inputs = self.get_inputs(batch)
        actor_inputs = actor_inputs.to(self.device)
        order_inputs = order_inputs.to(self.device)
        POI_inputs = POI_inputs.to(self.device)
        region_inputs = region_inputs.to(self.device)
        critic_inputs = critic_inputs.to(self.device)

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                """
                    get probs_now and values_now
                    probs_now.shape=(mini_batch_size, episode_limit, N, action_dim)
                    values_now.shape=(mini_batch_size, episode_limit, N)
                """

                actor_inputs_ = actor_inputs[index].permute(2, 0, 1, 3)
                region_inputs_ = region_inputs[index].permute(2, 0, 1, 3)
                POI_inputs_ = POI_inputs[index].permute(2, 0, 1, 3)
                order_inputs_ = order_inputs[index].permute(2, 0, 1, 3)
                gnn_input = {'driver': actor_inputs_, 'region': region_inputs_, 'POI': POI_inputs_, 'order': order_inputs_}

                graph_ = graph
                graph_ = graph_.to(self.device)
                graph_.nodes['driver'].data['feature'] = actor_inputs_
                graph_.nodes['region'].data['feature'] = region_inputs_
                graph_.nodes['POI'].data['feature'] = POI_inputs_
                graph_.nodes['order'].data['feature'] = order_inputs_

                probs_now, _ = self.actor(graph_.to(self.device), gnn_input)
                values_now = self.critic(critic_inputs[index]).squeeze(-1)

                dist_now = Categorical(probs_now)
                dist_entropy = dist_now.entropy()  # dist_entropy.shape=(mini_batch_size, episode_limit, N)
                # batch['a_n'][index].shape=(mini_batch_size, episode_limit, N)
                a_logprob_n_now = dist_now.log_prob(batch['a_n'][index].to(self.device))  # a_logprob_n_now.shape=(mini_batch_size, episode_limit, N)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_n_now - batch['a_logprob_n'][index].detach().to(self.device))  # ratios.shape=(mini_batch_size, episode_limit, N)
                surr1 = ratios * adv[index].to(self.device)
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index].to(self.device)
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

                if self.use_value_clip:
                    values_old = batch["v_n"][index, :-1].detach().to(self.device)
                    values_error_clip = torch.clamp(values_now - values_old, -self.epsilon, self.epsilon) + values_old - v_target[index].to(self.device)
                    values_error_original = values_now - v_target[index].to(self.device)
                    critic_loss = torch.max(values_error_clip ** 2, values_error_original ** 2)
                else:
                    critic_loss = (values_now - v_target[index]) ** 2

                self.ac_optimizer.zero_grad()
                ac_loss = actor_loss.mean() + critic_loss.mean()
                ac_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.ac_parameters, 10.0)
                self.ac_optimizer.step()

        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):  # Trick 6: learning rate Decay
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for p in self.ac_optimizer.param_groups:
            p['lr'] = lr_now

    def get_inputs(self, batch):
        actor_inputs, order_inputs, POI_inputs, region_inputs, critic_inputs = [], [], [], [], []
        actor_inputs.append(batch['obs_n'])
        order_inputs.append(batch['obs_order'])
        POI_inputs.append(batch['obs_POI'])
        region_inputs.append(batch['obs_region'])
        critic_inputs.append(batch['s'].unsqueeze(2).repeat(1, 1, self.N, 1))
        if self.add_agent_id:
            # agent_id_one_hot.shape=(mini_batch_size, max_episode_len, N, N)
            agent_id_one_hot = torch.eye(self.N).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, self.episode_limit, 1, 1)
            actor_inputs.append(agent_id_one_hot)
            critic_inputs.append(agent_id_one_hot)

        actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)  # actor_inputs.shape=(batch_size, episode_limit, N, actor_input_dim)
        critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)  # critic_inputs.shape=(batch_size, episode_limit, N, critic_input_dim)
        order_inputs = torch.cat([x for x in order_inputs], dim=-1)
        POI_inputs = torch.cat([x for x in POI_inputs], dim=-1)
        region_inputs = torch.cat([x for x in region_inputs], dim=-1)
        return actor_inputs, order_inputs, POI_inputs, region_inputs, critic_inputs

    def save_model(self, env_name, number, seed, total_steps):
        torch.save(self.actor.state_dict(), "./model/MAPPO_actor_env_{}_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, int(total_steps / 1000)))

    def load_model(self, env_name, number, seed, step):
        self.actor.load_state_dict(torch.load("./model/MAPPO_actor_env_{}_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, step)))