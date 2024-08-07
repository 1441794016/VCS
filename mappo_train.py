import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from env import Env
from mappo_agent import MAPPO


class Runner_MAPPO:
    def __init__(self, args, env_name, number):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = args.seed
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Create env
        self.env = Env(args)  # Discrete action space
        self.args.N = self.env.agent_n  # The number of agents
        self.args.obs_dim_n = [self.env.observation_space for i in range(self.args.N)]  # obs dimensions of N agents
        self.args.action_dim_n = [self.env.action_space for i in range(self.args.N)]  # actions dimensions of N agents
        # Only for homogenous agents environments like Spread in MPE,all agents have the same dimension of observation space and action space
        self.args.obs_dim = self.args.obs_dim_n[0]  # The dimensions of an agent's observation space
        self.args.action_dim = self.args.action_dim_n[0]  # The dimensions of an agent's action space
        # self.args.state_dim = np.sum(self.args.obs_dim_n)  # The dimensions of global state space（Sum of the dimensions of the local observation space of all agents）
        self.args.state_dim = args.gnn_embedding_dim * self.args.N
        print("observation_space=", self.env.observation_space)
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        print("action_space=", self.env.action_space)
        print("action_dim_n={}".format(self.args.action_dim_n))
        print("device:", args.device)
        print("gpu id:", args.gpu_id)

        # Create N agents
        self.agent_n = MAPPO(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard
        self.writer = SummaryWriter(log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}'.format(self.env_name, self.number, self.seed))

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)

    def run(self, ):
        evaluate_num = -1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            _, episode_steps, _, _, _, _, _, _, _, _, _ = self.run_episode_mpe(evaluate=False)  # Run an episode
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                graph = self.env.get_graph()
                self.agent_n.train(self.replay_buffer, self.total_steps, graph)  # Training
                self.replay_buffer.reset_buffer()
        self.agent_n.save_model(self.env_name, self.number, self.seed, self.total_steps)
        self.evaluate_policy()
        self.env.close()

    def evaluate_policy(self, ):
        evaluate_reward = 0
        evaluate_income = 0
        evaluate_accepted_order_number = 0
        evaluate_overdue_order_number = 0
        evaluate_collected_POI_number = 0
        evaluate_overdue_POI_number = 0
        evaluate_total_data_vol = 0
        evaluate_total_AOI = 0
        evaluate_total_data_utility = 0
        for _ in range(self.args.evaluate_times):
            episode_reward, _, episode_order_number, episode_income, episode_accepted_order_number, \
                episode_overdue_order_number, episode_collected_POI_number, episode_overdue_POI_number, \
                episode_total_data_vol, episode_total_AOI, episode_total_data_utility = self.run_episode_mpe(evaluate=True)
            evaluate_reward += episode_reward
            evaluate_income += episode_income
            evaluate_accepted_order_number += episode_accepted_order_number
            evaluate_overdue_order_number += episode_overdue_order_number
            evaluate_collected_POI_number += episode_collected_POI_number
            evaluate_overdue_POI_number += episode_overdue_POI_number
            evaluate_total_data_vol += episode_total_data_vol
            evaluate_total_AOI += episode_total_AOI
            evaluate_total_data_utility += episode_total_data_utility

        if evaluate_collected_POI_number == 0:
            evaluate_ava_AOI = 0
        else:
            evaluate_ava_AOI = evaluate_total_AOI / evaluate_collected_POI_number

        evaluate_reward = evaluate_reward / float(self.args.evaluate_times)
        evaluate_income = evaluate_income / float(self.args.evaluate_times)
        evaluate_accepted_order_number = evaluate_accepted_order_number / float(self.args.evaluate_times)
        evaluate_collected_POI_number = evaluate_collected_POI_number / float(self.args.evaluate_times)
        evaluate_total_data_vol = evaluate_total_data_vol / float(self.args.evaluate_times)
        evaluate_total_data_utility = evaluate_total_data_utility / float(self.args.evaluate_times)

        self.evaluate_rewards.append(evaluate_reward)
        print("total_steps:{} \t evaluate_reward:{} \t evaluate_income:{} \t evalute_accepted_order_number:{} \t evaluate_collected_POI_number:{} \t evaluate_total_data_vol:{} \t evaluate_total_data_utility:{}".format(
            self.total_steps, evaluate_reward, evaluate_income, evaluate_accepted_order_number, evaluate_collected_POI_number, evaluate_total_data_vol, evaluate_total_data_utility))
        self.writer.add_scalar('evaluate_step_rewards_{}'.format(self.env_name), evaluate_reward, global_step=self.total_steps)
        self.writer.add_scalar('evaluate_step_income_{}'.format(self.env_name), evaluate_income,
                               global_step=self.total_steps)
        self.writer.add_scalar('evaluate_step_accepted_order_number_{}'.format(self.env_name),
                               evaluate_accepted_order_number, global_step=self.total_steps)
        self.writer.add_scalar('evaluate_step_total_data_utility_{}'.format(self.env_name), evaluate_total_data_utility,
                               global_step=self.total_steps)
        self.writer.add_scalar('evaluate_step_total_data_vol_{}'.format(self.env_name), evaluate_total_data_vol,
                               global_step=self.total_steps)
        self.writer.add_scalar('evaluate_step_ava_AOI_{}'.format(self.env_name), evaluate_ava_AOI,
                               global_step=self.total_steps)
        # Save the rewards and models

    def run_episode_mpe(self, evaluate=False):
        episode_reward = 0
        obs_n, obs_region, obs_order, obs_POI = self.env.reset()
        graph = self.env.get_graph()

        if self.args.use_reward_scaling:
            self.reward_scaling.reset()

        for episode_step in range(self.args.episode_limit):
            a_n, a_logprob_n, gnn_output = self.agent_n.choose_action(obs_n, obs_order, obs_POI, obs_region, graph, evaluate=evaluate)  # Get actions and the corresponding log probabilities of N agents
            s = np.array(gnn_output.cpu().numpy()).flatten()  # In MPE, global state is the concatenation of all agents' local obs.
            v_n = self.agent_n.get_value(s)  # Get the state values (V(s)) of N agents
            obs_next_n, obs_region_next, obs_order_next, obs_POI_next, r_n, done_n, _ = self.env.step(a_n)
            episode_reward += r_n.sum() / float(self.args.N)

            if not evaluate:
                if self.args.use_reward_norm:
                    r_n = self.reward_norm(r_n)
                elif args.use_reward_scaling:
                    r_n = self.reward_scaling(r_n)

                # Store the transition
                self.replay_buffer.store_transition(episode_step, obs_n, obs_order, obs_POI, obs_region, s, v_n, a_n, a_logprob_n, r_n, done_n)

            obs_n = obs_next_n
            obs_region = obs_region_next
            obs_order = obs_order_next
            obs_POI = obs_POI_next
            graph = self.env.get_graph()

            if all(done_n):
                break

        if not evaluate:
            # An episode is over, store v_n in the last step
            _, _, gnn_output = self.agent_n.choose_action(obs_n, obs_order, obs_POI, obs_region,
                                                          graph, evaluate=evaluate)
            s = np.array(gnn_output.cpu().numpy()).flatten()
            v_n = self.agent_n.get_value(s)
            self.replay_buffer.store_last_value(episode_step + 1, v_n)

        return episode_reward, episode_step + 1, self.env.order_number, self.env.income, \
            self.env.accepted_order_number, self.env.overdue_order_number, self.env.collected_POI_number, \
            self.env.overdue_POI_number, self.env.total_data_vol, self.env.total_AOI, self.env.total_data_utility


if __name__ == '__main__':
    import dgl

    u = torch.load("./dataset/graph/u_140.pt")
    v = torch.load("./dataset/graph/v_140.pt")
    region_graph = dgl.graph((u, v))

    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(4e6), help=" Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=100, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=3200, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=10, help="Evaluate times")

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the rnn")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the mlp")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling. Here, we do not use it.")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=float, default=False, help="Whether to use relu, if False, we will use tanh")
    # parser.add_argument("--use_rnn", type=bool, default=False, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=float, default=True, help="Whether to add agent_id. Here, we do not use it.")
    parser.add_argument("--use_value_clip", type=float, default=True, help="Whether to use value clip.")

    parser.add_argument("--use_gnn", type=float, default=0, help="Whether to use GNN")
    parser.add_argument("--gnn_hidden_dim", type=int, default=64, help="GNN hidden dim")
    parser.add_argument("--gnn_embedding_dim", type=int, default=10, help="GNN embedding dim")
    parser.add_argument("--region_graph", default=region_graph, help="region_graph")
    parser.add_argument("--order_data_path", default="./dataset/5-10.csv", help="Dataset path.")
    parser.add_argument("--beta", type=float, default=0.07, help="Reward weight beta.")
    parser.add_argument("--omega", type=float, default=1.4, help="Reward weight omega.")
    parser.add_argument("--agent_n", type=int, default=100, help="Number of agents.")
    parser.add_argument("--region_n", type=int, default=140, help="Number of regions.")
    parser.add_argument("--seed", type=float, default=0, help="Random seed.")
    parser.add_argument("--device", default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        help="Use gpu or not.")
    parser.add_argument("--gpu_id", type=str, default="2", help="GPU id.")
    parser.add_argument("--u", default=u, help="Source.")
    parser.add_argument("--v", default=v, help="Des.")

    args = parser.parse_args()
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    runner = Runner_MAPPO(args, env_name="withgnn_seed_{}_lr_{}_omega_{}_time_slot_100_normal_distribution_agent_number_{}".format(args.seed, args.lr, args.omega, args.agent_n), number=4)
    runner.run()