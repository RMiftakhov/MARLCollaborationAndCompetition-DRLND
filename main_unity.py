# main function that sets up environments
# perform training loop
import torch

print(torch.__version__)
import envs
from buffer import ReplayBuffer
from maddpg import MADDPG
import torch
import numpy as np
from tensorboardX import SummaryWriter
import os
from utilities import transpose_list, transpose_to_tensor

# keep training awake
from workspace_utils import keep_awake


from unityagents import UnityEnvironment
import numpy as np


class UnityWrapper():
    def __init__(self, train_mode=True):
        #self.env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis.exe")
        self.env = UnityEnvironment(file_name="Tennis_Linux_NoVis/Tennis.x86_64")
        self.brain_name = self.env.brain_names[0]
        self.brain      = self.env.brains[self.brain_name]
        # reset the environment
        env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
        # number of agents
        self.num_agents = len(env_info.agents)
        print("Number of agents: {}".format(self.num_agents))
        # size of each action
        self.action_size = self.brain.vector_action_space_size
        # size of each state
        states = env_info.vector_observations
        self.state_size = states.shape[1]

    def reset(self, train_mode=True):
        obs_n = []
        env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
        next_state = env_info.vector_observations
        for agent in range(self.num_agents):
            obs_n.append(np.array(next_state[agent]))
        return np.stack(obs_n)

    def step(self, action):
        obs_n = []
        reward_n = []
        done_n = []

        env_info = self.env.step(action[0])[self.brain_name]
        next_state = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        for agent in range(self.num_agents):
            obs_n.append(np.array(next_state[agent]))
            reward_n.append(np.array(rewards[agent]))
            done_n.append(np.array(dones[agent]))
        return np.stack(obs_n), reward_n, done_n

    def close(self):
        self.env.close()


def main():
    #seeding()
    # number of parallel agents
    parallel_envs = 1
    # number of training episodes.
    # change this to higher number to experiment. say 30000.
    number_of_episodes = 10000
    episode_length = 100
    batchsize = 512
    # how many episodes to save policy and gif
    save_interval = 500
    t = 0

    # amplitude of OU noise
    # this slowly decreases to 0
    noise = 2
    noise_reduction = 0.9999

    # how many episodes before update
    episode_per_update = 1  # * parallel_envs

    log_path = os.getcwd() + "/log"
    model_dir = os.getcwd() + "/model_dir"

    os.makedirs(model_dir, exist_ok=True)

    torch.set_num_threads(4)

    env = UnityWrapper()
    # keep 5000 episodes worth of replay
    buffer = ReplayBuffer(int(5000 * episode_length))

    # initialize policy and critic
    maddpg = MADDPG(is_unity=True, in_actor=env.state_size, out_actor=env.action_size,
                    in_critic=2 * env.state_size + 2*env.action_size, discount_factor=0.999, tau=1e-3)
    logger = SummaryWriter(log_dir=log_path)
    agent0_reward = []
    agent1_reward = []

    # training loop
    # show progressbar
    import progressbar as pb
    widget = ['episode: ', pb.Counter(), '/', str(number_of_episodes), ' ',
              pb.Percentage(), ' ', pb.ETA(), ' ', pb.Bar(marker=pb.RotatingMarker()), ' ']

    timer = pb.ProgressBar(widgets=widget, maxval=number_of_episodes).start()

    # use keep_awake to keep workspace from disconnecting
    for episode in keep_awake(range(0, number_of_episodes, parallel_envs)):

        timer.update(episode)

        reward_this_episode = np.zeros((1, 2))
        all_obs = env.reset()  #
        obs = all_obs.reshape((1, all_obs.shape[0], all_obs.shape[1]))
        # obs = transpose_list(all_obs)
        obs_full = obs.reshape((1, -1))

        # for calculating rewards for this particular episode - addition of all time steps

        # save info or not
        save_info = ((episode) % save_interval < parallel_envs or episode == number_of_episodes - parallel_envs)
        frames = []
        tmax = 0

        # if save_info:
        #    frames.append(env.render('rgb_array'))

        for episode_t in range(episode_length):

            t += parallel_envs

            # explore = only explore for a certain number of episodes
            # action input needs to be transposed
            actions = maddpg.act(transpose_to_tensor(obs), noise=0.0)
            noise *= noise_reduction

            actions_array = torch.stack(actions).detach().numpy()
            #actions_array = np.clip(actions_array, -1, 1)
            # transpose the list of list
            # flip the first two indices
            # input to step requires the first index to correspond to number of parallel agents
            actions_for_env = np.rollaxis(actions_array, 1)

            # step forward one frame
            next_obs, rewards, dones = env.step(actions_for_env)
            rewards = np.reshape(np.asarray(rewards), (1, -1))
            dones = np.reshape(np.asarray(dones), (1, -1))
            next_obs = next_obs.reshape((1, next_obs.shape[0], next_obs.shape[1]))
            next_obs_full = next_obs.reshape((1, -1))

            # add data to buffer
            transition = (obs, obs_full, actions_for_env, rewards, next_obs, next_obs_full, dones)

            buffer.push(transition)

            reward_this_episode += rewards

            obs, obs_full = next_obs, next_obs_full

        # update once after every episode_per_update
        if len(buffer) > batchsize:
            for a_i in range(2):
                samples = buffer.sample(batchsize)
                maddpg.update(samples, a_i, logger)
            maddpg.update_targets()  # soft update the target network towards the actual networks

        for i in range(parallel_envs):
            agent0_reward.append(reward_this_episode[i, 0])
            agent1_reward.append(reward_this_episode[i, 1])
            # agent2_reward.append(reward_this_episode[i,2])

        logger.add_scalar('total/mean_episode_rewards', np.max(reward_this_episode), episode)

        if episode % 10 == 0 or episode == number_of_episodes - 1:
            avg_rewards = [np.mean(agent0_reward), np.mean(agent1_reward)]
            agent0_reward = []
            agent1_reward = []
            # agent2_reward = []
            for a_i, avg_rew in enumerate(avg_rewards):
                logger.add_scalar('agent%i/mean_episode_rewards' % a_i, avg_rew, episode)

        # saving model
        save_dict_list = []
        if save_info:
            for i in range(2):
                save_dict = {'actor_params': maddpg.maddpg_agent[i].actor.state_dict(),
                             'actor_optim_params': maddpg.maddpg_agent[i].actor_optimizer.state_dict(),
                             'critic_params': maddpg.maddpg_agent[i].critic.state_dict(),
                             'critic_optim_params': maddpg.maddpg_agent[i].critic_optimizer.state_dict()}
                save_dict_list.append(save_dict)

                torch.save(save_dict_list,
                           os.path.join(model_dir, 'episode-{}.pt'.format(episode)))

    env.close()
    logger.close()
    timer.finish()


if __name__ == '__main__':
    main()
