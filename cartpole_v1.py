from DDQN import DQNAgent
import numpy as np
from collections import deque
import tensorflow as tf

tf.set_random_seed(1)
np.random.seed(1)

class Cartpole:
    def __init__(self, env, N_A, N_S, HIDDEN, LR, E_INIT, E_GREEDY, GAMMA, BATCH_SIZE,
                 TARGET_REP_ITER, MEMORY_CAPACITY, E_INCREMENT, MAX_EPISODES, DUELING, DOUBLE_DQN):
        self.agent = DQNAgent(
            n_actions=N_A, n_features=N_S, hidden=HIDDEN, learning_rate=LR, e_init = E_INIT, e_greedy=E_GREEDY, reward_decay=GAMMA,
            batch_size=BATCH_SIZE, replace_target_iter=TARGET_REP_ITER,
            memory_size=MEMORY_CAPACITY, e_greedy_increment=E_INCREMENT, dueling=DUELING, double_dqn=DOUBLE_DQN)

        self.MAX_EPISODES = MAX_EPISODES
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.env = env
        self.saver = tf.train.Saver()

    def train(self):
        total_steps = 0
        reward_window = deque(maxlen=100)
        rewards_collection = []
        average_reward = []
        to_save = True
        for i_episode in range(self.MAX_EPISODES):
            s = self.env.reset()  # (coord_x, coord_y, vel_x, vel_y, angle, angular_vel, l_leg_on_ground, r_leg_on_ground)
            reward = 0
            while True:
                if total_steps > self.MAX_EPISODES: self.env.render()
                a = self.agent.select_action(s)
                s_next, r, done, _ = self.env.step(a)

                reward += r
                self.agent.record_step(s, a, r, done, s_next)
                if total_steps > self.MEMORY_CAPACITY:
                    self.agent.learn()
                if done:
                    reward_window.append(reward)       # save most recent score
                    rewards_collection.append(reward)
                    average_reward.append(np.mean(reward_window))

                    print('\rEpisode {}\tEpisode Reward: {:.2f}\tAverage Score: {:.2f}\tEpsilon: {:.3f}'.format(i_episode, reward, np.mean(reward_window),self.agent.epsilon))

                    # if np.mean(reward_window)>=200.0 and to_save:
                    #     model_name = "cartpole_model_200"
                    #     self.saver.save(self.agent.sess, './'+model_name)
                    #     to_save = False
                    break

                s = s_next
                total_steps += 1

        return rewards_collection, average_reward

    def run_trained_model(self):
        name = 'cartpole_model_240.meta'
        newsaver = tf.train.import_meta_graph(name)
        newsaver.restore(self.agent.sess, tf.train.latest_checkpoint('./'))
        for i_episode in range(100):
            s = self.env.reset()  # (coord_x, coord_y, vel_x, vel_y, angle, angular_vel, l_leg_on_ground, r_leg_on_ground)
            reward = 0
            while True:
                obs = s[np.newaxis,:]
                action_value = self.agent.sess.run(self.agent.eval_model, {self.agent.s: obs})
                a = np.argmax(action_value)
                # self.env.render()
                s_next, r, done, _ = self.env.step(a)

                reward += r

                if done:
                   print('\rEpisode {}\tEpisode Reward: {:.2f}'.format(i_episode, reward))
                   # print(reward)
                   break

                s = s_next
