import numpy as np
import tensorflow as tf
from builtNN import build_net

np.random.seed(1)
tf.set_random_seed(1)

class DQNAgent:
    def __init__(
            self,
            n_actions,
            n_features,
            hidden,
            learning_rate=0.001,
            reward_decay=0.9,
            e_init = 0,
            e_greedy=0.9,
            replace_target_iter=200,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            dueling=False,
            double_dqn = False,
            sess=None,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.hidden = hidden
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = e_init if e_greedy_increment is not None else self.epsilon_max

        self.dueling = dueling      # decide to use dueling DQN or not
        self.double_dqn = double_dqn    # decide to use double DQN or not

        # build neural network
        NN = build_net(self.n_features, self.n_actions, self.hidden, self.dueling, self.lr)
        self.s, self.eval_model = NN.build_evl_net()
        self.s_next, self.next_model = NN.build_target_net()
        self.target_model, self.loss, self.train = NN.build_train()

        self.step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features*2+3))
        target_params = tf.get_collection('target_net_params')
        eval_params = tf.get_collection('eval_net_params')
        self.replace_target = [tf.assign(t, e) for t, e in zip(target_params, eval_params)]

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

    def record_step(self, s, a, r, done, s_next):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        index = self.memory_counter % self.memory_size
        step = np.hstack((s, [a, r, done], s_next))
        self.memory[index, :] = step

        self.memory_counter += 1

    def select_action(self, obs):
        obs = obs[np.newaxis, :]
        if np.random.uniform() < self.epsilon:  # choosing action
            actions_value = self.sess.run(self.eval_model, feed_dict={self.s: obs})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target)
            # print('\ntarget_params_replaced\n')

        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        next_model = self.sess.run(self.next_model, feed_dict={self.s_next: batch_memory[:, -self.n_features:]}) # next observation
        eval_model_next = self.sess.run(self.eval_model, feed_dict={self.s: batch_memory[:, -self.n_features:]})
        eval_model = self.sess.run(self.eval_model, {self.s: batch_memory[:, :self.n_features]})

        target_model = eval_model.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        done = batch_memory[:, self.n_features + 2]

        if self.double_dqn:
            max_next_action = np.argmax(eval_model_next, axis=1)
            opt_q_next = next_model[batch_index, max_next_action]
        else:
            opt_q_next = np.max(next_model, axis=1)

        target_model[batch_index, eval_act_index] = reward + self.gamma * opt_q_next * (1-done)

        _, self.cost = self.sess.run([self.train, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.target_model: target_model})

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.step_counter += 1

