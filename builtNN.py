import tensorflow as tf

tf.set_random_seed(1)

class build_net:
    def __init__(self, n_features, n_actions, hidden, dueling, lr):
        self.n_features = n_features
        self.n_actions = n_actions
        self.dueling = dueling
        self.n_l1, self.n_l2, self.w_initializer, self.b_initializer = \
                hidden[0], hidden[1], tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers
        self.lr = lr

    def build_layers(self, s, collection_names, w_initializer, b_initializer):
        with tf.variable_scope('l1'):
            w1 = tf.get_variable('w1', [self.n_features, self.n_l1], initializer=w_initializer, collections=collection_names)
            b1 = tf.get_variable('b1', [1, self.n_l1], initializer=b_initializer, collections=collection_names)
            l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

        with tf.variable_scope('l2'):
            w2 = tf.get_variable('w2', [self.n_l1, self.n_l2], initializer=w_initializer, collections=collection_names)
            b2 = tf.get_variable('b2', [1, self.n_l2], initializer=b_initializer, collections=collection_names)
            l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

        if self.dueling:
            # Dueling DQN
            with tf.variable_scope('Value'):
                w3 = tf.get_variable('w3', [self.n_l2, 1], initializer=w_initializer, collections=collection_names)
                b3 = tf.get_variable('b3', [1, 1], initializer=b_initializer, collections=collection_names)
                self.V = tf.matmul(l2, w3) + b3

            with tf.variable_scope('Advantage'):
                w3 = tf.get_variable('w3', [self.n_l2, self.n_actions], initializer=w_initializer, collections=collection_names)
                b3 = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=collection_names)
                self.A = tf.matmul(l2, w3) + b3

            with tf.variable_scope('Q_value'):
                out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))     # Q = V(s) + A(s,a)
        else:
            with tf.variable_scope('Q_value'):
                w3 = tf.get_variable('w3', [self.n_l2, self.n_actions], initializer=w_initializer, collections=collection_names)
                b3 = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=collection_names)
                out = tf.matmul(l2, w3) + b3

        return out

    def build_evl_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        with tf.variable_scope('eval_net'):
            collection_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.eval_model = self.build_layers(self.s, collection_names, self.w_initializer, self.b_initializer)

        return self.s, self.eval_model

    def build_target_net(self):
        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            collection_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.next_model = self.build_layers(self.s_, collection_names, self.w_initializer, self.b_initializer)

        return self.s_, self.next_model

    def build_train(self):
        self.target_model = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_model, self.eval_model))
        with tf.variable_scope('train'):
            self.train = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        return self.target_model, self.loss, self.train
