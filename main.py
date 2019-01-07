import gym
from cartpole_v1 import Cartpole
import numpy as np

env = gym.make('CartPole-v1')
env.seed(1)
np.random.seed(1)

N_A = env.action_space.n
N_S = env.observation_space.shape[0]
MEMORY_CAPACITY = 1000
TARGET_REP_ITER = 200
MAX_EPISODES = 500
E_INIT = 0.5
E_GREEDY = 0.95
E_INCREMENT = 0.00001
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 32
HIDDEN = [64, 64]
DUELING = False
DOUBLE_DQN = False


cartpole = Cartpole(env, N_A, N_S, HIDDEN, LR, E_INIT, E_GREEDY, GAMMA, BATCH_SIZE, TARGET_REP_ITER,
                          MEMORY_CAPACITY, E_INCREMENT, MAX_EPISODES, DUELING, DOUBLE_DQN)
# train the model
r_colllection, avg_r = cartpole.train()

# cartpole.run_trained_model()

env.close()
