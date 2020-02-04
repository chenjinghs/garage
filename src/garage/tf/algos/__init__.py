"""Tensorflow implementation of reinforcement learning algorithms."""
from garage.tf.algos.batch_polopt import BatchPolopt
from garage.tf.algos.batch_polopt2 import BatchPolopt2
from garage.tf.algos.ddpg import DDPG
from garage.tf.algos.dqn import DQN
from garage.tf.algos.erwr import ERWR
from garage.tf.algos.npo import NPO
from garage.tf.algos.npo2 import NPO2
from garage.tf.algos.ppo import PPO
from garage.tf.algos.reps import REPS
from garage.tf.algos.rl2npo import RL2NPO
from garage.tf.algos.rl2npo2 import RL2NPO2
from garage.tf.algos.rl2npo3 import RL2NPO3
from garage.tf.algos.rl2ppo import RL2PPO
from garage.tf.algos.rl2ppo2 import RL2PPO2
from garage.tf.algos.rl2ppo3 import RL2PPO3
from garage.tf.algos.rl2trpo import RL2TRPO
from garage.tf.algos.td3 import TD3
from garage.tf.algos.tnpg import TNPG
from garage.tf.algos.trpo import TRPO
from garage.tf.algos.vpg import VPG
from garage.tf.algos.rl2 import RL2

__all__ = [
    'BatchPolopt',
    'BatchPolopt2',
    'DDPG',
    'DQN',
    'ERWR',
    'NPO',
    'NPO2',
    'PPO',
    'REPS',
    'TD3',
    'TNPG',
    'TRPO',
    'VPG',
    'RL2',
    'RL2NPO',
    'RL2NPO2',
    'RL2NPO3',
    'RL2PPO',
    'RL2PPO2',
    'RL2PPO3',
    'RL2TRPO'
]
