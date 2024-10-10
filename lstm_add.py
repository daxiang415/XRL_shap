import time                 # to measure the computation time
import gym
from gym import spaces, core
import numpy as np
import random
import pandas as pd
import math
import os
import torch.nn as nn
from stable_baselines3 import TD3
from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
import stable_baselines3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from stable_baselines3.common.callbacks import EvalCallback
from gym.utils import seeding
import torch.nn.functional as F

class Takasago_ENV(gym.Env):
  """A building energy system operational optimization for OpenAI gym and takasago"""

  def __init__(self):

    super(Takasago_ENV, self).__init__()
    self.data = pd.read_csv('train.csv')
    self.data = self.data.rename(columns=lambda x: x.strip())

    self.action_space = spaces.Box(
      low=np.array([-1]), high=np.array([1]), dtype=np.float32)
    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(170,), dtype=np.float32)

    self.Max_pv = self.data.max()['Solar generation']
    self.Max_hour = 24
    self.Max_month = 12
    self.Max_power = self.data.max()['Equipment Electric Power']
    self.Max_price = self.data.max()['Electricity Pricing']
    self.Max_co2 = self.data.max()['kg_CO2']
    self.history_length = 24
    self.optimize_length = 168
    self.battery_change = 5
    self.Max_day_type = 7
    self.prediction_window = 24

    # 数据标记
    self.time = 0
    self.t = 0


  def _next_observation(self):


    future_frame = np.array([
      self.data.loc[self.current_step:self.current_step + self.prediction_window - 1, 'Hour'] / self.Max_hour,
      self.data.loc[self.current_step:self.current_step + self.prediction_window - 1, 'Month'] / self.Max_month,
      self.data.loc[self.current_step:self.current_step + self.prediction_window - 1, 'Day Type'] / self.Max_day_type,
      self.data.loc[self.current_step:self.current_step + self.prediction_window - 1, 'Equipment Electric Power'] / self.Max_power,
      self.data.loc[self.current_step:self.current_step + self.prediction_window - 1, 'Solar generation'] / self.Max_pv,
      self.data.loc[self.current_step:self.current_step + self.prediction_window - 1, 'Electricity Pricing'] / self.Max_price,
      self.data.loc[self.current_step:self.current_step + self.prediction_window - 1, 'kg_CO2'] / self.Max_co2,

    ])

    flattened_tensor = future_frame.reshape(-1)


    step_in_epo = (self.current_step - self.original_step) / self.optimize_length

    obs = np.append(flattened_tensor, self.battery_state)
    obs = np.append(obs, step_in_epo)

    return obs.astype(np.float32)

  def reset(self):
    # Reset the state of the environment to an initial state
    self.battery_state = np.random.uniform(0.2, 0.8)
    #self.battery_state = 0.5
    # Set the current step to a random point within the data frame
    self.current_step = random.randint(0, self.data.shape[0] - self.optimize_length -self.prediction_window-1)
    #self.current_step = 4596

    self.original_step = self.current_step

    return self._next_observation()

  def reward(self, error, battery, battery_action):
    u = 0  # 均值μ
    # u01 = -4
    sig = math.sqrt(1)  # 标准差δ
    deta = 5
    reward_1 = (deta * np.exp(-(error - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)) - 1


    rewards = reward_1
    return rewards

  def seed(self, seed=None):
      self.np_random, seed = seeding.np_random(seed)

      return [seed]

  def step(self, action):

    # 读取计算结果，计算reward
    if self.current_step - self.original_step > self.optimize_length - 2:
      done = True
    else:
      done = False

    current_battery = self.battery_state

    # 电池动作
    if (action[-1] * self.battery_change) / 6.4 + self.battery_state < 0:
      action[-1] = - self.battery_state  # 改变动作，只能放这么多电
      self.battery_state = 0

    elif (action[-1] * self.battery_change) / 6.4 + self.battery_state > 1:
      action[-1] = 1 - self.battery_state
      self.battery_state = 1

    else:
      self.battery_state = self.battery_state + (action[-1] * self.battery_change) / 6.4



    pv_gen = self.data.iloc[self.current_step]['Solar generation']
    # error 小于0表示要从电网买电
    error = pv_gen - self.data.iloc[self.current_step]['Equipment Electric Power'] - action[0] * self.battery_change

    if error < 0: # cost大于0表示从电网买电
        cost = - error * self.data.iloc[self.current_step]['Electricity Pricing']
    else:
        cost = 0

    penalty = -(1.0 + np.sign(cost) * self.battery_state)

    reward = penalty * np.abs(cost)




    #reward = self.reward(error, self.battery_state, action[0])
    #
    # if error < 0:
    #     reward_sale = -0.5 * 0
    # else:
    #     reward_sale = 0
    #
    # reward = reward + reward_sale


    #
    self.current_step += 1
    state = self._next_observation()

    return state, reward, done, {'step': self.current_step, 'error': error,
                                 'battery_action': action[0], 'battery_state': current_battery}


class AttentionGRU(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.gru = nn.GRU(input_size=6, hidden_size=128, num_layers=1, bidirectional=False, batch_first=True, dropout=0.5)
        self.attn_linear = nn.Linear(7, 64)
        self.fc = nn.Linear(64, features_dim)

    def forward(self, observations):
        batch_size = observations.shape[0]
        tensor_main = observations[:, :144]
        tensor_last = observations[:, 144:]
        tensor_main_reshaped = tensor_main.view(batch_size, 6, 24).transpose(1, 2)

        gru_output, _ = self.gru(tensor_main_reshaped)

        # Create the custom key
        key = torch.cat((tensor_main_reshaped[:, 0, :], tensor_last), dim=1)  # (batch_size, 7)
        key = self.attn_linear(key).unsqueeze(2)  # (batch_size, 32, 1)

        # Compute attention scores
        attn_scores = torch.bmm(gru_output, key).squeeze(2)  # (batch_size, sequence_length)

        # Calculate attention weights
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(2)  # (batch_size, sequence_length, 1)

        # Apply attention weights
        attn_applied = torch.bmm(gru_output.transpose(1, 2), attn_weights).squeeze(2)  # (batch_size, hidden_size)

        return self.fc(attn_applied)

class CustomLSTM(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        self.model = nn.LSTM(input_size=7 + 2, hidden_size=128, num_layers=1, bidirectional=False,
                             batch_first=True)


        # Compute shape by doing one forward pass
        # with torch.no_grad():
        #     lstm_output, (h_n, c_n) = self.model(
        #         torch.as_tensor(observation_space.sample()[None]).float())a

        n_flatten = 128 * 24

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim))

    def forward(self, observations):
        # 146个特征，前144个是24小时 * 6铺开的特征，后面2个是蓄电池状态和step in epidose

        batch_size = observations.shape[0]

        tensor_main = observations[:, :168]  # 形状 [1, 144]
        tensor_last_two = observations[:, 168:]  # 形状 [1, 1]

        tensor_main_reshaped = tensor_main.view(batch_size, 7, 24).transpose(1, 2)

        # 将后两个特征重复24次并reshape为 (1, 24, 2)
        tensor_last_two_repeated = tensor_last_two.repeat(24, 1).view(batch_size, 24, 2)

        # 将两个部分在最后一个维度上拼接
        new_tensor = torch.cat((tensor_main_reshaped, tensor_last_two_repeated), dim=2)



        lstm_output, _ = self.model(new_tensor)

            # current_output = self.current_model(current_input)

        lstm_output = lstm_output.reshape(batch_size, -1)
            #
            # lstm_output = torch.cat([lstm_output, current_output], dim=-1)

        return self.linear(lstm_output)


class Gru(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        self.fc1 = nn.GRUCell(170, features_dim)
        self.output = nn.Linear(features_dim, features_dim, bias=True)

    def forward(self, x):
        """Feed forward."""
        y = F.relu(self.fc1(x))
        return self.output(y)


policy_kwargs = dict(
        features_extractor_class=CustomLSTM,
        features_extractor_kwargs=dict(features_dim=1024),
    )


#
if __name__ == "__main__":

    train_env = Takasago_ENV()
    random_seed = 26


    train_env.seed(random_seed)

    train_env.action_space.seed(random_seed)


    def seed_everything(seed):
        torch.manual_seed(seed)  # Current CPU
        torch.cuda.manual_seed(seed)  # Current GPU
        np.random.seed(seed)  # Numpy module
        random.seed(seed)  # Python random module
        torch.backends.cudnn.benchmark = False  # Close optimization
        torch.backends.cudnn.deterministic = True  # Close optimization
        torch.cuda.manual_seed_all(seed)  # All GPU (Optional)


    #
    seed_everything(random_seed)


    
    model = SAC("MlpPolicy", train_env, verbose=1, learning_rate=0.0002, tensorboard_log='./td3_tensorboard/', policy_kwargs=policy_kwargs, seed=random_seed)
    model.learn(total_timesteps=10000 * 168)
    model.save("SAC_lstm补充")
    #del model
    #print("model:",model)
    model = SAC.load("DDPG")
    env = train_env


    # model = PPO.load("ppo_takasago")
    # 创建几个空列表获取结果
    PV_gen = []
    Power_demand = []
    batteray_usage = []
    bio_gen = []
    error = []
    battery_state = []
    pv_action = []
    reward_all = []

    obs = env.reset()
    done = False
    data = pd.read_csv('train.csv').rename(columns=lambda x: x.strip())
    df = pd.DataFrame(columns=['bio_gen', 'battery_usage', 'pv_gen', 'power', 'error', 'battery_state', 'pv_action', 'reward'])
    print('开始最终测试')
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        Power_demand.append(data.loc[info['step'] - 1, 'Whole'])
        PV_gen.append(info['pv_action'] * data.loc[info['step'] - 1, 'PV_output'])
        bio_gen.append(info['bio_action'] * 80)
        batteray_usage.append(- info['battery_action'] * env.battery_change)
        error.append(info['error'])
        battery_state.append(info['battery_state'])
        pv_action.append(info['pv_action'])
        reward_all.append(reward)
        if -20 < info['error'] < 0:
            #print('good')
            #print(info['error'])
            continue
        else:
            print('not good')
            print(info['error'])

    df['battery_usage'] = batteray_usage
    df['bio_gen'] = bio_gen

    df['pv_gen'] = PV_gen
    df['power'] = Power_demand
    df['error'] = error
    df['battery_state'] = battery_state
    df['pv_action'] = pv_action
    df['reward']  = reward_all

    df.to_csv('rl.csv')























  
  

