# # Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
# # Remember to adjust your student ID in meta.xml

# import numpy as np
# import pickle
# import random
# import gym
# import gym_minigrid
# import imageio
# from IPython.display import Image
# """Import libraries"""
# import gym
# import numpy as np
# import matplotlib.pyplot as plt
# import random
# from IPython.display import clear_output
# from time import sleep
# from matplotlib import animation

# q_table = {}  # 這裡需要用你的訓練結果來初始化
# num_actions = 6  # Taxi-v3 的動作數量

# def agent_train():
#     """Initialize and validate the environment"""
#     env = gym.make("Taxi-v3", render_mode="rgb_array").env
#     state, _ = env.reset()

#     # Print dimensions of state and action space
#     print("State space: {}".format(env.observation_space))
#     print("Action space: {}".format(env.action_space))

#     # Sample random action
#     action = env.action_space.sample(env.action_mask(state))
#     next_state, reward, done, _, _ = env.step(action)

#     q_table = np.zeros([env.observation_space.n, env.action_space.n])

#     # Hyperparameters
#     alpha = 0.1  # Learning rate
#     gamma = 1.0  # Discount rate
#     epsilon = 0.1  # Exploration rate
#     num_episodes = 10000  # Number of episodes

#     # Output for plots
#     cum_rewards = np.zeros([num_episodes])
#     total_epochs = np.zeros([num_episodes])

#     for episode in range(1, num_episodes+1):
#         # Reset environment
#         state, info = env.reset()
#         epoch = 0 
#         num_failed_dropoffs = 0
#         done = False
#         cum_reward = 0

#         while not done:
            
#             if random.uniform(0, 1) < epsilon:
#                 "Basic exploration [~0.47m]"
#                 action = env.action_space.sample() # Sample random action (exploration)
                
#                 "Exploration with action mask [~1.52m]"
#             # action = env.action_space.sample(env.action_mask(state)) "Exploration with action mask"
#             else:      
#                 "Exploitation with action mask [~1m52s]"
#             # action_mask = np.where(info["action_mask"]==1,0,1) # invert
#             # masked_q_values = np.ma.array(q_table[state], mask=action_mask, dtype=np.float32)
#             # action = np.ma.argmax(masked_q_values, axis=0)

#                 "Exploitation with random tie breaker [~1m19s]"
#             #  action = np.random.choice(np.flatnonzero(q_table[state] == q_table[state].max()))
                
#                 "Basic exploitation [~47s]"
#                 action = np.argmax(q_table[state]) # Select best known action (exploitation)

#             next_state, reward, done, _ , info = env.step(action) 

#             cum_reward += reward
            
#             old_q_value = q_table[state, action]
#             next_max = np.max(q_table[next_state])
            
#             new_q_value = (1 - alpha) * old_q_value + alpha * (reward + gamma * next_max)
            
#             q_table[state, action] = new_q_value
            
#             if reward == -10:
#                 num_failed_dropoffs += 1

#             state = next_state
#             epoch += 1
            
#             total_epochs[episode-1] = epoch
#             cum_rewards[episode-1] = cum_reward

#         if episode % 100 == 0:
#             clear_output(wait=True)
#             print(f"Episode #: {episode}")
#     return q_table, cum_rewards, total_epochs


# q_table, rewards, epochs = agent_train()
# print(q_table)
# def get_action(obs):
#     # return random.choice([0, 1, 2, 3, 4, 5]) # Choose a random action

#     # TODO: Train your own agent
#     # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
#     # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
#     #       To prevent crashes, implement a fallback strategy for missing keys. 
#     #       Otherwise, even if your agent performs well in training, it may fail during testing.

#     # 將 obs 轉換成適合作為 key 的形式
#     # 如果 obs 原本就是整數，可以直接使用；如果不是，請做相應轉換

#     # 將 obs 轉換成整數型態作為索引
#     state_key = str(obs)
#     print(state_key)
#     # 檢查 state_key 是否落在 Q-table 的範圍內
#     if 0 <= state_key < q_table.shape[0]:
#         action = int(np.argmax(q_table[state_key]))
#     else:
#         # fallback 策略：隨機選擇一個動作
#         action = random.choice(range(num_actions))
#     return action

import numpy as np
import pickle
import random

# 嘗試載入預先訓練的 Q-table
try:
    with open("q_table.pkl", "rb") as f:
        Q_table = pickle.load(f)
    print("Q-table loaded successfully.")
except FileNotFoundError:
    Q_table = {}  # 如果檔案不存在，則初始化一個空的 Q-table
    print("Q-table not found. Initializing an empty Q-table.")
except Exception as e:
    Q_table = {}
    print(f"Error loading Q-table: {e}. Initializing an empty Q-table.")


def get_action(obs):
    """
    根據觀測值選擇最佳動作。
    如果 Q-table 中不存在該觀測值，則選擇隨機動作。
    """
    global Q_table
    # 將觀測值轉換為字串，以用作字典的鍵
    obs_str = str(obs)

    # 檢查 Q-table 中是否存在該觀測值
    if obs_str in Q_table:
        # 選擇具有最高 Q 值的動作
        action = np.argmax(Q_table[obs_str])
    else:
        # 如果觀測值不在 Q-table 中，則選擇隨機動作
        action = random.choice([0, 1, 2, 3, 4, 5])  # 動作空間為 0 到 5
        #print(f"Warning: Observation {obs_str} not found in Q-table. Choosing a random action.")

    return action

