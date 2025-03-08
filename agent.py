import gym
import numpy as np
import random
import pickle

# 環境設定
env = gym.make("Taxi-v3")

# Q-table 初始化
Q_table = {}

# 超參數
alpha = 0.1  # 學習率
gamma = 0.9  # 折扣因子
epsilon = 0.1  # 探索率
num_episodes = 20000 # 增加訓練次數

# Reward Shaping 函數
def reward_shaping(env, state, action, reward, next_state, done):
    """
    使用 Reward Shaping 來引導 Agent 學習。
    """
    # 提取狀態資訊 (Taxi-v3 的 encoded state)
    taxi_row, taxi_col, passenger_location, destination = env.decode(state)
    next_taxi_row, next_taxi_col, next_passenger_location, next_destination = env.decode(next_state)

    # 鼓勵接近乘客
    if passenger_location != 4 and passenger_location != taxi_row * 5 + taxi_col: #如果乘客不在車上
        current_distance = abs(taxi_row - (passenger_location // 5)) + abs(taxi_col - (passenger_location % 5))
        next_distance = abs(next_taxi_row - (passenger_location // 5)) + abs(next_taxi_col - (passenger_location % 5))
        if next_distance < current_distance:
            reward += 0.2  # 稍微獎勵接近乘客的行為

    # 鼓勵抵達目的地
    if passenger_location == 4: #如果乘客在車上
        current_distance = abs(taxi_row - (destination // 5)) + abs(taxi_col - (destination % 5))
        next_distance = abs(next_taxi_row - (destination // 5)) + abs(next_taxi_col - (destination % 5))
        if next_distance < current_distance:
            reward += 0.2  # 稍微獎勵接近目的地的行為
    
    #避免原地打轉
    if taxi_row == next_taxi_row and taxi_col == next_taxi_col and action in [0,1,2,3]:
        reward -= 0.1

    return reward


# 訓練迴圈
for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        # 探索或利用
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # 探索：隨機選擇動作
        else:
            state_str = str(state)
            if state_str in Q_table:
                action = np.argmax(Q_table[state_str])  # 利用：選擇最佳已知動作
            else:
                action = env.action_space.sample()  # 如果狀態未知，則隨機選擇動作

        # 執行動作
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Reward Shaping
        reward = reward_shaping(env, state, action, reward, next_state, done)

        total_reward += reward

        # 更新 Q-table
        state_str = str(state)
        next_state_str = str(next_state)

        if state_str not in Q_table:
            Q_table[state_str] = np.zeros(env.action_space.n)

        if next_state_str not in Q_table:
            Q_table[next_state_str] = np.zeros(env.action_space.n)

        old_value = Q_table[state_str][action]
        next_max = np.max(Q_table[next_state_str])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        Q_table[state_str][action] = new_value

        state = next_state

    if (episode + 1) % 100 == 0:
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
    
    # 衰減 epsilon
    epsilon = max(epsilon * 0.999, 0.01)

# 訓練完成後儲存 Q-table
with open("q_table.pkl", "wb") as f:
    pickle.dump(Q_table, f)

print("Q-table training finished and saved to q_table.pkl")
