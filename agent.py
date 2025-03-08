import numpy as np
import random
from collections import deque
import gym
from simple_custom_taxi_env import SimpleTaxiEnv  # 請確保此檔案在相同目錄下
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 超參數設定
state_size = 16      # 觀察值長度
action_size = 6      # 動作空間大小
batch_size = 32
n_episodes = 500     # 訓練回合數，可根據需求調整
gamma = 0.99         # 折扣因子
epsilon = 1.0        # 初始探索率
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001

# 經驗回放記憶庫
memory = deque(maxlen=2000)

def build_model():
    model = Sequential()
    model.add(Dense(64, input_dim=state_size, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    return model

def train_model(model):
    global epsilon
    env = SimpleTaxiEnv(fuel_limit=5000)
    for e in range(n_episodes):
        state, _ = env.reset()
        state = np.array(state)
        total_reward = 0
        done = False
        
        while not done:
            # 探索策略：epsilon-greedy
            if np.random.rand() <= epsilon:
                action = random.randrange(action_size)
            else:
                q_values = model.predict(state.reshape(1, -1), verbose=0)
                action = int(np.argmax(q_values[0]))
            
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.array(next_state)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # 當記憶庫有足夠資料後進行小批量訓練
            if len(memory) >= batch_size:
                minibatch = random.sample(memory, batch_size)
                states = np.array([sample[0] for sample in minibatch])
                actions = np.array([sample[1] for sample in minibatch])
                rewards = np.array([sample[2] for sample in minibatch])
                next_states = np.array([sample[3] for sample in minibatch])
                dones = np.array([sample[4] for sample in minibatch])
                
                # 取得下個狀態的最大 Q 值
                target_q = rewards + gamma * np.amax(model.predict(next_states, verbose=0), axis=1) * (1 - dones)
                target_f = model.predict(states, verbose=0)
                for i, act in enumerate(actions):
                    target_f[i][act] = target_q[i]
                model.fit(states, target_f, epochs=1, verbose=0)
        
        # 每回合結束後更新 epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        print(f"Episode {e+1}/{n_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")
    
    # 訓練完成後存檔
    model.save("dqn_model.h5")
    print("模型已儲存至 dqn_model.h5")

if __name__ == "__main__":
    model = build_model()
    train_model(model)
