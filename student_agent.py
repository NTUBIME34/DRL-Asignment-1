import numpy as np
import random
from tensorflow.keras.models import load_model

# 載入預先訓練好的 DQN 模型
model = load_model("dqn_model.h5")

def get_action(obs):
    """
    接收環境的觀察值 obs（長度16的 tuple），
    利用 DQN 模型預測各動作的 Q 值，
    返回 Q 值最大的動作（0~5）。
    """
    state = np.array(obs).reshape(1, -1)
    q_values = model.predict(state, verbose=0)
    action = int(np.argmax(q_values))
    return action
