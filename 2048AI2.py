import tensorflow as tf
from tensorflow.keras import layers
import random
import numpy as np
from math import log2


def create_initial_state():
    return np.array([0] * 16, dtype=np.float32).reshape(1, 16)


def right(board):
    r = 0
    for i in range(4):
        for j in range(3):
            if board[i][j] == board[i][j + 1] and board[i][j] != 0:
                r +=  log2(board[i][j]) ** 2
                board[i][j] = 0
                board[i][j + 1] = 2 * board[i][j + 1]
            elif board[i][j+1] == 0:
                board[i][j+1] = board[i][j]
                board[i][j] = 0
    return r, board
def left(board):
    r = 0
    for i in range(4):
        for j in range(3, 0, -1):
            if board[i][j] == board[i][j - 1] and board[i][j] != 0:
                r +=  log2(board[i][j]) ** 2
                board[i][j] = 0
                board[i][j - 1] = 2 * board[i][j - 1]
            elif board[i][j - 1] == 0:
                board[i][j - 1] = board[i][j]
                board[i][j] = 0
    return r, board
def down(board):
    r = 0
    for i in range(3):
        for j in range(4):
            if board[i][j] == board[i + 1][j] and board[i][j] != 0:
                r +=  log2(board[i][j]) ** 2
                board[i][j] = 0
                board[i + 1][j] = 2 * board[i + 1][j]
            elif board[i + 1][j] == 0:
                board[i + 1][j] = board[i][j]
                board[i][j] = 0
    return r, board
def up(board):
    r = 0
    for i in range(3, 0, -1):
        for j in range(4):
            if board[i][j] == board[i - 1][j] and board[i][j] != 0:
                r +=   log2(board[i][j]) ** 2
                board[i][j] = 0
                board[i - 1][j] = 2 * board[i - 1][j]
            elif board[i-1][j] == 0:
                board[i-1][j] = board[i][j]
                board[i][j] = 0
    return r, board






class MyEnv:

    def __init__(self):
        self.state = create_initial_state()
    def reset():
        return [0] * 16

    def step(self, board ,action):
        # изменяем состояние
        global reward
        if action == 0:
            move = up(board)
            reward += move[0]
            self.state = move[1]
        elif action == 1:
            move = right(board)
            reward += move[0]
            self.state = move[1]
        elif action == 2:
            move = down(board)
            reward += move[0]
            self.state = move[1]
            self.state = move[1]
        elif action == 3:
            move = right(board)
            reward += move[0]
            self.state = move[1]
        space = []
        for i in range(4):
            for j in range(4):
                if board[i][j] == 0:
                    space.append((i, j))
        if len(space) == 0:
            global done
            reward -= 100
            done = 1
        else:
            choice = random.choice(space)
            reward += 0.1
            board[choice[0]][choice[1]] = 2
                        
        # конвертация матрицы обратно в вектор
        flat = np.array(board, dtype=np.float32).reshape(1, 16)
        self.state = flat


        return self.state, reward, done, {}



# Политика: 16 входов → скрытый слой → 4 выхода (вероятности действий)
def create_policy_network():
    inputs = layers.Input(shape=(16,))   # входной вектор из 16 признаков

    x = layers.Dense(32, activation='relu')(inputs)  # скрытый слой
    outputs = layers.Dense(4, activation='softmax')(x)  # 4 действия

    model = tf.keras.Model(inputs, outputs)
    return model

policy = create_policy_network()
policy.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

@tf.function
def train_step(state, action, advantage):
    with tf.GradientTape() as tape:
        probs = policy(state, training=True)
        action_prob = tf.reduce_sum(probs * tf.one_hot(action, 4), axis=1)
        loss = -40 * tf.math.log(action_prob + 1e-8) * (advantage - sum(rewards) / len(rewards))

    grads = tape.gradient(loss, policy.trainable_variables)
    optimizer.apply_gradients(zip(grads, policy.trainable_variables))
    return loss
env = MyEnv()
rewards = [100]
total_rewards = []
for i in range(20000):
    state = np.array([0] * 16, dtype=np.float32).reshape(1, 16)  # начальное состояние
    done = False
    total_reward = 0
    while not done:
        reward = 0
        # 1. Сеть предсказывает вероятности действий
        probs = policy(state)

        # 2. Выбираем действие (по вероятностям)
        action = tf.argmax(probs[0]).numpy()

        # 3. Делаем шаг в среде
        next_state, reward, done, info = env.step(state.reshape(4, 4).tolist(), action)
        rewards.append(reward)
        total_reward += reward


        
        train_step(
            tf.convert_to_tensor(state, dtype=tf.float32),
            action,  # уже int32
            tf.convert_to_tensor(reward, dtype=tf.float32)
        )
        
        # 5. Переходим к следующему состоянию
        state = next_state
    total_rewards.append(total_reward)
    if i % 20 == 0:
        for j in range(4):
            print(state[0][4 * j:4 * j + 4])
            pass
        print(sum(total_rewards)/len(total_rewards))
        print()

