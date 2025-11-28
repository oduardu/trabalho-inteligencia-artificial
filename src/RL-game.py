import os
import pickle
import random
import sys
from collections import defaultdict

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces

# Importações para Stable Baselines3 (A2C)
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

from config import (
    # ALPHA,
    # ALPHA_DECAY,
    # DECAY_STEP,
    ENT_COEF,
    # EPSILON,
    # EPSILON_DECAY,
    GAMMA,
    GOAL_REWARD,
    HOLE_REWARD,
    LR,
    N_STEPS,
    RENDERS,
    SIMMULATION_NUMBER,
    STEP_REWARD,
    TRAIN,
    WALL_REWARD,
)


class MazeEnv(gym.Env):
    def __init__(self):
        super(MazeEnv, self).__init__()

        # Define grid size
        self.height = 15
        self.width = 15

        self.turn = 0

        # Define elements
        self.EMPTY = 0
        self.WALL = 1
        self.HOLE = 2
        self.AGENT = 3
        self.GOAL = 4
        self.agent_pos = None
        self.goal_pos = None
        self.goal_id = None

        # Initialize visualization
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        plt.ion()

        # Action space: up, right, down, left
        self.action_space = spaces.Discrete(4)
        # Observation space: single number representing state
        self.observation_space = spaces.MultiDiscrete([5, 5, 5, 5, 4, 15 * 15])

        # Create grid
        self.grid = None
        self.grid_initialization()

    def grid_initialization(self):
        self.turn = 0

        # Create grid
        self.grid = np.zeros((self.height, self.width))

        # Set agent, enemy and goal position
        self.agent_pos = [
            random.randint(2, self.height - 3),
            random.randint(2, self.width - 3),
        ]

        self.goal_id = random.randint(1, 4)
        if self.goal_id == 1:
            self.goal_pos = [1, 1]
        elif self.goal_id == 2:
            self.goal_pos = [1, self.width - 2]
        elif self.goal_id == 3:
            self.goal_pos = [self.height - 2, 1]
        else:
            self.goal_pos = [self.height - 2, self.width - 2]

        # Add border walls
        self.grid[0 : self.height, 0] = self.WALL
        self.grid[0, 0 : self.width] = self.WALL
        self.grid[self.height - 1, 0 : self.width] = self.WALL
        self.grid[0 : self.height, self.width - 1] = self.WALL
        # Set initial positions
        self.grid[self.agent_pos[0], self.agent_pos[1]] = self.AGENT
        self.grid[self.goal_pos[0], self.goal_pos[1]] = self.GOAL
        # Add inner walls
        for i in range(1, self.height):
            for j in range(1, self.width):
                if self.grid[i][j] == self.EMPTY:
                    r = random.random()
                    if r < 0.2:
                        self.grid[i][j] = self.WALL
                    elif r < 0.25:
                        self.grid[i][j] = self.HOLE

    def get_state(self, isnumpy=True):
        up_view = self.grid[self.agent_pos[0] - 1][self.agent_pos[1]]
        down_view = self.grid[self.agent_pos[0] + 1][self.agent_pos[1]]
        left_view = self.grid[self.agent_pos[0]][self.agent_pos[1] - 1]
        right_view = self.grid[self.agent_pos[0]][self.agent_pos[1] + 1]
        agent_position = self.agent_pos[0] * self.width + self.agent_pos[1]
        observation = (
            up_view,
            down_view,
            left_view,
            right_view,
            self.goal_id - 1,
            agent_position,
        )
        if isnumpy:
            # Retorna como np.array de int32 para SB3
            return np.array(observation, dtype=np.int32)
        else:
            return observation

    def step(self, action, isnumpy=True):
        # Add 1 turn
        self.turn += 1

        # Save previous position
        prev_pos = self.agent_pos.copy()

        # Move agent
        if action == 0:  # up
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # right
            self.agent_pos[1] = min(self.width - 1, self.agent_pos[1] + 1)
        elif action == 2:  # down
            self.agent_pos[0] = min(self.height - 1, self.agent_pos[0] + 1)
        elif action == 3:  # left
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)

        # Check if new position is valid
        new_pos_value = self.grid[self.agent_pos[0], self.agent_pos[1]]

        # Define rewards and terminal states
        terminated = False
        truncated = False
        reward = STEP_REWARD()  # small negative reward for each step

        if new_pos_value == self.WALL:
            self.agent_pos = prev_pos  # revert move
            reward = WALL_REWARD()
        elif new_pos_value == self.HOLE:
            terminated = True
            reward = HOLE_REWARD()
        elif self.agent_pos == self.goal_pos:
            terminated = True
            reward = GOAL_REWARD()

        # Update grid
        if new_pos_value != self.WALL:
            self.grid[prev_pos[0], prev_pos[1]] = self.EMPTY
            self.grid[self.agent_pos[0], self.agent_pos[1]] = self.AGENT

        if self.turn == 200:
            truncated = True
            return self.get_state(isnumpy), reward, terminated, truncated, {}

        return self.get_state(isnumpy), reward, terminated, truncated, {}

    def reset(self, *, seed=None, options=None, return_info=False, isnumpy=True):
        super().reset(seed=seed)
        # Reset game to initial state
        self.turn = 0
        self.grid_initialization()

        observation = self.get_state(isnumpy)
        info = {}
        # Retorno para o formato Gymnasium (obs, info)
        return observation, info

    def render(self):
        self.ax.clear()
        # Define colors for each element
        colors = {
            self.EMPTY: "white",
            self.WALL: "gray",
            self.HOLE: "black",
            self.AGENT: "blue",
            self.GOAL: "green",
        }

        name = {
            self.EMPTY: "Vazio",
            self.WALL: "Parede",
            self.HOLE: "Buraco",
            self.AGENT: "Agente",
            self.GOAL: "Objetivo",
        }

        # Create color map
        cmap = plt.cm.colors.ListedColormap(list(colors.values()))
        # Plot the grid
        self.ax.imshow(self.grid, cmap=cmap)

        # Add legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=color, label=name[key])
            for key, color in colors.items()
        ]
        self.ax.legend(
            handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5)
        )

        plt.axis("off")
        plt.pause(0.1)
        self.fig.canvas.draw()


# ====================================================================
# CLASSE E LÓGICA Q-LEARNING (COMENTADA)
# ====================================================================

# class QLearningAgent:
#     def __init__(self, action_space, learning_rate, discount_factor, epsilon):
#         self.q_table = defaultdict(lambda: np.zeros(action_space.n))
#         self.alpha = learning_rate
#         self.gamma = discount_factor
#         self.epsilon = epsilon
#         self.action_space = action_space

#     def save_model(self, filename):
#         with open(filename, "wb") as f:
#             pickle.dump(dict(self.q_table), f)

#     def load_model(self, filename):
#         if not os.path.exists(filename):
#             raise FileNotFoundError(
#                 f"Modelo não encontrado: 'q_learning_model.pkl'. Por favor, treine o modelo primeiro (configure TRAIN = True no config.py)."
#             )
#         with open(filename, "rb") as f:
#             self.q_table = defaultdict(
#                 lambda: np.zeros(self.action_space.n), pickle.load(f)
#             )

#     def get_action(self, state):
#         if random.random() < self.epsilon:
#             return self.action_space.sample()

#         q_values = self.q_table[state]
#         exp_q = np.exp(q_values - np.max(q_values))
#         probs = exp_q / np.sum(exp_q)
#         return np.random.choice(len(q_values), p=probs)

#     def update(self, state, action, reward, next_state):
#         old_value = self.q_table[state][action]
#         next_max = np.max(self.q_table[next_state])
#         new_value = (1 - self.alpha) * old_value + self.alpha * (
#             reward + self.gamma * next_max
#         )
#         self.q_table[state][action] = new_value


# # ---------------------------------------------------------------------------------------------------
# # BLOCO DE TREINO Q-LEARNING (COMENTADO)
# # ---------------------------------------------------------------------------------------------------

# if TRAIN() and not os.path.exists(os.path.join(os.path.dirname(__file__), "modelos", "a2c")): # Condição ajustada para não conflitar com o treino A2C
#     env = MazeEnv()
#     # O Q-Learning usaria ALPHA e EPSILON do config.py
#     agent = QLearningAgent(env.action_space, ALPHA(), GAMMA(), EPSILON())

#     episodes = SIMMULATION_NUMBER()
#     total_reward = 0
#     sucess = 0
#     for episode in range(1, episodes + 1):
#         state, _ = env.reset(isnumpy=False)
#         done = False
#         if RENDERS():
#             env.render()

#         while not done:
#             action = agent.get_action(state)
#             next_state, reward, terminated, truncated, _ = env.step(action, isnumpy=False)
#             done = terminated or truncated # Atualizado para a lógica Gymnasium

#             agent.update(state, action, reward, next_state)
#             state = next_state
#             total_reward += reward
#             if reward == GOAL_REWARD():
#                 sucess += 1
#             if RENDERS():
#                 env.render()

#         # Parameter Decay
#         if episode % DECAY_STEP() == 0:
#             agent.epsilon *= EPSILON_DECAY()
#             agent.alpha *= ALPHA_DECAY()
#             print(
#                 f"Episode {episode}, Mean Reward: {(total_reward / DECAY_STEP()):.2f}, Success Rate: {(sucess / DECAY_STEP()):.2f}"
#             )
#             print("Explore Chance (epsilon): ", agent.epsilon)
#             print("Exploit Chance (1-epsilon): ", 1 - agent.epsilon)
#             print("Learning Rate (alpha): ", agent.alpha)
#             total_reward = 0
#             sucess = 0

#     # Save the trained agent
#     model_path = os.path.join(os.path.dirname(__file__), "q_learning_model.pkl")
#     agent.save_model(model_path)
#     print(f"Modelo salvo em: {model_path}")

# # ---------------------------------------------------------------------------------------------------
# # BLOCO DE TESTE Q-LEARNING (COMENTADO)
# # ---------------------------------------------------------------------------------------------------

# elif not TRAIN() and os.path.exists(os.path.join(os.path.dirname(__file__), "q_learning_model.pkl")): # Condição ajustada
#     env = MazeEnv()

#     # Load the trained agent
#     if len(sys.argv) > 1:
#         model_path = sys.argv[1]
#     else:
#         model_path = os.path.join(os.path.dirname(__file__), "q_learning_model.pkl")
#     agent = QLearningAgent(env.action_space, ALPHA(), GAMMA(), 0)

#     try:
#         agent.load_model(model_path)
#     except FileNotFoundError as e:
#         print(e)
#         exit()

#     # --- Lógica para Testar com 1000 Simulações (Requisito 2) ---
#     NUM_TEST_EPISODES = 1000
#     test_rewards = []
#     test_successes = 0

#     should_render = RENDERS()

#     if should_render:
#         print("Atenção: Rodando 1000 testes sem renderização para velocidade. Uma simulação será renderizada ao final.")

#     print(f"Iniciando teste de {NUM_TEST_EPISODES} episódios. Pode levar um momento...")

#     for test_episode in range(NUM_TEST_EPISODES):
#         state, _ = env.reset(isnumpy=False)
#         done = False
#         episode_reward = 0

#         while not done:
#             action = agent.get_action(state)
#             state, reward, terminated, truncated, _ = env.step(action, isnumpy=False)
#             done = terminated or truncated
#             episode_reward += reward

#         test_rewards.append(episode_reward)
#         if reward == GOAL_REWARD():
#             test_successes += 1

#         if (test_episode + 1) % 100 == 0:
#             print(f"Testando episódio {test_episode + 1}/{NUM_TEST_EPISODES}")

#     mean_reward = np.mean(test_rewards)
#     success_rate_decimal = test_successes / NUM_TEST_EPISODES

#     print("\n" + "=" * 40)
#     print("--- Resultados do Teste (1000 Simulações) ---")
#     print(f"Recompensa Média: {mean_reward:.2f}")
#     print(f"Taxa de Sucesso: {success_rate_decimal:.2f} (ou {success_rate_decimal * 100:.2f}%)")
#     print("=" * 40 + "\n")

#     # --- SIMULAÇÃO VISUAL (Se RENDERS = True) ---
#     if should_render:
#         print("\nRodando uma simulação visual do modelo treinado...")
#         state, _ = env.reset(isnumpy=False)
#         done = False
#         env.render()

#         while not done:
#             action = agent.get_action(state)
#             state, reward, terminated, truncated, _ = env.step(action, isnumpy=False)
#             done = terminated or truncated
#             print(f"Action: {action}, Reward: {reward}")
#             env.render()
# else: # Este 'else' captura qualquer caso não coberto pelo Q-Learning, permitindo que a lógica A2C funcione.


# -------------------------------------------------------------------------------------------------------
# LÓGICA A2C ATIVA
# -------------------------------------------------------------------------------------------------------

# Configurações do Stable Baselines3 (SB3)
NUM_ENVS = 4  # Número de ambientes paralelos (VecEnv) para acelerar o A2C
POLICY = "MlpPolicy"  # Política padrão para espaços vetoriais como o MultiDiscrete
MODEL_ID = 1  # ID padrão para o modelo único de treino/teste


# Funções auxiliares para salvar e carregar modelos A2C
def save_a2c_model(model, identifier):
    # Caminho ajustado para 'modelos/a2c'
    model_dir = os.path.join(os.path.dirname(__file__), "modelos", "a2c")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"a2c_model_{identifier}.zip")
    model.save(model_path)
    return model_path


def load_a2c_model(model_path, env):
    return A2C.load(model_path, env=env, verbose=0)


if TRAIN():
    # Bloco de TREINO A2C

    lr = LR()
    n_steps = N_STEPS()
    gamma = GAMMA()
    ent_coef = ENT_COEF()

    print("-" * 40)
    print(f"Treinando Modelo A2C ID: {MODEL_ID} (Usando Config.py)")
    print(f"LR: {lr}, n_steps: {n_steps}, Gamma: {gamma}, Ent_coef: {ent_coef}")

    env_train = make_vec_env(lambda: MazeEnv(), n_envs=NUM_ENVS, seed=0)

    model = A2C(
        POLICY,
        env_train,
        learning_rate=lr,
        n_steps=n_steps,
        gamma=gamma,
        ent_coef=ent_coef,
        verbose=0,
    )

    model.learn(total_timesteps=SIMMULATION_NUMBER())

    save_a2c_model(model, MODEL_ID)

    print("\nTreinamento A2C concluído. Modelo salvo.")


else:
    # Bloco de TESTE A2C (APENAS MODELO 1)

    NUM_TEST_EPISODES = 1000
    MODEL_DIR_PATH = os.path.join(os.path.dirname(__file__), "modelos", "a2c")

    env_test = MazeEnv()
    model_id = 1
    model_name = f"a2c_model_{model_id}.zip"
    model_path = os.path.join(MODEL_DIR_PATH, model_name)

    print("\n" + "=" * 60)
    print(f"Iniciando Teste do Único Modelo A2C (ID: {model_id}, 1000 Simulações)...")

    try:
        model_to_test = load_a2c_model(model_path, env_test)
    except Exception as e:
        print(
            f"ERRO: Modelo {model_name} não encontrado. Verifique se ele está em '{MODEL_DIR_PATH}'. Detalhes: {e}"
        )
        sys.exit(1)

    print("-" * 40)
    print(f"Testando Modelo ID: {model_id} ({model_name})")

    test_successes = 0
    test_rewards_list = []

    for _ in range(NUM_TEST_EPISODES):
        state, _ = env_test.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = model_to_test.predict(state, deterministic=True)

            # CORREÇÃO DO ERRO INDEXERROR
            if isinstance(action, np.ndarray):
                action_value = action.item()
            else:
                action_value = action

            state, reward, terminated, truncated, _ = env_test.step(action_value)
            done = terminated or truncated
            episode_reward += reward

        test_rewards_list.append(episode_reward)
        if reward == GOAL_REWARD():
            test_successes += 1

    # Calcular as métricas
    mean_reward_final = np.mean(test_rewards_list)
    success_rate_decimal = test_successes / NUM_TEST_EPISODES

    print(f"Recompensa Média: {mean_reward_final:.2f}")
    print(
        f"Taxa de Sucesso: {success_rate_decimal:.3f} (ou {success_rate_decimal * 100:.3f}%)"
    )

    print("=" * 60 + "\n")

    # ----------------------------------------------------
    # GERAÇÃO DA TABELA FINAL (Apenas Modelo 1)
    # ----------------------------------------------------

    print(f"## 📊 Resultados do Teste (Modelo {model_id} - 1000 Simulações)")
    print("| ID | Recompensa Média | Taxa de Sucesso |")
    print("|:---:|:---:|:---:|")
    print(
        f"| {model_id} | {mean_reward_final:.2f} | {success_rate_decimal * 100:.3f}% |"
    )

    # --- SIMULAÇÃO VISUAL (Se RENDERS = True) ---
    if RENDERS():
        print(f"\nRodando simulação visual do modelo A2C (ID: {model_id})...")

        state, _ = env_test.reset()
        done = False
        env_test.render()

        while not done:
            action, _ = model_to_test.predict(state, deterministic=True)

            if isinstance(action, np.ndarray):
                action_value = action.item()
            else:
                action_value = action

            state, reward, terminated, truncated, _ = env_test.step(action_value)
            done = terminated or truncated
            print(f"Action: {action_value}, Reward: {reward}")
            env_test.render()
