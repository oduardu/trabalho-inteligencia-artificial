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
    ALPHA,
    ALPHA_DECAY,
    DECAY_STEP,
    ENT_COEF,
    EPSILON,
    EPSILON_DECAY,
    GAMMAQ,
    GAMMAA2C,
    GOAL_REWARD,
    HOLE_REWARD,
    LR,
    N_STEPS,
    RENDERS,
    SIMMULATION_NUMBER,
    STEP_REWARD,
    TRAIN,
    WALL_REWARD,
    TEST,
    MODEL
)

NUM_ENVS = 4
POLICY = "MlpPolicy"
A2C_MODEL_DIR_PATH = os.path.join(os.path.dirname(__file__), "modelos", "a2c")
Q_LEARNING_MODEL_DIR = os.path.join(os.path.dirname(__file__), "modelos", "q-learning")
DEFAULT_Q_FILENAME_BASE = "q_learning_model_"
DEFAULT_Q_FILENAME_EXT = ".pkl"


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


class QLearningAgent:
    def __init__(self, action_space, learning_rate, discount_factor, epsilon):
        self.q_table = defaultdict(lambda: np.zeros(action_space.n))
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.action_space = action_space

    def save_model(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load_model(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"Modelo não encontrado: '{filename}'. Por favor, treine o modelo primeiro (configure TRAIN = True no config.py)."
            )
        with open(filename, "rb") as f:
            self.q_table = defaultdict(
                lambda: np.zeros(self.action_space.n), pickle.load(f)
            )

    def get_action(self, state):
        if random.random() < self.epsilon:
            return self.action_space.sample()

        q_values = self.q_table[state]
        exp_q = np.exp(q_values - np.max(q_values))
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(len(q_values), p=probs)

    def update(self, state, action, reward, next_state):
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.alpha) * old_value + self.alpha * (
            reward + self.gamma * next_max
        )
        self.q_table[state][action] = new_value

existing_models = []
if os.path.exists(A2C_MODEL_DIR_PATH):
    for filename in os.listdir(A2C_MODEL_DIR_PATH):
        if filename.startswith("a2c_model_") and filename.endswith(".zip"):
            try:
                model_num = int(filename.replace("a2c_model_", "").replace(".zip", ""))
                existing_models.append(model_num)
            except ValueError:
                continue

def find_last_a2c_model_id():
    if existing_models:
        return max(existing_models)
    return 1

def find_next_a2c_model_id():
    if existing_models:
        return max(existing_models) + 1
    return 1


def get_existing_q_models_ids():
    q_models_ids = []
    if os.path.exists(Q_LEARNING_MODEL_DIR):
        for filename in os.listdir(Q_LEARNING_MODEL_DIR):
            if filename.startswith(DEFAULT_Q_FILENAME_BASE) and filename.endswith(DEFAULT_Q_FILENAME_EXT):
                try:
                    name_parts = filename.replace(DEFAULT_Q_FILENAME_BASE, '').replace(DEFAULT_Q_FILENAME_EXT, '')
                    if name_parts.isdigit():
                        q_models_ids.append(int(name_parts))
                except ValueError:
                    continue
    return sorted(q_models_ids)

def find_last_q_model_id():
    q_models_ids = get_existing_q_models_ids()
    if q_models_ids:
        return q_models_ids[-1]
    return 1

def find_next_q_model_id():
    q_models_ids = get_existing_q_models_ids()
    if q_models_ids:
        return q_models_ids[-1] + 1
    return 1


def save_a2c_model(model, identifier):
    os.makedirs(A2C_MODEL_DIR_PATH, exist_ok=True)
    model_path = os.path.join(A2C_MODEL_DIR_PATH, f"a2c_model_{identifier}.zip")
    model.save(model_path)
    return model_path


def load_a2c_model(model_path, env):
    return A2C.load(model_path, env=env, verbose=0)

A2C_MODEL_ID = None
Q_MODEL_ID = None

is_id_provided_by_argv = False
if len(sys.argv) > 1:
    try:
        A2C_MODEL_ID = int(sys.argv[1])
        Q_MODEL_ID = A2C_MODEL_ID
        print(f"Usando ID fornecido via argumento: {A2C_MODEL_ID}")
        is_id_provided_by_argv = True
    except ValueError:
        print("ERRO: O argumento deve ser um número inteiro para o MODEL_ID.")
        sys.exit(1)
else:
    if TRAIN():
        print("Nenhum ID fornecido. Usando Próximo ID para Treinamento.")
        A2C_MODEL_ID = find_next_a2c_model_id()
        Q_MODEL_ID = find_next_q_model_id()

    if not TRAIN() and (RENDERS() or TEST()):
        A2C_MODEL_ID = find_last_a2c_model_id()
        Q_MODEL_ID = find_last_q_model_id()
    elif not TRAIN():
        A2C_MODEL_ID = find_next_a2c_model_id()


RENDER_POST_TRAIN = RENDERS() and TRAIN()

if TRAIN() and MODEL() in ["q-learning", "both"]:
    env = MazeEnv()
    agent = QLearningAgent(env.action_space, ALPHA(), GAMMAQ(), EPSILON())

    episodes = SIMMULATION_NUMBER()
    total_reward = 0
    sucess = 0

    q_save_id = Q_MODEL_ID

    print("-" * 40)
    print(f"Treinando Modelo Q-Learning (Total de {episodes} passos). Salvando com ID: {q_save_id}")
    print(f"Alpha: {ALPHA()}, Gamma: {GAMMAQ()}, Epsilon Inicial: {EPSILON()}")

    for episode in range(1, episodes + 1):
        state, _ = env.reset(isnumpy=False)
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action, isnumpy=False)
            done = terminated or truncated

            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            if reward == GOAL_REWARD():
                sucess += 1

        # Parameter Decay
        if episode % DECAY_STEP() == 0:
            agent.epsilon *= EPSILON_DECAY()
            agent.alpha *= ALPHA_DECAY()
            print(f"--- Época {episode} ---")
            print(f"Epsilon: {agent.epsilon:.4f}, Alpha: {agent.alpha:.4f}")
            total_reward = 0
            sucess = 0

    q_model_filename = DEFAULT_Q_FILENAME_BASE + str(q_save_id) + DEFAULT_Q_FILENAME_EXT
    q_model_path = os.path.join(Q_LEARNING_MODEL_DIR, q_model_filename)
    agent.save_model(q_model_path)
    print(f"Modelo Q-Learning salvo em: {q_model_path}")

    if RENDER_POST_TRAIN and MODEL() in ["q-learning", "both"]:
        print("\n--- INICIANDO RENDERIZAÇÃO PÓS-TREINO Q-LEARNING ---")
        env_render_ql = MazeEnv()
        agent_render_ql = QLearningAgent(env_render_ql.action_space, ALPHA(), GAMMAQ(), 0)
        try:
            agent_render_ql.load_model(q_model_path)
        except FileNotFoundError:
            print("ERRO: Modelo recém-treinado não encontrado para renderização.")
        else:
            state, _ = env_render_ql.reset(isnumpy=False)
            done = False
            env_render_ql.render()
            while not done:
                action = agent_render_ql.get_action(state)
                state, reward, terminated, truncated, _ = env_render_ql.step(action, isnumpy=False)
                done = terminated or truncated
                print(f"Action: {action}, Reward: {reward}")
                env_render_ql.render()


if TRAIN() and MODEL() in ["a2c", "both"]:
    lr = LR()
    n_steps = N_STEPS()
    gamma_a2c = GAMMAA2C()
    ent_coef = ENT_COEF()

    print("-" * 40)
    print(f"Treinando Modelo A2C ID: {A2C_MODEL_ID} (Usando Config.py)")

    env_train = make_vec_env(lambda: MazeEnv(), n_envs=NUM_ENVS, seed=0)

    model = A2C(
        POLICY,
        env_train,
        learning_rate=lr,
        n_steps=n_steps,
        gamma=gamma_a2c,
        ent_coef=ent_coef,
        verbose=0,
    )

    model.learn(total_timesteps=SIMMULATION_NUMBER())

    a2c_model_path_saved = save_a2c_model(model, A2C_MODEL_ID)

    print("\nTreinamento A2C concluído. Modelo salvo.")

    if RENDER_POST_TRAIN and MODEL() in ["a2c", "both"]:
        print("\n--- INICIANDO RENDERIZAÇÃO PÓS-TREINO A2C ---")
        env_test = MazeEnv()
        try:
            model_to_test = load_a2c_model(a2c_model_path_saved, env_test)
        except Exception as e:
            print(f"ERRO: Modelo recém-treinado A2C não encontrado para renderização. Detalhes: {e}")
        else:
            state, _ = env_test.reset()
            done = False
            env_test.render()
            while not done:
                action, _ = model_to_test.predict(state, deterministic=True)
                action_value = action.item() if isinstance(action, np.ndarray) else action
                state, reward, terminated, truncated, _ = env_test.step(action_value)
                done = terminated or truncated
                print(f"Action: {action_value}, Reward: {reward}")
                env_test.render()


if TEST() and MODEL() in ["q-learning", "both"]:
    env = MazeEnv()

    q_load_id = Q_MODEL_ID
    if not is_id_provided_by_argv:

        q_models_ids = get_existing_q_models_ids()

        if not q_models_ids:
            print("ATENÇÃO: Nenhum modelo Q-Learning treinado encontrado para teste. Pulando teste.")
            sys.exit(1)

        print("\n" + "=" * 50)
        print("## Configuração de Teste Q-Learning")

        last_q_id = find_last_q_model_id()
        available_ids_str = ", ".join(map(str, q_models_ids))

        user_input = input(f"Digite o ID do modelo Q-Learning para testar ({available_ids_str}). Último treinado: {last_q_id}: ")

        try:
            q_load_id = int(user_input)
        except ValueError:
            q_load_id = last_q_id # Fallback

        print(f"Usando ID {q_load_id} para o teste.")
        print("=" * 50 + "\n")

    q_model_filename = DEFAULT_Q_FILENAME_BASE + str(q_load_id) + DEFAULT_Q_FILENAME_EXT
    q_learning_model_path = os.path.join(Q_LEARNING_MODEL_DIR, q_model_filename)

    agent = QLearningAgent(env.action_space, ALPHA(), GAMMAQ(), 0)

    try:
        agent.load_model(q_learning_model_path)
    except FileNotFoundError as e:
        print(e)
        if MODEL() == "q-learning":
            sys.exit(1)
        else:
            print("Pulando teste Q-Learning.")

    else:
        NUM_TEST_EPISODES = 1000
        test_rewards = []
        test_successes = 0

        print(f"Iniciando teste de {NUM_TEST_EPISODES} episódios para Q-Learning...")

        for test_episode in range(NUM_TEST_EPISODES):
            state, _ = env.reset(isnumpy=False)
            done = False
            episode_reward = 0

            while not done:
                action = agent.get_action(state)
                state, reward, terminated, truncated, _ = env.step(action, isnumpy=False)
                done = terminated or truncated
                episode_reward += reward

            test_rewards.append(episode_reward)
            if reward == GOAL_REWARD():
                test_successes += 1

            if (test_episode + 1) % 100 == 0:
                print(f"Testando episódio {test_episode + 1}/{NUM_TEST_EPISODES}")

        mean_reward = np.mean(test_rewards)
        success_rate_decimal = test_successes / NUM_TEST_EPISODES

        print("\n" + "=" * 40)
        print("--- Resultados do Teste Q-Learning (1000 Simulações) ---")
        print(f"Modelo Testado: {q_model_filename}")
        print(f"Recompensa Média: {mean_reward:.2f}")
        print(f"Taxa de Sucesso: {success_rate_decimal:.3f} (ou {success_rate_decimal * 100:.3f}%)")
        print("=" * 40 + "\n")


if TEST() and MODEL() in ["a2c", "both"]:
    NUM_TEST_EPISODES = 1000

    env_test = MazeEnv()

    test_a2c_model_id = A2C_MODEL_ID
    if not is_id_provided_by_argv:

        if not existing_models:
            print("ATENÇÃO: Nenhum modelo A2C treinado encontrado para teste. Pulando teste.")
            sys.exit(1)

        print("\n" + "=" * 50)
        print("## Configuração de Teste A2C")

        last_a2c_id = find_last_a2c_model_id()
        existing_a2c_models = sorted([int(f.replace("a2c_model_", "").replace(".zip", ""))
                               for f in os.listdir(A2C_MODEL_DIR_PATH)
                               if f.startswith("a2c_model_") and f.endswith(".zip")])
        available_ids_str = ", ".join(map(str, existing_a2c_models))

        user_input = input(f"Digite o ID do modelo A2C para testar ({available_ids_str}). Último treinado: {last_a2c_id}: ")

        try:
            test_a2c_model_id = int(user_input)
        except ValueError:
            test_a2c_model_id = last_a2c_id # Fallback

        print(f"Usando ID {test_a2c_model_id} para o teste.")
        print("=" * 50 + "\n")

    model_name = f"a2c_model_{test_a2c_model_id}.zip"
    model_path = os.path.join(A2C_MODEL_DIR_PATH, model_name)

    print("\n" + "=" * 60)
    print(f"Iniciando Teste do Único Modelo A2C (ID: {test_a2c_model_id}, 1000 Simulações)...")
    try:
        model_to_test = load_a2c_model(model_path, env_test)
    except Exception as e:
        print(
            f"ERRO: Modelo {model_name} não encontrado. Verifique se ele está em '{A2C_MODEL_DIR_PATH}'. Detalhes: {e}"
        )
        sys.exit(1)

    print("-" * 40)
    print(f"Testando Modelo ID: {test_a2c_model_id} ({model_name})")

    test_successes = 0
    test_rewards_list = []

    for _ in range(NUM_TEST_EPISODES):
        state, _ = env_test.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = model_to_test.predict(state, deterministic=True)

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

    mean_reward_final = np.mean(test_rewards_list)
    success_rate_decimal = test_successes / NUM_TEST_EPISODES

    print(f"Recompensa Média: {mean_reward_final:.2f}")
    print(
        f"Taxa de Sucesso: {success_rate_decimal:.3f} (ou {success_rate_decimal * 100:.3f}%)"
    )

    print("=" * 60 + "\n")

    print(f"## 📊 Resultados do Teste (Modelo {test_a2c_model_id} - 1000 Simulações)")
    print("| ID | Recompensa Média | Taxa de Sucesso |")
    print("|:---:|:---:|:---:|")
    print(
        f"| {test_a2c_model_id} | {mean_reward_final:.2f} | {success_rate_decimal * 100:.3f}% |"
    )

if RENDERS() and MODEL() in ["q-learning", "both"] and not TRAIN():
    env = MazeEnv()

    q_load_id = Q_MODEL_ID

    if not is_id_provided_by_argv:

        q_models_ids = get_existing_q_models_ids()

        if not q_models_ids:
            print("ATENÇÃO: Nenhum modelo Q-Learning treinado encontrado para renderizar. Pulando renderização.")
            sys.exit(1)

        print("\n" + "=" * 50)
        print("## Configuração de Renderização Q-Learning")

        last_q_id = find_last_q_model_id()
        available_ids_str = ", ".join(map(str, q_models_ids))

        user_input = input(f"Digite o ID do modelo Q-Learning para renderizar ({available_ids_str}). Último treinado: {last_q_id}: ")

        try:
            q_load_id = int(user_input)
        except ValueError:
            q_load_id = last_q_id

        print(f"Usando ID {q_load_id} para a renderização.")
        print("=" * 50 + "\n")

    q_model_filename = DEFAULT_Q_FILENAME_BASE + str(q_load_id) + DEFAULT_Q_FILENAME_EXT
    model_path_to_use = os.path.join(Q_LEARNING_MODEL_DIR, q_model_filename)

    agent_visual = QLearningAgent(env.action_space, ALPHA(), GAMMAQ(), 0)

    try:
        agent_visual.load_model(model_path_to_use)
    except FileNotFoundError as e:
        print(e)
        print("Não é possível renderizar sem modelo treinado.")
    else:
        print("\nRodando uma simulação visual do modelo Q-Learning treinado...")
        state, _ = env.reset(isnumpy=False)
        done = False
        env.render()

        while not done:
            action = agent_visual.get_action(state)
            state, reward, terminated, truncated, _ = env.step(action, isnumpy=False)
            done = terminated or truncated
            print(f"Action: {action}, Reward: {reward}")
            env.render()


if RENDERS() and MODEL() in ["a2c", "both"] and not TRAIN():
    env_test = MazeEnv()

    render_a2c_model_id = A2C_MODEL_ID

    if not is_id_provided_by_argv:

        existing_a2c_models = [int(f.replace("a2c_model_", "").replace(".zip", ""))
                               for f in os.listdir(A2C_MODEL_DIR_PATH)
                               if f.startswith("a2c_model_") and f.endswith(".zip")]
        last_a2c_id = find_last_a2c_model_id()

        if not existing_a2c_models:
             print("ATENÇÃO: Nenhum modelo A2C treinado encontrado para renderizar. Pulando renderização.")
             sys.exit(1)

        if RENDERS() and not TRAIN() and not TEST():
            print("\n" + "=" * 50)
            print("## Configuração de Renderização A2C")
            available_ids_str = ", ".join(map(str, sorted(existing_a2c_models)))
            user_input = input(f"Digite o ID do modelo A2C para renderizar ({available_ids_str}). Último treinado: {last_a2c_id}: ")
            try:
                render_a2c_model_id = int(user_input)
            except ValueError:
                render_a2c_model_id = last_a2c_id

            print(f"Usando ID {render_a2c_model_id} para a renderização.")
            print("=" * 50 + "\n")

    model_name = f"a2c_model_{render_a2c_model_id}.zip"
    model_path = os.path.join(A2C_MODEL_DIR_PATH, model_name)

    print(f"\nRodando simulação visual do modelo A2C (ID: {render_a2c_model_id})..., '{A2C_MODEL_DIR_PATH}'")

    try:
        model_to_test = load_a2c_model(model_path, env_test)
    except Exception as e:
        print(
            f"ERRO: Modelo {model_name} não encontrado. Verifique se ele está em '{A2C_MODEL_DIR_PATH}'. Detalhes: {e}"
        )
        sys.exit(1)

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
