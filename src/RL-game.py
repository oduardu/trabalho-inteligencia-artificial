import os
import pickle
import random
from collections import defaultdict

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
import sys

from config import (
    ALPHA,
    ALPHA_DECAY,
    DECAY_STEP,
    EPSILON,
    EPSILON_DECAY,
    GAMMA,
    GOAL_REWARD,
    HOLE_REWARD,
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
            return np.array(observation)
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
        done = False
        reward = STEP_REWARD()  # small negative reward for each step

        if new_pos_value == self.WALL:
            self.agent_pos = prev_pos  # revert move
            reward = WALL_REWARD()
        elif new_pos_value == self.HOLE:
            done = True
            reward = HOLE_REWARD()
        elif self.agent_pos == self.goal_pos:
            done = True
            reward = GOAL_REWARD()

        # Update grid
        if new_pos_value != self.WALL:
            self.grid[prev_pos[0], prev_pos[1]] = self.EMPTY
            self.grid[self.agent_pos[0], self.agent_pos[1]] = self.AGENT

        if self.turn == 200:
            return self.get_state(isnumpy), reward, True, False, {}

        return self.get_state(isnumpy), reward, done, False, {}

    def reset(self, *, seed=None, options=None, return_info=False, isnumpy=True):
        super().reset(seed=seed)
        # Reset game to initial state
        self.turn = 0
        self.grid_initialization()
        return self.get_state(isnumpy), {}

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


# Training Model/Agent
if TRAIN():
    env = MazeEnv()
    agent = QLearningAgent(env.action_space, ALPHA(), GAMMA(), EPSILON())

    episodes = SIMMULATION_NUMBER()
    total_reward = 0
    sucess = 0
    for episode in range(1, episodes + 1):
        state, _ = env.reset(isnumpy=False)
        done = False
        if RENDERS():
            env.render()

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action, isnumpy=False)

            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            if reward == GOAL_REWARD():
                sucess += 1
            if RENDERS():
                env.render()

        # Parameter Decay
        if episode % DECAY_STEP() == 0:
            agent.epsilon *= EPSILON_DECAY()
            agent.alpha *= ALPHA_DECAY()
            print(
                f"Episode {episode}, Mean Reward: {(total_reward / DECAY_STEP()):.2f}, Success Rate: {(sucess / DECAY_STEP()):.2f}"
            )
            print("Explore Chance (epsilon): ", agent.epsilon)
            print("Exploit Chance (1-epsilon): ", 1 - agent.epsilon)
            print("Learning Rate (alpha): ", agent.alpha)
            total_reward = 0
            sucess = 0

    # Save the trained agent
    model_path = os.path.join(os.path.dirname(__file__), "q_learning_model.pkl")
    agent.save_model(model_path)
    print(f"Modelo salvo em: {model_path}")

# Test the trained agent
else:
    env = MazeEnv()

    # Load the trained agent
    # Get model path from command line argument or use default
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = os.path.join(os.path.dirname(__file__), "q_learning_model.pkl")
    agent = QLearningAgent(env.action_space, ALPHA(), GAMMA(), 0)

    try:
        agent.load_model(model_path)
    except FileNotFoundError as e:
        print(e)
        # Se o modelo não for encontrado, encerra a execução
        exit()

    # --- Lógica para Testar com 1000 Simulações (Requisito 2) ---
    NUM_TEST_EPISODES = 1000  # Número de simulações para o teste (Fixo em 1000)
    test_rewards = []
    test_successes = 0

    should_render = RENDERS()  # Salva a configuração original de RENDERS

    if should_render:
        print(
            "Atenção: Rodando 1000 testes sem renderização para velocidade. Uma simulação será renderizada ao final."
        )

    print(f"Iniciando teste de {NUM_TEST_EPISODES} episódios. Pode levar um momento...")

    for test_episode in range(NUM_TEST_EPISODES):
        state, _ = env.reset(isnumpy=False)
        done = False
        episode_reward = 0

        while not done:
            action = agent.get_action(state)
            state, reward, done, _, _ = env.step(action, isnumpy=False)
            episode_reward += reward

        test_rewards.append(episode_reward)
        # Verifica se o agente atingiu o objetivo (GOAL_REWARD)
        if reward == GOAL_REWARD():
            test_successes += 1

        # Opcional: Imprimir progresso
        if (test_episode + 1) % 100 == 0:
            print(f"Testando episódio {test_episode + 1}/{NUM_TEST_EPISODES}")

    # Calcular as métricas
    mean_reward = np.mean(test_rewards)
    success_rate_decimal = test_successes / NUM_TEST_EPISODES

    # Imprimir o resultado no formato solicitado
    print("\n" + "=" * 40)
    print("--- Resultados do Teste (1000 Simulações) ---")
    print(f"Recompensa Média: {mean_reward:.2f}")
    print(
        f"Taxa de Sucesso: {success_rate_decimal:.2f} (ou {success_rate_decimal * 100:.2f}%)"
    )
    print("=" * 40 + "\n")

    # --- SIMULAÇÃO VISUAL (Se RENDERS = True) ---
    if should_render:
        print("\nRodando uma simulação visual do modelo treinado...")
        state, _ = env.reset(isnumpy=False)
        done = False
        env.render()

        while not done:
            action = agent.get_action(state)
            state, reward, done, _, _ = env.step(action, isnumpy=False)
            print(f"Action: {action}, Reward: {reward}")
            env.render()
