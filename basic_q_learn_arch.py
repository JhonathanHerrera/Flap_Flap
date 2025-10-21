import numpy as np
import random


#Code Logic Format from https://medium.com/@goldengrisha/a-beginners-guide-to-q-learning-understanding-with-a-simple-gridworld-example-2b6736e7e2c9
# -------------------------
# GridWorld Environment
# -------------------------
class GridWorld:
    def __init__(self):
        self.grid = np.array([
            [0, 0, 0, 1],   # Goal at (0, 3)
            [0, -1, 0, 0],  # Wall with reward -1
            [0, 0, 0, 0],
            [0, 0, 0, 0]    # Start at (3, 0)
        ])
        self.start_state = (3, 0)
        self.state = self.start_state

    def reset(self):
        self.state = self.start_state
        return self.state

    def is_terminal(self, state):
        return self.grid[state] == 1 or self.grid[state] == -1

    def get_next_state(self, state, action):
        next_state = list(state)
        if action == 0:  # Up
            next_state[0] = max(0, state[0] - 1)
        elif action == 1:  # Right
            next_state[1] = min(3, state[1] + 1)
        elif action == 2:  # Down
            next_state[0] = min(3, state[0] + 1)
        elif action == 3:  # Left
            next_state[1] = max(0, state[1] - 1)
        return tuple(next_state)

    def step(self, action):
        next_state = self.get_next_state(self.state, action)
        reward = self.grid[next_state]
        self.state = next_state
        done = self.is_terminal(next_state)
        return next_state, reward, done


# -------------------------
# Q-Learning Agent
# -------------------------
class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.q_table = np.zeros((4, 4, 4))  # [row, col, action]
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, 3)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update_q_value(self, state, action, reward, next_state):
        max_future_q = np.max(self.q_table[next_state])
        current_q = self.q_table[state][action]
        # Q-learning formula
        self.q_table[state][action] = current_q + self.learning_rate * (
            reward + self.discount_factor * max_future_q - current_q
        )


# -------------------------
# Training Loop
# -------------------------
if __name__ == "__main__":
    env = GridWorld()
    agent = QLearningAgent()

    episodes = 1000

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_value(state, action, reward, next_state)
            state = next_state

    print("Training complete!")
    print("Learned Q-table:")
    print(agent.q_table)
