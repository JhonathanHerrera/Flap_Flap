# file: v6_optimized.py
"""
OPTIMIZED DQN for Flappy Bird

Changes from v6.py:
1. Cleaner epsilon decay schedule
2. Simplified reward function
3. Better hyperparameters
4. More training episodes
"""

import random
from collections import deque
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from flappy_bird_easy import Bird, Pipe, Base, WIN_WIDTH, WIN_HEIGHT, FLOOR

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Optimized DQN Agent for Flappy Bird")
print(f"Using device: {device}")

# ==================== OPTIMIZED HYPERPARAMETERS ====================

# Training
FPS_TRAIN = 0  # No frame limiting - train as fast as possible!
MAX_STEPS = 2000  # Slightly reduced
EPISODES = 12000  # Reduced but still plenty

# Replay Buffer
BUFFER_SIZE = 100_000
BATCH_SIZE = 64

# Learning
GAMMA = 0.99
LR = 1e-4
TAU = 0.005
N_STEPS = 3

# Regularization - Prevent overfitting
L1_LAMBDA = 1e-5  # L1 (Lasso) regularization strength
L2_LAMBDA = 1e-4  # L2 (Ridge) regularization strength (weight decay)

# Early Stopping - Stop when model stops improving
EARLY_STOP_PATIENCE = 2000  # Stop if no improvement for 2000 episodes
EARLY_STOP_MIN_DELTA = 1.0   # Minimum improvement in avg score to count
EVAL_WINDOW = 100            # Evaluate average over last 100 episodes

# Epsilon - LINEAR DECAY (cleaner than exponential)
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 8000  # Reach minimum after 8k episodes (was 10k)

# Rewards - SIMPLIFIED
REWARD_DEATH = -1.0
REWARD_PASS_PIPE = 1.0
REWARD_ALIVE = 0.01
REWARD_CENTER_WEIGHT = 0.1  # Small bonus for staying centered

# =================================================================

class DuelingDQN(nn.Module):
    """Same excellent architecture from v6.py"""
    def __init__(self, input_dim=6, hidden=128, output_dim=2):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.val = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.adv = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        f = self.feature(x)
        v = self.val(f)
        a = self.adv(f)
        return v + (a - a.mean(dim=1, keepdim=True))


class DQNAgent:
    def __init__(self):
        self.q_net = DuelingDQN().to(device)
        self.target_net = DuelingDQN().to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.q_net.parameters(), 
            lr=LR,
            weight_decay=L2_LAMBDA  # L2 regularization via weight decay
        )
        self.memory = deque(maxlen=BUFFER_SIZE)
        self.nstep_buf = deque(maxlen=N_STEPS)

        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        self.tau = TAU
        self.train_steps = 0
        
        self.episode = 0

    def get_epsilon(self):
        """LINEAR epsilon decay - cleaner than exponential"""
        if self.episode >= EPS_DECAY_STEPS:
            return EPS_END
        
        # Linear interpolation
        progress = self.episode / EPS_DECAY_STEPS
        return EPS_START - (EPS_START - EPS_END) * progress

    def get_state(self, bird, pipes):
        """Same excellent 6-feature state from v6.py"""
        pipe_idx = 0
        if len(pipes) > 1 and bird.x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
            pipe_idx = 1
        
        top = pipes[pipe_idx].height
        bottom = pipes[pipe_idx].bottom
        gap_center = 0.5 * (top + bottom)
        dx = pipes[pipe_idx].x - bird.x

        return np.array([
            bird.y / WIN_HEIGHT,
            abs(bird.y - top) / WIN_HEIGHT,
            abs(bird.y - bottom) / WIN_HEIGHT,
            (bird.y - gap_center) / WIN_HEIGHT,
            bird.vel / 16.0,
            dx / WIN_WIDTH,
        ], dtype=np.float32)

    def choose_action(self, state):
        epsilon = self.get_epsilon()
        
        if random.random() < epsilon:
            return random.randint(0, 1)
        
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q = self.q_net(s)
        return int(q.argmax().item())

    def _append_nstep(self, s, a, r, sn, done):
        """Same N-step logic from v6.py"""
        self.nstep_buf.append((s, a, r, sn, done))
        if len(self.nstep_buf) < N_STEPS and not done:
            return None
        
        R, s0, a0 = 0.0, self.nstep_buf[0][0], self.nstep_buf[0][1]
        sN, dN = (sn, done)
        
        for i, (_, _, ri, s_i1, d_i1) in enumerate(self.nstep_buf):
            R += (self.gamma ** i) * ri
            if d_i1:
                sN, dN = s_i1, d_i1
                break
        
        if sN is None:
            sN = np.zeros_like(s0)
        return (s0, a0, R, sN, dN)

    def remember(self, s, a, r, sn, done):
        tr = self._append_nstep(s, a, r, sn, done)
        if tr is not None:
            self.memory.append(tr)
        if done:
            self.nstep_buf.clear()
        else:
            while len(self.nstep_buf) > N_STEPS - 1:
                self.nstep_buf.popleft()

    def soft_update(self):
        with torch.no_grad():
            for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        s = torch.tensor(states, dtype=torch.float32, device=device)
        a = torch.tensor(actions, dtype=torch.long, device=device)
        r = torch.tensor(rewards, dtype=torch.float32, device=device).clamp_(-5.0, 5.0)
        sn = torch.tensor(next_states, dtype=torch.float32, device=device)
        d = torch.tensor(dones, dtype=torch.bool, device=device)

        with torch.no_grad():
            # Double DQN
            next_actions = self.q_net(sn).argmax(1)
            next_q = self.target_net(sn).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = r + (self.gamma ** N_STEPS) * next_q * (~d)

        q_sa = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q_sa, target_q)
        
        # Add L1 regularization (Lasso)
        if L1_LAMBDA > 0:
            l1_norm = sum(p.abs().sum() for p in self.q_net.parameters())
            loss = loss + L1_LAMBDA * l1_norm

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self.soft_update()
        self.train_steps += 1

        if self.train_steps % 500 == 0:
            with torch.no_grad():
                mean_q = self.q_net(s).mean().item()
            eps = self.get_epsilon()
            print(f"[Step {self.train_steps:5d}] Loss={loss.item():.4f} | MeanQ={mean_q:.2f} | ε={eps:.3f} | Mem={len(self.memory)}")


def compute_reward(alive, passed_pipe, bird, pipes, action):
    """SIMPLIFIED reward function"""
    if not alive:
        return REWARD_DEATH
    
    reward = REWARD_ALIVE
    
    if passed_pipe:
        reward += REWARD_PASS_PIPE
    
    # Small bonus for staying near gap center
    if pipes:
        pipe_idx = 0
        if len(pipes) > 1 and bird.x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
            pipe_idx = 1
        
        top = pipes[pipe_idx].height
        bottom = pipes[pipe_idx].bottom
        gap_center = 0.5 * (top + bottom)
        gap_height = bottom - top
        
        distance_from_center = abs(bird.y - gap_center)
        normalized_dist = distance_from_center / (gap_height / 2.0)
        
        # Reward being centered (max +0.1)
        centering_bonus = REWARD_CENTER_WEIGHT * (1.0 - min(1.0, normalized_dist))
        reward += centering_bonus
    
    return float(np.clip(reward, -5.0, 5.0))


def train_dqn(episodes=EPISODES):
    print("=" * 60)
    print(f"Training for up to {episodes} episodes")
    print(f"Epsilon: {EPS_START:.2f} → {EPS_END:.2f} over {EPS_DECAY_STEPS} episodes")
    print(f"Regularization: L1={L1_LAMBDA:.0e}, L2={L2_LAMBDA:.0e}")
    print(f"Early stopping: patience={EARLY_STOP_PATIENCE}, min_delta={EARLY_STOP_MIN_DELTA}")
    print("=" * 60)
    
    agent = DQNAgent()
    scores = []
    best_score = 0
    
    # Early stopping tracking
    best_avg_score = 0
    episodes_without_improvement = 0
    stopped_early = False

    for ep in range(episodes):
        agent.episode = ep
        
        bird = Bird(230, random.randint(250, 450))
        base = Base(FLOOR)
        pipes = [Pipe(700)]
        score, done, steps = 0, False, 0
        state = agent.get_state(bird, pipes)
        clock = pygame.time.Clock()

        while not done and steps < MAX_STEPS:
            steps += 1
            # No FPS limiting - train as fast as possible!

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action = agent.choose_action(state)
            if action == 1:
                bird.jump()

            bird.move()
            base.move()

            rem, add_pipe = [], False
            for pipe in pipes:
                pipe.move()
                if pipe.collide(bird, None):
                    done = True
                    break
                if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                    rem.append(pipe)
                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if add_pipe:
                score += 1
                pipes.append(Pipe(WIN_WIDTH))
            
            for r in rem:
                pipes.remove(r)

            if bird.y + bird.img.get_height() >= FLOOR or bird.y < -50:
                done = True

            reward = compute_reward(not done, add_pipe, bird, pipes, action)
            next_state = agent.get_state(bird, pipes) if not done else None

            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state

        scores.append(score)
        avg_window = np.mean(scores[-EVAL_WINDOW:]) if len(scores) >= EVAL_WINDOW else np.mean(scores)
        
        # Update best score
        if score > best_score:
            best_score = score
            torch.save(agent.q_net.state_dict(), "dqn_flappy_optimized_best.pth")

        # Early stopping check
        if len(scores) >= EVAL_WINDOW:
            if avg_window > best_avg_score + EARLY_STOP_MIN_DELTA:
                # Significant improvement!
                best_avg_score = avg_window
                episodes_without_improvement = 0
            else:
                # No significant improvement
                episodes_without_improvement += 1
            
            # Check if we should stop
            if episodes_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"\n{'='*60}")
                print(f"⏹️  EARLY STOPPING at episode {ep}")
                print(f"No improvement for {EARLY_STOP_PATIENCE} episodes")
                print(f"Best avg score: {best_avg_score:.2f}")
                print(f"{'='*60}\n")
                stopped_early = True
                break

        if ep % 100 == 0:
            eps = agent.get_epsilon()
            patience_info = f" | NoImprove={episodes_without_improvement}/{EARLY_STOP_PATIENCE}" if len(scores) >= EVAL_WINDOW else ""
            print(f"Ep {ep:5d} | Score: {score:3d} | Avg{EVAL_WINDOW}: {avg_window:.2f} | Best: {best_score:3d} | ε={eps:.3f}{patience_info}")

        if ep % 1000 == 0 and ep > 0:
            torch.save(agent.q_net.state_dict(), f"dqn_flappy_optimized_ep{ep}.pth")
            print(f"  💾 Checkpoint saved")

    # Save final model
    torch.save(agent.q_net.state_dict(), "dqn_flappy_optimized_final.pth")
    
    print("\n" + "=" * 60)
    if stopped_early:
        print(f"✅ Training stopped early at episode {ep} (saved time!)")
    else:
        print(f"✅ Training complete - all {episodes} episodes")
    print(f"🏆 Best Score: {best_score}")
    print(f"📊 Best Avg{EVAL_WINDOW}: {best_avg_score:.2f}")
    print(f"📈 Final Avg{EVAL_WINDOW}: {avg_window:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    pygame.init()
    train_dqn(episodes=EPISODES)