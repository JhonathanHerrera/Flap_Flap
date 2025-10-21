# file: train_flappy_q_improved.py
"""
Improved Q-learning trainer for Flappy Bird with enhanced features.
All hyperparameters are at the top for easy tuning.
"""

# ==================== HYPERPARAMETERS - EDIT THESE ====================

# --- Learning Parameters ---
LEARNING_RATE = 0.08
# Range: 0.01-0.3 | How fast the agent learns. Higher = faster but less stable.

LEARNING_RATE_DECAY = 0.99995
# Range: 0.9999-1.0 | Decay factor per update. 1.0 = no decay.

MIN_LEARNING_RATE = 0.02
# Range: 0.01-0.05 | Minimum learning rate after decay.

DISCOUNT_FACTOR = 0.995
# Range: 0.95-0.999 | How much to value future rewards. Higher = more forward-thinking.

# --- Exploration Parameters ---
EXPLORATION_START = 0.35
# Range: 0.1-0.5 | Initial random action probability. Higher = more exploration at start.

EXPLORATION_DECAY = 0.99975
# Range: 0.999-0.9999 | How fast exploration decreases. Higher = slower decay.

MIN_EXPLORATION = 0.01
# Range: 0.0-0.05 | Minimum random action probability. Always keep some exploration.

# --- Training Volume ---
TOTAL_EPISODES_PER_WORKER = 50000
# Range: 1000-10000 | Episodes each worker trains. More = better but slower.

SYNC_INTERVAL = 300
# Range: 100-500 | Episodes between syncing workers. Higher = more diverse exploration.

MAX_STEPS = 10000
# Range: 1000-10000 | Max steps per episode before timeout. Higher allows longer runs.

MAX_WORKERS = 15
# Range: 1-CPU_COUNT | Hard cap on parallel workers. Set based on your CPU cores.

# --- State Discretization (bins for tabular Q-learning) ---
Y_BINS = 10
# Range: 6-15 | Bins for bird's vertical position. More = finer control but larger table.

DY_GAP_BINS = 12
# Range: 8-20 | Bins for distance to gap center. More = better precision.

TTB_BINS = 10
# Range: 8-15 | Bins for time-to-pipe. More = better timing precision.

VEL_BINS = 9
# Range: 5-12 | Bins for bird velocity. More = smoother velocity control.

NEXT_GAP_BINS = 10
# Range: 6-15 | Bins for next pipe's gap position. More = better lookahead.

DY_NEXT_GAP_BINS = 10
# Range: 6-15 | Bins for distance to next gap. Helps with trajectory planning.

PIPE_GAP_SIZE_BINS = 6
# Range: 3-8 | Bins for gap size (if variable). More = better gap size awareness.

ACCEL_BINS = 5
# Range: 3-7 | Bins for bird acceleration. Captures momentum changes.

# --- Action Space ---
ACTIONS = 2
# Options: 2, 3, or 4 | Number of actions: 2=(no-flap,flap), 3=(no,small,big), 4=(no,small,med,big)

# --- Reward Shaping ---
DEATH_PENALTY = -5.0
# Range: -10.0 to -3.0 | Punishment for dying. More negative = agent avoids death more.

PIPE_PASS_REWARD = 5.0
# Range: 3.0-10.0 | Reward for passing a pipe. Higher = more aggressive pipe-seeking.

PERFECT_PASS_BONUS = 2.0
# Range: 1.0-5.0 | Extra reward for passing through center. Encourages precision.

PERFECT_PASS_THRESHOLD = 0.25
# Range: 0.15-0.35 | Fraction of gap height to be considered "perfect" (0.25 = within 25%).

CENTERING_REWARD_SCALE = 0.5
# Range: 0.2-1.0 | Multiplier for staying centered. Higher = stronger centering.

CENTERING_REWARD_SHARPNESS = 2.0
# Range: 1.0-4.0 | How sharply centering reward drops off. Higher = rewards only near center.

FLAP_PENALTY = 0.03
# Range: 0.01-0.1 | Small penalty per flap. Discourages excessive flapping.

FLAP_WHEN_CENTERED_PENALTY = 0.02
# Range: 0.01-0.05 | Extra penalty for flapping when already centered.

NEXT_PIPE_LOOKAHEAD_REWARD = 0.1
# Range: 0.0-0.3 | Reward for good positioning for next pipe. Encourages planning.

DISTANCE_LOSS_WEIGHT = 0.02
# Range: 0.01-0.05 | Weight for distance-to-gap-center penalty (legacy).

# --- Multi-Horizon Bootstrapping ---
HORIZONS = [5, 10, 20, 50]
# Range: Any list like [3,5,10] to [5,10,20,50,100] | Look ahead N steps. More horizons = better long-term planning.

# --- Experience Replay ---
USE_REPLAY_BUFFER = True
# Options: True/False | Enable experience replay for more stable learning.

REPLAY_BUFFER_SIZE = 50000
# Range: 10000-200000 | Max experiences to store. More = better but uses more memory.

REPLAY_BATCH_SIZE = 32
# Range: 16-128 | Experiences sampled per update. Higher = more stable but slower.

REPLAY_PRIORITY_ALPHA = 0.6
# Range: 0.0-1.0 | Priority exponent. 0=uniform, 1=fully prioritized by TD-error.

REPLAY_MIN_PRIORITY = 0.01
# Range: 0.001-0.1 | Minimum priority for all experiences. Ensures some sampling of all.

# --- Curriculum Learning ---
USE_CURRICULUM = True
# Options: True/False | Start easy, gradually increase difficulty.

CURRICULUM_PHASES = [
    {'episodes': 1000, 'gap_size': 200, 'pipe_speed': 5},
    {'episodes': 2000, 'gap_size': 180, 'pipe_speed': 6},
    {'episodes': 3500, 'gap_size': 160, 'pipe_speed': 7},
    {'episodes': 5000, 'gap_size': 140, 'pipe_speed': 8},
]
# Edit phases: 'episodes' = when to switch, 'gap_size' = pipe gap height, 'pipe_speed' = horizontal speed

# --- Optimistic Initialization ---
OPTIMISTIC_INIT_VALUE = 0.2
# Range: 0.0-1.0 | Initial Q-value for idle action. Higher = more initial exploration.

OPTIMISTIC_INIT_FLAP_VALUE = 0.1
# Range: 0.0-1.0 | Initial Q-value for flap action. Usually lower than idle.

# --- Multi-Action Jump Strengths (if ACTIONS > 2) ---
JUMP_STRENGTHS = {
    0: 0,      # no jump
    1: -7,     # small jump
    2: -10.5,  # normal jump
    3: -13,    # strong jump
}
# Range: 0 to -15 | Negative = upward velocity. Customize for ACTIONS=3 or 4.

# --- System Performance ---
DEFAULT_NICE_VALUE = 10
# Range: 0-19 (Linux) or 5-15 (Windows) | Lower CPU priority. Higher = more background-friendly.

USE_IONICE = False
# Options: True/False | Linux only - lower I/O priority to avoid disk bottlenecks.

DEFAULT_MAXTASKSPERCHILD = 200
# Range: 50-500 or None | Recycle workers after N tasks to prevent memory leaks. None = no recycling.

# --- Export Settings ---
EXPORT_DIR_NAME = "exports"
# Directory name for saving Q-tables and replays.

SAVE_BEST_Q = True
SAVE_AVG_Q = True
SAVE_REPLAY = True
# Options: True/False | Toggle what to save after training.

# ==================== END OF HYPERPARAMETERS ====================


import math
import os
import random
import time
import json
import importlib
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
from collections import deque

import numpy as np
import multiprocessing as mp

EXPORT_DIR = Path(EXPORT_DIR_NAME)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== Priority helpers ====================
def set_process_priority(nice_value: int = 10, ionice: bool = False, affinity: Optional[List[int]] = None):
    """Lower CPU/I/O priority and optionally set CPU affinity."""
    try:
        import psutil
    except Exception:
        psutil = None

    try:
        if os.name == "posix":
            if nice_value is not None:
                try:
                    os.nice(int(nice_value))
                except OSError:
                    pass
        elif os.name == "nt" and psutil:
            p = psutil.Process(os.getpid())
            if nice_value is None or nice_value < 5:
                p.nice(psutil.NORMAL_PRIORITY_CLASS)
            elif nice_value >= 15:
                p.nice(psutil.IDLE_PRIORITY_CLASS)
            else:
                p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass

    if ionice and os.name == "posix":
        try:
            import psutil
            p = psutil.Process(os.getpid())
            if hasattr(psutil, "IOPRIO_CLASS_IDLE"):
                p.ionice(psutil.IOPRIO_CLASS_IDLE)
        except Exception:
            pass

    if affinity:
        try:
            import psutil
            psutil.Process(os.getpid()).cpu_affinity(affinity)
        except Exception:
            pass

def _pool_initializer(nice_value: int, ionice_flag: bool, affinity: Optional[List[int]], headless: bool):
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        os.environ.setdefault("FB_NO_DISPLAY", "1")
    set_process_priority(nice_value, ionice_flag, affinity)

# ==================== Lazy import of environment ====================
def load_fb_module():
    fb = importlib.import_module("flappy_bird")
    return fb

def make_env_objects(fb):
    Bird, Pipe, Base = fb.Bird, fb.Pipe, fb.Base
    WIN_WIDTH, WIN_HEIGHT, FLOOR = fb.WIN_WIDTH, fb.WIN_HEIGHT, fb.FLOOR
    return Bird, Pipe, Base, WIN_WIDTH, WIN_HEIGHT, FLOOR

# ==================== Utilities ====================
def pipe_index_for_bird(bird_x, pipes):
    if not pipes:
        return 0
    for i, p in enumerate(pipes):
        if bird_x <= p.x + p.PIPE_TOP.get_width():
            return i
    return len(pipes) - 1

def gap_center(p): 
    return (p.height + p.bottom) / 2.0

def avg_forward_speed(): 
    return 8.0

# ==================== Experience Replay Buffer ====================
class ReplayBuffer:
    """Prioritized experience replay for more stable learning."""
    def __init__(self, capacity=REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
    
    def add(self, transition, td_error):
        """Add experience with priority based on TD error."""
        self.buffer.append(transition)
        priority = (abs(td_error) ** REPLAY_PRIORITY_ALPHA) + REPLAY_MIN_PRIORITY
        self.priorities.append(priority)
    
    def sample(self, batch_size=REPLAY_BATCH_SIZE):
        """Sample experiences based on priority."""
        if len(self.buffer) == 0:
            return []
        
        probs = np.array(self.priorities, dtype=np.float64)
        probs = probs / probs.sum()
        
        sample_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), size=sample_size, p=probs, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)

# ==================== Curriculum Manager ====================
class CurriculumManager:
    """Manages progressive difficulty during training."""
    def __init__(self):
        self.phases = CURRICULUM_PHASES
        self.current_phase_idx = 0
    
    def get_difficulty(self, episode):
        """Returns current difficulty settings based on episode count."""
        if not USE_CURRICULUM:
            return self.phases[-1] if self.phases else None
        
        for i, phase in enumerate(self.phases):
            if episode < phase['episodes']:
                self.current_phase_idx = i
                return phase
        return self.phases[-1]

# ==================== Discretizer ====================
class Discretizer:
    """Enhanced state discretization with more features."""
    def __init__(self, FLOOR: int):
        self.FLOOR = FLOOR
        
        # Bin edges
        self.y_edges = np.linspace(0, FLOOR, Y_BINS + 1)[1:-1]
        
        self.dy_max = 250.0
        self.dy_edges = np.linspace(0, self.dy_max, DY_GAP_BINS + 1)[1:-1]
        
        self.ttb_max = 80.0
        self.ttb_edges = np.linspace(0, self.ttb_max, TTB_BINS + 1)[1:-1]
        
        self.vel_min, self.vel_max = -12.0, 12.0
        self.vel_edges = np.linspace(self.vel_min, self.vel_max, VEL_BINS + 1)[1:-1]
        
        self.next_gap_edges = np.linspace(0, FLOOR, NEXT_GAP_BINS + 1)[1:-1]
        
        # Enhanced features
        self.dy_next_max = 300.0
        self.dy_next_edges = np.linspace(0, self.dy_next_max, DY_NEXT_GAP_BINS + 1)[1:-1]
        
        self.gap_size_min, self.gap_size_max = 100.0, 250.0
        self.gap_size_edges = np.linspace(self.gap_size_min, self.gap_size_max, PIPE_GAP_SIZE_BINS + 1)[1:-1]
        
        self.accel_min, self.accel_max = -5.0, 5.0
        self.accel_edges = np.linspace(self.accel_min, self.accel_max, ACCEL_BINS + 1)[1:-1]
        
        self.v_forward = avg_forward_speed()

    @staticmethod
    def _bin(val, edges):
        return int(np.clip(np.digitize(val, edges), 0, len(edges)))

    def discretize(self, bird, pipes, prev_vel=None):
        """Enhanced discretization with 8 features."""
        if not pipes:
            return (0, 0, 0, VEL_BINS // 2, 0, 0, 0, ACCEL_BINS // 2)
        
        idx = pipe_index_for_bird(bird.x, pipes)
        cur = pipes[idx]
        nxt = pipes[min(idx + 1, len(pipes) - 1)]
        
        # Basic features
        y = float(np.clip(bird.y, 0, self.FLOOR))
        gc = float(np.clip(gap_center(cur), 0, self.FLOOR))
        dy = min(self.dy_max, abs(y - gc))
        dist = max(0.0, cur.x - bird.x)
        ttb = min(self.ttb_max, dist / max(1e-3, self.v_forward))
        vel = float(np.clip(bird.vel, self.vel_min, self.vel_max))
        next_gc = float(np.clip(gap_center(nxt), 0, self.FLOOR))
        
        # Enhanced features
        dy_next = min(self.dy_next_max, abs(y - gap_center(nxt)))
        gap_size = float(np.clip(cur.bottom - cur.height, self.gap_size_min, self.gap_size_max))
        
        if prev_vel is not None:
            accel = float(np.clip(bird.vel - prev_vel, self.accel_min, self.accel_max))
        else:
            accel = 0.0
        
        return (
            self._bin(y, self.y_edges),
            self._bin(dy, self.dy_edges),
            self._bin(ttb, self.ttb_edges),
            self._bin(vel, self.vel_edges),
            self._bin(next_gc, self.next_gap_edges),
            self._bin(dy_next, self.dy_next_edges),
            self._bin(gap_size, self.gap_size_edges),
            self._bin(accel, self.accel_edges),
        )

# ==================== Enhanced Reward Function ====================
def enhanced_reward(alive, passed_pipe, bird, pipes, action):
    """Improved reward shaping with multiple components."""
    if not alive:
        return DEATH_PENALTY
    
    if not pipes:
        return 0.0
    
    idx = pipe_index_for_bird(bird.x, pipes)
    p = pipes[idx]
    gc = gap_center(p)
    gap_height = p.bottom - p.height
    
    # Distance to gap center
    dist_to_center = abs(bird.y - gc)
    normalized_dist = dist_to_center / (gap_height / 2.0)
    
    r = 0.0
    
    # Smooth centering reward (gaussian-like)
    r += CENTERING_REWARD_SCALE * math.exp(-normalized_dist * CENTERING_REWARD_SHARPNESS)
    
    # Pipe passing rewards
    if passed_pipe:
        r += PIPE_PASS_REWARD
        # Perfect pass bonus
        if dist_to_center < gap_height * PERFECT_PASS_THRESHOLD:
            r += PERFECT_PASS_BONUS
    
    # Flapping penalties
    if action == 1:
        r -= FLAP_PENALTY
        # Extra penalty for flapping when centered
        if dist_to_center < gap_height * 0.3:
            r -= FLAP_WHEN_CENTERED_PENALTY
    
    # Look-ahead positioning for next pipe
    if len(pipes) > idx + 1:
        next_p = pipes[idx + 1]
        next_gc = gap_center(next_p)
        next_dist = abs(bird.y - next_gc)
        next_gap_height = next_p.bottom - next_p.height
        normalized_next_dist = next_dist / (next_gap_height / 2.0)
        r += NEXT_PIPE_LOOKAHEAD_REWARD * (1.0 - normalized_next_dist)
    
    return r

# ==================== Q-Learning Agent ====================
def q_shape() -> Tuple[int, ...]:
    return (Y_BINS, DY_GAP_BINS, TTB_BINS, VEL_BINS, NEXT_GAP_BINS, 
            DY_NEXT_GAP_BINS, PIPE_GAP_SIZE_BINS, ACCEL_BINS, ACTIONS)

def optimistic_q_init(q: np.ndarray):
    """Initialize Q-table with optimistic values."""
    q.fill(0.0)
    q[..., 0] = OPTIMISTIC_INIT_VALUE  # idle action
    if ACTIONS > 1:
        q[..., 1:] = OPTIMISTIC_INIT_FLAP_VALUE  # flap actions

class QAgent:
    """Q-learning agent with adaptive learning rate and multi-horizon."""
    def __init__(self, d: Discretizer, init_q: Optional[np.ndarray] = None):
        self.d = d
        self.Q = np.zeros(q_shape(), dtype=np.float32)
        optimistic_q_init(self.Q)
        if init_q is not None:
            np.copyto(self.Q, init_q.astype(np.float32, copy=False))
        
        # Adaptive learning rate
        self.current_lr = LEARNING_RATE
        self.update_count = 0
        
        # Exploration
        self.eps = EXPLORATION_START
        
        # Experience replay
        self.replay_buffer = ReplayBuffer() if USE_REPLAY_BUFFER else None
        
        # Track previous velocity for acceleration feature
        self.prev_vel = None

    def act(self, s):
        """Epsilon-greedy action selection."""
        if random.random() < self.eps:
            return random.randint(0, ACTIONS - 1)
        return int(np.argmax(self.Q[s]))

    def decay_lr(self):
        """Decay learning rate over time."""
        self.current_lr = max(MIN_LEARNING_RATE, self.current_lr * LEARNING_RATE_DECAY)

    def learn_multihorizon(self, s, a, r, sp, terminal):
        """Multi-horizon Q-learning update."""
        if terminal:
            target = r
        else:
            # Multi-horizon bootstrapping
            targets = []
            for h in HORIZONS:
                gamma_h = DISCOUNT_FACTOR ** h
                targets.append(r + gamma_h * np.max(self.Q[sp]))
            target = np.mean(targets)
        
        current_q = self.Q[s + (a,)]
        td_error = target - current_q
        self.Q[s + (a,)] += self.current_lr * td_error
        
        # Update learning rate periodically
        self.update_count += 1
        if self.update_count % 1000 == 0:
            self.decay_lr()
        
        # Add to replay buffer
        if self.replay_buffer is not None and not terminal:
            self.replay_buffer.add((s, a, r, sp, terminal), td_error)
        
        return td_error

    def replay_updates(self, n_updates=5):
        """Sample from replay buffer and perform additional updates."""
        if self.replay_buffer is None or len(self.replay_buffer) < REPLAY_BATCH_SIZE:
            return
        
        for _ in range(n_updates):
            batch = self.replay_buffer.sample()
            for s, a, r, sp, terminal in batch:
                self.learn_multihorizon(s, a, r, sp, terminal)

    def decay_eps(self):
        """Decay exploration rate."""
        self.eps = max(MIN_EXPLORATION, self.eps * EXPLORATION_DECAY)

# ==================== Replay Recorder ====================
class ReplayRecorder:
    """Records episode for later visualization."""
    def __init__(self):
        self.frames = []
    
    def log(self, step, bird, pipes, score, WIN_WIDTH, FLOOR):
        frame = {
            "step": step,
            "bird_y": float(bird.y),
            "bird_vel": float(bird.vel),
            "score": int(score),
            "pipes": [{"x": float(p.x), "height": float(p.height), "bottom": float(p.bottom)} for p in pipes]
        }
        self.frames.append(frame)
    
    def save(self, path: Path, meta: dict):
        data = {"meta": meta, "frames": self.frames}
        np.savez_compressed(path, data=json.dumps(data))

# ==================== Training Loop ====================
def run_episodes(agent: QAgent, num_episodes: int, max_steps: int, Bird, Pipe, Base, WIN_WIDTH, FLOOR):
    """Run training episodes for a worker."""
    curriculum = CurriculumManager()
    
    best_score = -1
    total_score = 0
    
    for ep in range(num_episodes):
        difficulty = curriculum.get_difficulty(ep)
        
        bird = Bird(230, random.randint(250, 450))
        base = Base(FLOOR)
        pipes = [Pipe(700)]
        score = 0
        done = False
        steps = 0
        
        agent.prev_vel = bird.vel
        
        while not done and steps < max_steps:
            steps += 1
            s = agent.d.discretize(bird, pipes, agent.prev_vel)
            a = agent.act(s)
            
            # Multi-action support
            if ACTIONS > 2 and a > 0:
                bird.vel = JUMP_STRENGTHS.get(a, JUMP_STRENGTHS[1])
            elif a == 1:
                bird.jump()
            
            agent.prev_vel = bird.vel
            bird.move()
            base.move()
            
            rem, passed = [], False
            for p in list(pipes):
                p.move()
                if p.collide(bird, None):
                    done = True
                    break
                if p.x + p.PIPE_TOP.get_width() < 0:
                    rem.append(p)
                if not p.passed and p.x < bird.x:
                    p.passed = True
                    passed = True
            
            for r in rem:
                pipes.remove(r)
            
            if passed:
                score += 1
                pipes.append(Pipe(WIN_WIDTH))
            
            if bird.y + bird.img.get_height() >= FLOOR or bird.y < -50:
                done = True
            
            sp = agent.d.discretize(bird, pipes, bird.vel)
            r = enhanced_reward(not done, passed, bird, pipes, a)
            agent.learn_multihorizon(s, a, r, sp, done)
        
        # Replay updates at end of episode
        if USE_REPLAY_BUFFER:
            agent.replay_updates(n_updates=5)
        
        agent.decay_eps()
        best_score = max(best_score, score)
        total_score += score
    
    return {
        "best_score": best_score,
        "avg": total_score / max(1, num_episodes),
        "eps": agent.eps,
        "Q": agent.Q,
    }

# ==================== Worker Function ====================
def worker_round(worker_id: int, seed: int, episodes_this_round: int, max_steps: int, 
                 global_q_snapshot: np.ndarray):
    """Worker process for parallel training."""
    fb = load_fb_module()
    Bird, Pipe, Base, WIN_WIDTH, WIN_HEIGHT, FLOOR = make_env_objects(fb)
    
    random.seed(seed)
    np.random.seed(seed)
    d = Discretizer(FLOOR)
    agent = QAgent(d, init_q=global_q_snapshot)
    agent.eps = max(MIN_EXPLORATION, EXPLORATION_START * (0.9 + 0.2 * (worker_id % 3) / 2.0))
    
    stats = run_episodes(agent, episodes_this_round, max_steps, Bird, Pipe, Base, WIN_WIDTH, FLOOR)
    return {
        "worker": worker_id,
        "best_score": stats["best_score"],
        "avg": stats["avg"],
        "eps": stats["eps"],
        "Q": stats["Q"].astype(np.float32, copy=False),
    }

# ==================== Coordinator ====================
def average_q(list_of_q: List[np.ndarray]) -> np.ndarray:
    """FedAvg: average Q-tables from all workers."""
    out = np.zeros_like(list_of_q[0], dtype=np.float32)
    for q in list_of_q:
        out += q.astype(np.float32, copy=False)
    out /= float(len(list_of_q))
    return out

def _clamp_workers(n: int) -> int:
    return max(1, min(int(n), MAX_WORKERS))

def train_multiprocess(num_workers: Optional[int] = None,
                       total_episodes_per_worker: int = TOTAL_EPISODES_PER_WORKER,
                       sync_interval: int = SYNC_INTERVAL,
                       max_steps: int = MAX_STEPS,
                       nice_value: int = DEFAULT_NICE_VALUE,
                       ionice_flag: bool = USE_IONICE,
                       affinity: Optional[List[int]] = None,
                       headless: bool = True,
                       maxtasksperchild: Optional[int] = DEFAULT_MAXTASKSPERCHILD):
    """Main training coordinator."""
    if num_workers is None:
        num_workers = mp.cpu_count()
    num_workers = _clamp_workers(num_workers)
    
    rounds = int(math.ceil(total_episodes_per_worker / float(sync_interval)))
    print("=" * 60)
    print("IMPROVED Q-LEARNING FLAPPY BIRD")
    print("=" * 60)
    print(f"CPU Workers: {num_workers} (cap={MAX_WORKERS})")
    print(f"Episodes/Worker: {total_episodes_per_worker} | Sync: {sync_interval} | Rounds: {rounds}")
    print(f"State dims: {q_shape()}")
    print(f"Actions: {ACTIONS}")
    print(f"Replay Buffer: {USE_REPLAY_BUFFER} | Curriculum: {USE_CURRICULUM}")
    print("=" * 60)
    
    set_process_priority(nice_value, ionice_flag, affinity)
    
    gQ = np.zeros(q_shape(), dtype=np.float32)
    optimistic_q_init(gQ)
    global_best = {"score": -1, "Q": gQ.copy()}
    
    start = time.time()
    with mp.Pool(processes=num_workers,
                 initializer=_pool_initializer,
                 initargs=(nice_value, ionice_flag, affinity, headless),
                 maxtasksperchild=(maxtasksperchild if (maxtasksperchild is None or maxtasksperchild > 0) else None)) as pool:
        
        for rd in range(rounds):
            ep_this = sync_interval if (rd < rounds - 1) else (total_episodes_per_worker - sync_interval * (rounds - 1))
            seeds = [(rd + 1) * 10000 + 123 + w for w in range(num_workers)]
            tasks = [(w, seeds[w], ep_this, max_steps, gQ) for w in range(num_workers)]
            
            results = pool.starmap(worker_round, tasks)
            
            local_Qs = [res["Q"] for res in results]
            gQ = average_q(local_Qs)
            
            best_in_round = max(results, key=lambda r: r["best_score"])
            avg_avgs = float(np.mean([r["avg"] for r in results]))
            print(f"[Round {rd+1}/{rounds}] "
                  f"Worker#{best_in_round['worker']} score={best_in_round['best_score']} | "
                  f"Avg={avg_avgs:.2f} | eps={np.mean([r['eps'] for r in results]):.3f}")
            
            if best_in_round["best_score"] > global_best["score"]:
                global_best["score"] = best_in_round["best_score"]
                global_best["Q"] = best_in_round["Q"].copy()
    
    dur = time.time() - start
    total_episodes = num_workers * total_episodes_per_worker
    print("\n" + "=" * 60)
    print(f"✅ TRAINING COMPLETE in {dur:.1f}s | {total_episodes/dur:.1f} eps/sec")
    print(f"Best Score: {global_best['score']}")
    print("=" * 60)
    
    if SAVE_BEST_Q:
        np.save(EXPORT_DIR / "best_q_table_improved.npy", global_best["Q"])
        print(f"💾 Saved best Q -> {EXPORT_DIR/'best_q_table_improved.npy'}")
    
    if SAVE_AVG_Q:
        np.save(EXPORT_DIR / "avg_q_table_improved.npy", gQ)
        print(f"💾 Saved averaged Q -> {EXPORT_DIR/'avg_q_table_improved.npy'}")
    
    if SAVE_REPLAY:
        score = record_greedy_replay(global_best["Q"], seed=123, 
                                     save_path=EXPORT_DIR / "replay_improved.npz")
        print(f"🎥 Replay saved (score {score}) -> {EXPORT_DIR/'replay_improved.npz'}")
    
    return global_best, gQ

# ==================== Replay Runner ====================
def record_greedy_replay(Q, seed: int, save_path: Path, max_steps: int = 6000):
    """Record greedy policy for visualization."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("FB_NO_DISPLAY", "1")
    fb = load_fb_module()
    Bird, Pipe, Base, WIN_WIDTH, WIN_HEIGHT, FLOOR = make_env_objects(fb)
    
    random.seed(seed)
    np.random.seed(seed)
    d = Discretizer(FLOOR)
    agent = QAgent(d, init_q=Q)
    agent.eps = 0.0
    
    bird = Bird(230, random.randint(250, 450))
    base = Base(FLOOR)
    pipes = [Pipe(700)]
    score, done, steps = 0, False, 0
    
    rec = ReplayRecorder()
    prev_vel = bird.vel
    
    while not done and steps < max_steps:
        steps += 1
        s = d.discretize(bird, pipes, prev_vel)
        a = agent.act(s)
        
        if ACTIONS > 2 and a > 0:
            bird.vel = JUMP_STRENGTHS.get(a, JUMP_STRENGTHS[1])
        elif a == 1:
            bird.jump()
        
        prev_vel = bird.vel
        bird.move()
        base.move()
        
        rem, passed = [], False
        for p in list(pipes):
            p.move()
            if p.collide(bird, None):
                done = True
                break
            if p.x + p.PIPE_TOP.get_width() < 0:
                rem.append(p)
            if not p.passed and p.x < bird.x:
                p.passed = True
                passed = True
        
        for r in rem:
            pipes.remove(r)
        
        if passed:
            score += 1
            pipes.append(Pipe(WIN_WIDTH))
        
        if bird.y + bird.img.get_height() >= FLOOR or bird.y < -50:
            done = True
        
        rec.log(steps, bird, pipes, score, WIN_WIDTH, FLOOR)
    
    meta = {"seed": seed, "episode_score": int(score), "note": "improved Q-learning"}
    rec.save(save_path, meta)
    return score

# ==================== CLI ====================
def parse_affinity(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out

def main():
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    
    ap = argparse.ArgumentParser(description="Improved Q-learning for Flappy Bird")
    ap.add_argument("--workers", type=int, default=None, help=f"Number of workers (max {MAX_WORKERS})")
    ap.add_argument("--episodes-per-worker", type=int, default=TOTAL_EPISODES_PER_WORKER)
    ap.add_argument("--sync-interval", type=int, default=SYNC_INTERVAL)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--nice", type=int, default=DEFAULT_NICE_VALUE)
    ap.add_argument("--ionice", action="store_true", help="Enable I/O priority lowering (Linux)")
    ap.add_argument("--affinity", type=str, default=None, help="CPU cores e.g. '2-15'")
    ap.add_argument("--maxtasksperchild", type=int, default=DEFAULT_MAXTASKSPERCHILD)
    args = ap.parse_args()
    
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"System RAM: {mem.total/1e9:.1f} GB")
    except:
        pass
    
    requested = args.workers if args.workers is not None else mp.cpu_count()
    num_workers = _clamp_workers(requested)
    affinity = parse_affinity(args.affinity)
    
    train_multiprocess(num_workers=num_workers,
                       total_episodes_per_worker=args.episodes_per_worker,
                       sync_interval=args.sync_interval,
                       max_steps=args.max_steps,
                       nice_value=args.nice,
                       ionice_flag=args.ionice,
                       affinity=affinity,
                       headless=True,
                       maxtasksperchild=args.maxtasksperchild)

if __name__ == "__main__":
    mp.freeze_support()
    main()