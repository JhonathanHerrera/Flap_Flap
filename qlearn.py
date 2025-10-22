# file: train_flappy_q_fixed.py
"""
FIXED Q-learning trainer - addresses the "not learning" problem.
Key fixes:
1. Smaller, more manageable state space
2. Proper exploration decay
3. Better reward shaping
4. Simpler curriculum
"""

# ==================== HYPERPARAMETERS - EDIT THESE ====================

# --- Learning Parameters ---
LEARNING_RATE = 0.15
# Range: 0.1-0.3 | Higher = faster learning (increased from 0.08)

DISCOUNT_FACTOR = 0.99
# Range: 0.95-0.995 | Lowered for shorter-term thinking (better for early learning)

# --- Exploration Parameters ---
EXPLORATION_START = 0.5
# Range: 0.3-0.7 | Start with MORE exploration

EXPLORATION_DECAY = 0.99
# Range: 0.9995-0.9999 | Decay per EPISODE (not per step!)

MIN_EXPLORATION = 0.05
# Range: 0.02-0.1 | Keep more exploration throughout

# --- Training Volume ---
TOTAL_EPISODES_PER_WORKER = 3000
# Reduced for faster iteration - increase once it's working

SYNC_INTERVAL = 200
# Sync more frequently for stability

MAX_STEPS = 3000
# Shorter episodes for faster learning cycles

MAX_WORKERS = 15

# --- SIMPLIFIED State Discretization ---
# KEY FIX: Much smaller state space for faster learning
Y_BINS = 6  # Reduced from 10
# Range: 4-8 | Coarser vertical position

DY_GAP_BINS = 8  # Reduced from 12
# Range: 6-10 | Distance to gap

TTB_BINS = 6  # Reduced from 10
# Range: 4-8 | Time to pipe

VEL_BINS = 5  # Reduced from 9
# Range: 4-7 | Velocity bins

NEXT_GAP_BINS = 6  # Reduced from 10
# Range: 4-8 | Next gap position

# REMOVED complex features for now
# DY_NEXT_GAP_BINS = 0  # Not used
# PIPE_GAP_SIZE_BINS = 0  # Not used
# ACCEL_BINS = 0  # Not used

# New Q-table size: 6×8×6×5×6×2 = 17,280 states (vs 64.8M!)
# This is 3700x smaller and will learn much faster!

# --- Action Space ---
ACTIONS = 2  # Keep it simple

# --- IMPROVED Reward Shaping ---
DEATH_PENALTY = -10.0
# Increased penalty - make death REALLY bad

PIPE_PASS_REWARD = 10.0
# Increased reward - make success REALLY good

ALIVE_REWARD = 0.1
# Small reward just for surviving each step

DISTANCE_REWARD_SCALE = 0.5
# Reward for being near gap center

FLAP_PENALTY = 0.0
# REMOVED flap penalty - let it flap freely while learning

# --- Multi-Horizon (simplified) ---
HORIZONS = [1, 5, 10]
# Simplified - focus on immediate and short-term

# --- Experience Replay ---
USE_REPLAY_BUFFER = False
# DISABLED for now - adds complexity

# --- Curriculum Learning ---
USE_CURRICULUM = False
# DISABLED - standard game is fine

# --- Optimistic Initialization ---
OPTIMISTIC_INIT_VALUE = 1.0
# Higher optimism = more exploration

# --- System Performance ---
DEFAULT_NICE_VALUE = 10
USE_IONICE = False
DEFAULT_MAXTASKSPERCHILD = 200

# --- Export Settings ---
EXPORT_DIR_NAME = "exports"
SAVE_BEST_Q = True
SAVE_AVG_Q = True
SAVE_REPLAY = True

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

import numpy as np
import multiprocessing as mp

EXPORT_DIR = Path(EXPORT_DIR_NAME)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== Priority helpers ====================
def set_process_priority(nice_value: int = 10, ionice: bool = False, affinity: Optional[List[int]] = None):
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

# ==================== Lazy import ====================
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

# ==================== SIMPLIFIED Discretizer ====================
class Discretizer:
    """Simplified state with only 5 features."""
    def __init__(self, FLOOR: int):
        self.FLOOR = FLOOR
        self.y_edges = np.linspace(0, FLOOR, Y_BINS + 1)[1:-1]
        
        self.dy_max = 300.0
        self.dy_edges = np.linspace(0, self.dy_max, DY_GAP_BINS + 1)[1:-1]
        
        self.ttb_max = 100.0
        self.ttb_edges = np.linspace(0, self.ttb_max, TTB_BINS + 1)[1:-1]
        
        self.vel_min, self.vel_max = -12.0, 12.0
        self.vel_edges = np.linspace(self.vel_min, self.vel_max, VEL_BINS + 1)[1:-1]
        
        self.next_gap_edges = np.linspace(0, FLOOR, NEXT_GAP_BINS + 1)[1:-1]

    @staticmethod
    def _bin(val, edges):
        return int(np.clip(np.digitize(val, edges), 0, len(edges)))

    def discretize(self, bird, pipes):
        """Simple 5-feature state."""
        if not pipes:
            return (0, 0, 0, VEL_BINS // 2, 0)
        
        idx = pipe_index_for_bird(bird.x, pipes)
        cur = pipes[idx]
        nxt = pipes[min(idx + 1, len(pipes) - 1)]
        
        y = float(np.clip(bird.y, 0, self.FLOOR))
        gc = float(np.clip(gap_center(cur), 0, self.FLOOR))
        dy = min(self.dy_max, abs(y - gc))
        dist = max(0.0, cur.x - bird.x)
        ttb = min(self.ttb_max, dist / 8.0)  # assume speed ~8
        vel = float(np.clip(bird.vel, self.vel_min, self.vel_max))
        next_gc = float(np.clip(gap_center(nxt), 0, self.FLOOR))
        
        return (
            self._bin(y, self.y_edges),
            self._bin(dy, self.dy_edges),
            self._bin(ttb, self.ttb_edges),
            self._bin(vel, self.vel_edges),
            self._bin(next_gc, self.next_gap_edges),
        )

# ==================== IMPROVED Reward Function ====================
def improved_reward(alive, passed_pipe, bird, pipes, action):
    """Clear, simple rewards."""
    if not alive:
        return DEATH_PENALTY
    
    r = ALIVE_REWARD  # Small reward for survival
    
    if passed_pipe:
        r += PIPE_PASS_REWARD  # Big reward for passing
    
    # Distance-based shaping
    if pipes:
        idx = pipe_index_for_bird(bird.x, pipes)
        p = pipes[idx]
        gc = gap_center(p)
        gap_height = p.bottom - p.height
        
        dist = abs(bird.y - gc)
        normalized_dist = dist / (gap_height / 2.0)
        
        # Reward being centered (0 when far, 1 when centered)
        centering_reward = DISTANCE_REWARD_SCALE * (1.0 - normalized_dist)
        r += max(0, centering_reward)
    
    return r

# ==================== Q-Learning Agent ====================
def q_shape() -> Tuple[int, ...]:
    return (Y_BINS, DY_GAP_BINS, TTB_BINS, VEL_BINS, NEXT_GAP_BINS, ACTIONS)

def optimistic_q_init(q: np.ndarray):
    q.fill(OPTIMISTIC_INIT_VALUE)

class QAgent:
    def __init__(self, d: Discretizer, init_q: Optional[np.ndarray] = None):
        self.d = d
        self.Q = np.zeros(q_shape(), dtype=np.float32)
        optimistic_q_init(self.Q)
        if init_q is not None:
            np.copyto(self.Q, init_q.astype(np.float32, copy=False))
        
        self.eps = EXPLORATION_START
        self.episode_count = 0

    def act(self, s):
        if random.random() < self.eps:
            return random.randint(0, ACTIONS - 1)
        return int(np.argmax(self.Q[s]))

    def learn_multihorizon(self, s, a, r, sp, terminal):
        """Multi-horizon update."""
        if terminal:
            target = r
        else:
            # Average over multiple horizons
            targets = []
            for h in HORIZONS:
                gamma_h = DISCOUNT_FACTOR ** h
                targets.append(r + gamma_h * np.max(self.Q[sp]))
            target = np.mean(targets)
        
        current_q = self.Q[s + (a,)]
        td_error = target - current_q
        self.Q[s + (a,)] += LEARNING_RATE * td_error
        
        return td_error

    def decay_eps(self):
        """Decay epsilon per EPISODE."""
        self.episode_count += 1
        self.eps = max(MIN_EXPLORATION, self.eps * EXPLORATION_DECAY)

# ==================== Replay Recorder ====================
class ReplayRecorder:
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
    best_score = -1
    total_score = 0
    
    for ep in range(num_episodes):
        bird = Bird(230, random.randint(250, 450))
        base = Base(FLOOR)
        pipes = [Pipe(700)]
        score = 0
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            steps += 1
            s = agent.d.discretize(bird, pipes)
            a = agent.act(s)
            
            if a == 1:
                bird.jump()
            
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
            
            sp = agent.d.discretize(bird, pipes)
            r = improved_reward(not done, passed, bird, pipes, a)
            agent.learn_multihorizon(s, a, r, sp, done)
        
        agent.decay_eps()  # Decay after each episode
        best_score = max(best_score, score)
        total_score += score
    
    return {
        "best_score": best_score,
        "avg": total_score / max(1, num_episodes),
        "eps": agent.eps,
        "Q": agent.Q,
    }

# ==================== Worker ====================
def worker_round(worker_id: int, seed: int, episodes_this_round: int, max_steps: int, 
                 global_q_snapshot: np.ndarray):
    fb = load_fb_module()
    Bird, Pipe, Base, WIN_WIDTH, WIN_HEIGHT, FLOOR = make_env_objects(fb)
    
    random.seed(seed)
    np.random.seed(seed)
    d = Discretizer(FLOOR)
    agent = QAgent(d, init_q=global_q_snapshot)
    
    # Vary exploration per worker
    agent.eps = EXPLORATION_START * (0.8 + 0.4 * (worker_id % 3) / 2.0)
    
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
    if num_workers is None:
        num_workers = mp.cpu_count()
    num_workers = _clamp_workers(num_workers)
    
    rounds = int(math.ceil(total_episodes_per_worker / float(sync_interval)))
    
    # Calculate Q-table size
    q_size = np.prod(q_shape())
    q_mb = (q_size * 4) / (1024 * 1024)  # 4 bytes per float32
    
    print("=" * 60)
    print("FIXED Q-LEARNING FLAPPY BIRD")
    print("=" * 60)
    print(f"CPU Workers: {num_workers}")
    print(f"Episodes/Worker: {total_episodes_per_worker} | Sync: {sync_interval} | Rounds: {rounds}")
    print(f"State space: {q_shape()} = {q_size:,} states")
    print(f"Q-table size: {q_mb:.2f} MB")
    print(f"Initial exploration: {EXPLORATION_START:.2%}")
    print(f"Exploration decay: {EXPLORATION_DECAY} per episode")
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
            avg_eps = float(np.mean([r["eps"] for r in results]))
            
            print(f"[Round {rd+1}/{rounds}] "
                  f"Worker#{best_in_round['worker']} score={best_in_round['best_score']} | "
                  f"Avg={avg_avgs:.2f} | eps={avg_eps:.3f}")
            
            if best_in_round["best_score"] > global_best["score"]:
                global_best["score"] = best_in_round["best_score"]
                global_best["Q"] = best_in_round["Q"].copy()
                print(f"  🎯 NEW BEST: {best_in_round['best_score']} pipes!")
    
    dur = time.time() - start
    total_episodes = num_workers * total_episodes_per_worker
    print("\n" + "=" * 60)
    print(f"✅ TRAINING COMPLETE in {dur:.1f}s | {total_episodes/dur:.1f} eps/sec")
    print(f"🏆 Best Score: {global_best['score']} pipes")
    print("=" * 60)
    
    if SAVE_BEST_Q:
        np.save(EXPORT_DIR / "best_q_table_fixed.npy", global_best["Q"])
        print(f"💾 Saved best Q -> {EXPORT_DIR/'best_q_table_fixed.npy'}")
    
    if SAVE_AVG_Q:
        np.save(EXPORT_DIR / "avg_q_table_fixed.npy", gQ)
        print(f"💾 Saved averaged Q -> {EXPORT_DIR/'avg_q_table_fixed.npy'}")
    
    if SAVE_REPLAY:
        score = record_greedy_replay(global_best["Q"], seed=123, 
                                     save_path=EXPORT_DIR / "replay_fixed.npz")
        print(f"🎥 Replay saved (score {score}) -> {EXPORT_DIR/'replay_fixed.npz'}")
    
    return global_best, gQ

# ==================== Replay Runner ====================
def record_greedy_replay(Q, seed: int, save_path: Path, max_steps: int = 5000):
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
    
    while not done and steps < max_steps:
        steps += 1
        s = d.discretize(bird, pipes)
        a = agent.act(s)
        
        if a == 1:
            bird.jump()
        
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
    
    meta = {"seed": seed, "episode_score": int(score), "note": "fixed Q-learning"}
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
    
    ap = argparse.ArgumentParser(description="Fixed Q-learning for Flappy Bird")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--episodes-per-worker", type=int, default=TOTAL_EPISODES_PER_WORKER)
    ap.add_argument("--sync-interval", type=int, default=SYNC_INTERVAL)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--nice", type=int, default=DEFAULT_NICE_VALUE)
    ap.add_argument("--ionice", action="store_true")
    ap.add_argument("--affinity", type=str, default=None)
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