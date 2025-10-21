# file: train_flappy_q_multihorizon.py
import random
import time
import json
from pathlib import Path

import numpy as np
import pygame

# Use your provided environment
from flappy_bird import Bird, Pipe, Base, WIN_WIDTH, WIN_HEIGHT, FLOOR

# ==================== Hyperparameters ====================
LEARNING_RATE = 0.08
DISCOUNT_FACTOR = 0.995
EXPLORATION_START = 0.25
EXPLORATION_DECAY = 0.9995
MIN_EXPLORATION = 0.02

MAX_STEPS = 3000
EPISODES = 2000

# Discretization bins
Y_BINS = 8
DY_GAP_BINS = 12
TTB_BINS = 10
VEL_BINS = 7
NEXT_GAP_BINS = 8

ACTIONS = 2  # 0=idle, 1=jump
FLAP_PENALTY = 0.02

# Multi-horizon steps for bootstrap
HORIZONS = [5, 10, 20, 50]

# Replay export
EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== Utilities / Features ====================
def pipe_index_for_bird(bird_x, pipes):
    """First pipe whose right edge is ahead of the bird."""
    if not pipes:
        return 0
    for i, p in enumerate(pipes):
        if bird_x <= p.x + p.PIPE_TOP.get_width():
            return i
    return len(pipes) - 1

def gap_center(p): return (p.height + p.bottom) / 2.0
def gap_size(p):   return max(1.0, p.bottom - p.height)

def avg_forward_speed():
    # Rough forward progress in px/frame for time-to-pipe
    return 8.0

# ==================== Discretizer ====================
class Discretizer:
    """Robust bins for compact tabular Q."""
    def __init__(self):
        self.y_edges = np.linspace(0, FLOOR, Y_BINS + 1)[1:-1]
        self.dy_max = 250.0
        self.dy_edges = np.linspace(0, self.dy_max, DY_GAP_BINS + 1)[1:-1]
        self.ttb_max = 80.0
        self.ttb_edges = np.linspace(0, self.ttb_max, TTB_BINS + 1)[1:-1]
        self.vel_min, self.vel_max = -12.0, 12.0
        self.vel_edges = np.linspace(self.vel_min, self.vel_max, VEL_BINS + 1)[1:-1]
        self.next_gap_edges = np.linspace(0, FLOOR, NEXT_GAP_BINS + 1)[1:-1]
        self.v_forward = avg_forward_speed()

    @staticmethod
    def _bin(val, edges):
        return int(np.clip(np.digitize(val, edges), 0, len(edges)))

    def discretize(self, bird, pipes):
        if not pipes:
            return (0, 0, 0, VEL_BINS // 2, 0)
        idx = pipe_index_for_bird(bird.x, pipes)
        cur = pipes[idx]
        nxt = pipes[min(idx + 1, len(pipes) - 1)]

        y = float(np.clip(bird.y, 0, FLOOR))
        gc = float(np.clip(gap_center(cur), 0, FLOOR))
        dy = min(self.dy_max, abs(y - gc))
        dist = max(0.0, cur.x - bird.x)
        ttb = min(self.ttb_max, dist / max(1e-3, self.v_forward))
        vel = float(np.clip(bird.vel, self.vel_min, self.vel_max))
        next_gc = float(np.clip(gap_center(nxt), 0, FLOOR))

        return (
            self._bin(y, self.y_edges),
            self._bin(dy, self.dy_edges),
            self._bin(ttb, self.ttb_edges),
            self._bin(vel, self.vel_edges),
            self._bin(next_gc, self.next_gap_edges),
        )

# ==================== Distance-based Reward ====================
def distance_loss(bird, pipes):
    """Absolute vertical distance to the horizontal line at the nearest gap center."""
    if not pipes:
        return 0.0
    idx = pipe_index_for_bird(bird.x, pipes)
    p = pipes[idx]
    return abs(float(bird.y) - float(gap_center(p)))

def shaped_reward(alive, passed_pipe, bird, pipes, loss_weight=0.02):
    """Dense negative of distance; small extras for stability."""
    if not alive:
        return -4.0
    L = distance_loss(bird, pipes)
    r = -loss_weight * L  # core signal (highly important per requirements)
    if passed_pipe:
        r += 3.0  # keep small to not dwarf the distance term
    # Mild preference for being inside the gap
    if pipes:
        idx = pipe_index_for_bird(bird.x, pipes)
        p = pipes[idx]
        if p.height <= bird.y <= p.bottom:
            r += 0.2
    return r

# ==================== Agent ====================
class QAgent:
    """Tabular Q with multi-horizon bootstrap targets."""
    def __init__(self, d: Discretizer):
        self.d = d
        self.shape = (Y_BINS, DY_GAP_BINS, TTB_BINS, VEL_BINS, NEXT_GAP_BINS, ACTIONS)
        self.Q = np.zeros(self.shape, dtype=np.float32)
        self.Q[..., 0] = 0.2  # optimistic idle
        self.alpha = LEARNING_RATE
        self.gamma = DISCOUNT_FACTOR
        self.eps = EXPLORATION_START

        # Buffers
        self.s_buf, self.a_buf, self.r_buf = [], [], []

        self.best_score = 0

    @staticmethod
    def _argmax_tie(arr):
        m = np.max(arr)
        idxs = np.flatnonzero(arr == m)
        return int(random.choice(idxs))

    def act(self, s):
        if random.random() < self.eps:
            return random.randint(0, ACTIONS - 1)
        return self._argmax_tie(self.Q[s])

    def decay_eps(self):
        self.eps = max(MIN_EXPLORATION, self.eps * EXPLORATION_DECAY)

    # -------- Multi-horizon n-step update --------
    def _update_ready(self, done):
        return any(len(self.r_buf) >= h for h in HORIZONS) or (done and len(self.r_buf) > 0)

    def _compute_target(self, start_idx, next_state, done):
        """Average of available n-step targets over requested horizons."""
        targets = []
        max_len = len(self.r_buf) - start_idx
        for h in HORIZONS:
            h_eff = min(h, max_len)
            G = 0.0
            for i in range(h_eff):
                G += (self.gamma ** i) * self.r_buf[start_idx + i]
            # bootstrap only if we have exactly h steps and not terminal
            can_bootstrap = (not done) and (max_len >= h)
            if can_bootstrap:
                G += (self.gamma ** h) * np.max(self.Q[next_state])
            targets.append(G)
        return float(np.mean(targets))

    def update_multi_h(self, next_state, done):
        """Pop oldest transition; update with averaged multi-horizon target."""
        while self._update_ready(done):
            # always update oldest tuple
            s0 = self.s_buf[0]
            a0 = self.a_buf[0]
            target = self._compute_target(0, next_state, done)
            q_sa = self.Q[s0][a0]
            self.Q[s0][a0] = q_sa + self.alpha * (target - q_sa)
            # pop oldest reward and (s,a)
            self.s_buf.pop(0)
            self.a_buf.pop(0)
            self.r_buf.pop(0)

# ==================== Replay Recorder ====================
class ReplayRecorder:
    """Record minimal state for deterministic playback."""
    def __init__(self):
        self.t = []; self.bx=[]; self.by=[]; self.bv=[]
        self.score=[]; self.cur_px=[]; self.cur_top=[]; self.cur_bot=[]
        self.nxt_px=[]; self.nxt_top=[]; self.nxt_bot=[]

    def log(self, t, bird, pipes, score):
        idx = pipe_index_for_bird(bird.x, pipes) if pipes else 0
        cur = pipes[idx] if pipes else None
        nxt = pipes[min(idx + 1, len(pipes) - 1)] if pipes else None

        def px(p): return float(p.x) if p else float(WIN_WIDTH + 100)
        def pt(p): return float(p.height) if p else 0.0
        def pb(p): return float(p.bottom) if p else float(FLOOR)

        self.t.append(t)
        self.bx.append(float(bird.x)); self.by.append(float(bird.y)); self.bv.append(float(bird.vel))
        self.score.append(int(score))
        self.cur_px.append(px(cur)); self.cur_top.append(pt(cur)); self.cur_bot.append(pb(cur))
        self.nxt_px.append(px(nxt)); self.nxt_top.append(pt(nxt)); self.nxt_bot.append(pb(nxt))

    def save(self, path: Path, meta: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            t=np.array(self.t, dtype=np.int32),
            bx=np.array(self.bx, dtype=np.float32),
            by=np.array(self.by, dtype=np.float32),
            bv=np.array(self.bv, dtype=np.float32),
            score=np.array(self.score, dtype=np.int32),
            cur_px=np.array(self.cur_px, dtype=np.float32),
            cur_top=np.array(self.cur_top, dtype=np.float32),
            cur_bot=np.array(self.cur_bot, dtype=np.float32),
            nxt_px=np.array(self.nxt_px, dtype=np.float32),
            nxt_top=np.array(self.nxt_top, dtype=np.float32),
            nxt_bot=np.array(self.nxt_bot, dtype=np.float32),
            meta=json.dumps(meta).encode("utf-8"),
        )

# ==================== Training Loop ====================
def train(seed=7, episodes=EPISODES):
    random.seed(seed); np.random.seed(seed)
    d = Discretizer()
    agent = QAgent(d)
    all_scores = []
    best = {"score": -1, "Q": None}

    for ep in range(1, episodes + 1):
        bird = Bird(230, random.randint(250, 450))
        base = Base(FLOOR)
        # Start with a normal pipe set; we use your game logic
        pipes = [Pipe(700)]
        score, done, steps = 0, False, 0

        # clear buffers
        agent.s_buf.clear(); agent.a_buf.clear(); agent.r_buf.clear()

        s = d.discretize(bird, pipes)

        while not done and steps < MAX_STEPS:
            steps += 1
            a = agent.act(s)
            if a == 1:
                bird.jump()

            bird.move()
            base.move()

            rem, passed = [], False
            for p in list(pipes):
                p.move()
                if p.collide(bird, None):
                    done = True; break
                if p.x + p.PIPE_TOP.get_width() < 0:
                    rem.append(p)
                if not p.passed and p.x < bird.x:
                    p.passed = True
                    passed = True
            for r in rem: pipes.remove(r)
            if passed:
                score += 1
                pipes.append(Pipe(WIN_WIDTH))  # spawn next pipe

            if bird.y + bird.img.get_height() >= FLOOR or bird.y < -50:
                done = True

            # reward from distance loss + events
            r = shaped_reward(not done, passed, bird, pipes) - (FLAP_PENALTY if a == 1 else 0.0)

            # buffer + multi-horizon update
            agent.s_buf.append(s); agent.a_buf.append(a); agent.r_buf.append(r)
            s_next = d.discretize(bird, pipes) if not done else s
            agent.update_multi_h(s_next, done)

            s = s_next

        all_scores.append(score)
        agent.best_score = max(agent.best_score, score)
        if score > best["score"]:
            best = {"score": score, "Q": agent.Q.copy()}

        agent.decay_eps()

        if ep % 100 == 0:
            last = all_scores[-100:] if len(all_scores) >= 100 else all_scores
            print(f"[Ep {ep:4d}] Score:{score:3d} | Avg100:{np.mean(last):.2f} | "
                  f"Max100:{max(last) if last else 0:3d} | Best:{agent.best_score:3d} | ε={agent.eps:.3f}")

    # Save best Q
    np.save(EXPORT_DIR / "best_q_table_multih.npy", best["Q"])
    print(f"💾 Saved best Q (score={best['score']}) -> {EXPORT_DIR/'best_q_table_multih.npy'}")

    # Deterministic greedy replay for best Q
    rp_path = EXPORT_DIR / "replay_best_multih.npz"
    score = record_greedy_replay(best["Q"], seed=123, save_path=rp_path)
    print(f"🎥 Replay saved (greedy score {score}) -> {rp_path}")

    return agent, best, all_scores

# ==================== Replay Runner ====================
def record_greedy_replay(Q, seed: int, save_path: Path, max_steps: int = 4000):
    random.seed(seed); np.random.seed(seed)
    d = Discretizer()
    # temp agent with provided Q
    agent = QAgent(d)
    agent.Q = Q.copy()
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
        if a == 1: bird.jump()
        bird.move(); base.move()

        rem, passed = [], False
        for p in list(pipes):
            p.move()
            if p.collide(bird, None):
                done = True; break
            if p.x + p.PIPE_TOP.get_width() < 0:
                rem.append(p)
            if not p.passed and p.x < bird.x:
                p.passed = True; passed = True
        for r in rem: pipes.remove(r)
        if passed:
            score += 1
            pipes.append(Pipe(WIN_WIDTH))
        if bird.y + bird.img.get_height() >= FLOOR or bird.y < -50:
            done = True

        rec.log(steps, bird, pipes, score)

    meta = {"seed": seed, "episode_score": int(score), "note": "multi-horizon Q-learning"}
    rec.save(save_path, meta)
    return score

# ==================== Entrypoint ====================
if __name__ == "__main__":
    start = time.time()
    train()
    print(f"Done in {time.time()-start:.1f}s")
