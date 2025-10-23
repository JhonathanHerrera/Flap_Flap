# file: dqn_flappy_multi_fixed.py
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import multiprocessing as mp

from flappy_bird_easy import Bird, Pipe, Base, WIN_WIDTH, WIN_HEIGHT, FLOOR

# ---------------------- Repro / Device ----------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- Hyperparameters ---------------------
NUM_ENVS          = 15
TOTAL_ENV_STEPS   = 10_000_000
MAX_STEPS_PER_EP  = 4000

BUFFER_SIZE       = 1_000_000
BATCH_SIZE        = 256
WARMUP_STEPS      = 100_000

# CRUCIAL: match updates to env steps (≈1 update per env-step)
UPDATE_RATIO      = 1.0   # updates per single-environment step
TARGET_TAU        = 0.005
GAMMA             = 0.99
LR                = 2.5e-4
GRAD_CLIP_NORM    = 5.0
ACTION_REPEAT     = 2

# ε by global env steps
EPS_START         = 1.0
EPS_END           = 0.05
EPS_DECAY_STEPS   = 6_000_000

# Rewards
REWARD_DEATH      = -1.0
REWARD_PASS_PIPE  = +10.0
REWARD_ALIVE      = +0.01
REWARD_FLAP_PEN   = -0.003
CENTER_BONUS_W    = 0.90   # stronger center bonus
BONUS_DX_WINDOW   = 0.35

LOG_EVERY_UPDATES = 5000
SAVE_EVERY_STEPS  = 500_000

OUTDIR = Path("checkpoints_multi")
OUTDIR.mkdir(exist_ok=True)

STATE_DIM = 7
N_ACTIONS = 2

# ---------------------- Network -----------------------------
class QNet(nn.Module):
    def __init__(self, input_dim=STATE_DIM, hidden=512, output_dim=N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )
    def forward(self, x): return self.net(x)

# ---------------------- Replay Buffer -----------------------
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.s = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self.a = np.zeros((capacity,), dtype=np.int64)
        self.r = np.zeros((capacity,), dtype=np.float32)
        self.sn = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self.d = np.zeros((capacity,), dtype=np.bool_)
        self.cap = capacity
        self.idx = 0
        self.full = False
    def __len__(self): return self.cap if self.full else self.idx
    def push(self, s, a, r, sn, d):
        i = self.idx
        self.s[i] = s; self.a[i] = a; self.r[i] = r; self.sn[i] = sn; self.d[i] = d
        self.idx = (self.idx + 1) % self.cap
        self.full = self.full or self.idx == 0
    def sample(self, batch_size: int):
        m = len(self)
        idxs = np.random.randint(0, m, size=batch_size)
        return (
            torch.as_tensor(self.s[idxs],  dtype=torch.float32, device=device),
            torch.as_tensor(self.a[idxs],  dtype=torch.long,    device=device),
            torch.as_tensor(self.r[idxs],  dtype=torch.float32, device=device),
            torch.as_tensor(self.sn[idxs], dtype=torch.float32, device=device),
            torch.as_tensor(self.d[idxs],  dtype=torch.bool,    device=device),
        )

# ---------------------- Schedules / State -------------------
def epsilon_by_step(t: int) -> float:
    if t >= EPS_DECAY_STEPS: return EPS_END
    return EPS_START + (EPS_END - EPS_START) * (t / float(EPS_DECAY_STEPS))

def _next_pipe_idx(bird, pipes) -> int:
    idx = 0
    if len(pipes) > 1 and bird.x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
        idx = 1
    return idx

def get_state(bird, pipes) -> np.ndarray:
    if not pipes:
        return np.zeros(STATE_DIM, dtype=np.float32)
    i = _next_pipe_idx(bird, pipes)
    p = pipes[i]
    top, bottom = p.height, p.bottom
    gap_center = 0.5 * (top + bottom)
    gap_size = bottom - top
    dx = (p.x - bird.x) / WIN_WIDTH
    y_norm = bird.y / WIN_HEIGHT
    v_norm = bird.vel / 16.0
    center_off = (bird.y - gap_center) / max(1.0, gap_size)
    top_rel = (top - bird.y) / WIN_HEIGHT
    bot_rel = (bottom - bird.y) / WIN_HEIGHT
    return np.array([y_norm, v_norm, dx, center_off, gap_size/WIN_HEIGHT, top_rel, bot_rel], dtype=np.float32)

def shaped_reward(alive: bool, passed_pipe: bool, bird, pipes, action: int) -> float:
    r = REWARD_ALIVE
    if passed_pipe: r += REWARD_PASS_PIPE
    if action == 1: r += REWARD_FLAP_PEN
    if pipes:
        i = _next_pipe_idx(bird, pipes)
        p = pipes[i]
        dx = (p.x - bird.x) / WIN_WIDTH
        if 0.0 <= dx <= BONUS_DX_WINDOW:
            gap_center = 0.5 * (p.height + p.bottom)
            gap_half = max(1.0, (p.bottom - p.height) / 2.0)
            dist = abs(bird.y - gap_center) / gap_half
            bonus = CENTER_BONUS_W * (1.0 - min(1.0, dist))
            proximity = 1.0 - (dx / BONUS_DX_WINDOW)
            r += bonus * proximity
    if not alive:
        r += REWARD_DEATH
    return float(np.clip(r, -1.0, 1.0))

# ---------------------- Worker Env --------------------------
@dataclass
class StepResult:
    ns: np.ndarray
    r: float
    d: bool
    passed: bool
    score_inc: int

def worker_proc(conn, seed: int):
    random.seed(seed); np.random.seed(seed)
    pygame.init()
    try:
        bird = Bird(230, random.randint(250, 450))
        base = Base(FLOOR)
        pipes = [Pipe(700)]
        steps = 0
        ep_steps = 0
        ep_score = 0

        def reset_env():
            nonlocal bird, base, pipes, steps, ep_steps, ep_score
            bird = Bird(230, random.randint(250, 450))
            base = Base(FLOOR)
            pipes = [Pipe(700)]
            steps = 0
            ep_steps = 0
            ep_score = 0
            return get_state(bird, pipes)

        s = get_state(bird, pipes)

        while True:
            cmd, payload = conn.recv()
            if cmd == "reset":
                s = reset_env()
                conn.send(s)
            elif cmd == "step":
                action = int(payload)
                total_passed = False
                done = False
                for _ in range(ACTION_REPEAT):
                    if action == 1:
                        bird.jump()
                    bird.move()
                    base.move()
                    rem, add_pipe = [], False
                    for p in pipes:
                        p.move()
                        if p.collide(bird, None):
                            done = True
                            break
                        if p.x + p.PIPE_TOP.get_width() < 0:
                            rem.append(p)
                        if not p.passed and p.x < bird.x:
                            p.passed = True
                            add_pipe = True
                    if add_pipe:
                        pipes.append(Pipe(WIN_WIDTH))
                        ep_score += 1
                        total_passed = True
                    for rp in rem:
                        pipes.remove(rp)
                    if bird.y + bird.img.get_height() >= FLOOR or bird.y < -50:
                        done = True
                    if done:
                        break
                r = shaped_reward(not done, total_passed, bird, pipes, action)
                ns = get_state(bird, pipes) if not done else np.zeros_like(s, dtype=np.float32)
                s_out = StepResult(ns=ns, r=r, d=done, passed=total_passed, score_inc=(1 if total_passed else 0))
                conn.send(s_out)
                s = ns
                steps += 1; ep_steps += 1
                if done or ep_steps >= MAX_STEPS_PER_EP:
                    s = reset_env()
            elif cmd == "close":
                break
    finally:
        try: conn.close()
        except Exception: pass
        pygame.quit()

class VecEnv:
    def __init__(self, n: int):
        self.n = n
        self.parent_conns, self.child_conns = zip(*[mp.Pipe() for _ in range(n)])
        self.ps = []
        for i in range(n):
            p = mp.Process(target=worker_proc, args=(self.child_conns[i], SEED + i + 1), daemon=True)
            p.start()
            self.ps.append(p)
        for c in self.child_conns: c.close()
        for pc in self.parent_conns: pc.send(("reset", None))
        self.states = [pc.recv() for pc in self.parent_conns]
    def step(self, actions: List[int]) -> List[StepResult]:
        for pc, a in zip(self.parent_conns, actions):
            pc.send(("step", int(a)))
        return [pc.recv() for pc in self.parent_conns]
    def reset(self):
        for pc in self.parent_conns: pc.send(("reset", None))
        self.states = [pc.recv() for pc in self.parent_conns]
        return self.states
    def close(self):
        for pc in self.parent_conns:
            try: pc.send(("close", None))
            except Exception: pass
        for p in self.ps:
            p.join(timeout=1.0)
            if p.is_alive(): p.terminate()

# ---------------------- Agent / Trainer ---------------------
class DQNAgent:
    def __init__(self):
        self.q = QNet().to(device)
        self.tgt = QNet().to(device)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=LR)
        self.buf = ReplayBuffer(BUFFER_SIZE)
        self.train_steps = 0

    @torch.no_grad()
    def act_batch(self, states: np.ndarray, eps: float) -> np.ndarray:
        # Per-env ε-greedy
        s = torch.as_tensor(np.array(states), dtype=torch.float32, device=device)
        q = self.q(s)
        greedy = q.argmax(1).detach().cpu().numpy().astype(np.int64)
        rand_actions = np.random.randint(0, N_ACTIONS, size=len(states), dtype=np.int64)
        explore_mask = (np.random.rand(len(states)) < eps)
        return np.where(explore_mask, rand_actions, greedy)

    @torch.no_grad()
    def _soft_update(self):
        for tp, p in zip(self.tgt.parameters(), self.q.parameters()):
            tp.data.mul_(1.0 - TARGET_TAU).add_(TARGET_TAU * p.data)

    def train_step(self):
        if len(self.buf) < BATCH_SIZE: return None
        s, a, r, sn, d = self.buf.sample(BATCH_SIZE)
        with torch.no_grad():
            na = self.q(sn).argmax(1, keepdim=True)
            nq = self.tgt(sn).gather(1, na).squeeze(1)
            target = r + GAMMA * nq * (~d)
        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), GRAD_CLIP_NORM)
        self.opt.step()
        self._soft_update()
        self.train_steps += 1
        return float(loss.item())

# ---------------------- Main Training Loop ------------------
def train():
    print(f"CUDA: {device}, ENVs: {NUM_ENVS}")
    vec = VecEnv(NUM_ENVS)
    agent = DQNAgent()

    global_steps = 0
    updates_backlog = 0.0

    # Proper episodic score tracking per env
    recent_scores = deque(maxlen=1000)  # rolling window across all workers (episodic)
    env_scores = [0 for _ in range(NUM_ENVS)]  # current episode scores per env

    try:
        while global_steps < TOTAL_ENV_STEPS:
            eps = EPS_START if global_steps < WARMUP_STEPS else epsilon_by_step(global_steps)

            # Act
            actions = np.random.randint(0, N_ACTIONS, size=NUM_ENVS, dtype=np.int64) if global_steps < WARMUP_STEPS \
                      else agent.act_batch(vec.states, eps)

            results = vec.step(actions)
            next_states = []
            for i, res in enumerate(results):
                agent.buf.push(vec.states[i], int(actions[i]), float(res.r), res.ns, bool(res.d))
                next_states.append(res.ns)
                if res.score_inc:
                    env_scores[i] += res.score_inc
                if res.d:
                    # episode finished for this worker
                    recent_scores.append(env_scores[i])
                    env_scores[i] = 0

            vec.states = next_states
            global_steps += NUM_ENVS

            # Schedule updates proportional to data collected
            if global_steps >= WARMUP_STEPS:
                updates_backlog += NUM_ENVS * UPDATE_RATIO  # collect update credits
                num_updates = int(updates_backlog)
                if num_updates > 0:
                    last_loss = None
                    for _ in range(num_updates):
                        loss_val = agent.train_step()
                        if loss_val is not None:
                            last_loss = loss_val
                    updates_backlog -= num_updates

                    # Logging (by update count)
                    if agent.train_steps and (agent.train_steps % LOG_EVERY_UPDATES == 0) and last_loss is not None:
                        probe = torch.as_tensor(np.array(vec.states[:min(32, NUM_ENVS)]), dtype=torch.float32, device=device)
                        mean_q = float(agent.q(probe).mean().item())
                        avg_score = (np.mean(recent_scores) if recent_scores else 0.0)
                        print(f"[train {agent.train_steps:7d} | steps {global_steps:9d}] "
                              f"loss={last_loss:.4f} meanQ={mean_q:.2f} eps={eps:.3f} "
                              f"buf={len(agent.buf)} avgEp={avg_score:.2f}")

            if (global_steps % SAVE_EVERY_STEPS) == 0:
                torch.save(agent.q.state_dict(), OUTDIR / f"steps_{global_steps}.pth")
    except KeyboardInterrupt:
        torch.save(agent.q.state_dict(), OUTDIR / "interrupt.pth")
        print("\nInterrupted. Saved to:", OUTDIR / "interrupt.pth")
    finally:
        vec.close()

    torch.save(agent.q.state_dict(), OUTDIR / "final.pth")
    print("Done. Saved to:", OUTDIR / "final.pth")

if __name__ == "__main__":
    pygame.init()
    train()
    pygame.quit()
