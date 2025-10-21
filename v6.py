
import random
from collections import deque
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from flappy_bird import Bird, Pipe, Base, WIN_WIDTH, WIN_HEIGHT, FLOOR

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Gpu wasnt working for me, maybe its a code or hardward issue for me
print("Training DQN Agent (v6.3R: Restored + Smooth Hop)")
print(f"Using device: {device}")

FPS_TRAIN = 75
MAX_STEPS = 2200
BUFFER_SIZE = 100_000 #<---- this is the replay buffer size. We want to keep it large for better training and performance. can maybe be increased to 200k+. depeing how many RAM you have
BATCH_SIZE = 32
GAMMA = 0.99 #<---- this is the discount factor
LR = 3e-4 #<---- this is the learning rate
TAU = 0.005 #<---- this is the soft update rate
N_STEPS = 3 #<---- this is the n-step buffer size
FORCE_EXPLORE = 1000   #<------ this can vary depending on how many epsiodes we are using,we want to force epsiodes to test random instead of using the eps
EPS_DECAY = 0.999 #<---- this is the epsilon decay rate. Keep at 0.999 or higher. we want to decay slowly for better training and performance.
EPS_MIN = 0.05 #<---- this is the minimum epsilon value. Keep at 0.05 

# MODEL 
class DuelingDQN(nn.Module):
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

# AGENT 
class DQNAgent:
    def __init__(self):
        self.q_net = DuelingDQN().to(device)
        self.target_net = DuelingDQN().to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR) 
        self.memory = deque(maxlen=BUFFER_SIZE)
        self.nstep_buf = deque(maxlen=N_STEPS)

        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        self.tau = TAU
        self.train_steps = 0

        self.epsilon = 1.0
        self.eps_min = EPS_MIN
        self.eps_decay = EPS_DECAY
        self.force_explore = FORCE_EXPLORE

    def get_state(self, bird, pipes):
        pipe_idx = 0
        if len(pipes) > 1 and bird.x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
            pipe_idx = 1
        top = pipes[pipe_idx].height
        bottom = pipes[pipe_idx].bottom
        gap_center = 0.5 * (top + bottom)
        dx = pipes[pipe_idx].x - bird.x

        #NOTEEE: I originally had 3 states, but the model wasnt training well at all and not passing any pipes.
        #I make changes into 6 states and it worked alot better, this can be changed
        '''
        OG 3 States:
        bird.y
        abs(bird.y - pipes[pipe_ind].height)
        abs(bird.y - pipes[pipe_ind].bottom)
        '''
        return np.array([
            bird.y / WIN_HEIGHT, #<--- Birds vertical posiiton
            abs(bird.y - top) / WIN_HEIGHT, #<--- Birds vertical distance to the top of the pipe
            abs(bird.y - bottom) / WIN_HEIGHT, #<--- Birds vertical distance to the bottom of the pipe
            (bird.y - gap_center) / WIN_HEIGHT, #<--- Birds vertical distance to the center of the pipe
            bird.vel / 16.0, #<--- Birds vertical velocity
            dx / WIN_WIDTH, #<--- Birds horizontal distance to the pipe
        ], dtype=np.float32)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q = self.q_net(s)
        return int(q.argmax().item())

    def _append_nstep(self, s, a, r, sn, done):
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
            next_actions = self.q_net(sn).argmax(1)
            next_q = self.target_net(sn).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = r + (self.gamma ** N_STEPS) * next_q * (~d)

        q_sa = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q_sa, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self.soft_update()
        self.train_steps += 1

        if self.train_steps % 300 == 0:
            with torch.no_grad():
                mean_q = self.q_net(s).mean().item()
            print(f"[Step {self.train_steps:5d}] Loss={loss.item():.4f} | MeanQ={mean_q:.2f} | Mem={len(self.memory)}")

    def update_epsilon(self, episode):
        #This is super duper imporant, this is how we do explores  and exploration rate decay. (IMPORTANT FOR PRESENTATIOn)
        if episode < self.force_explore:
            self.epsilon = 1.0
        else:
            self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

# REWARD SHIT
def arrival_reward(alive, passed_pipe, bird, pipes, action):
    if not alive:
        return -5.0 #We dont want no dying bird 
    reward = 0.2
    if passed_pipe:
        reward += 8.0 #I noticed that when it was too high of a reward, the bird would jump TOO much when not needed. can keep at 8.0 or lower.
    pipe_idx = 0
    if len(pipes) > 1 and bird.x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
        pipe_idx = 1
    top = pipes[pipe_idx].height
    bottom = pipes[pipe_idx].bottom
    gap_center = 0.5 * (top + bottom)
    dx = max(0.0, pipes[pipe_idx].x - bird.x)
    dx_norm = min(1.0, dx / WIN_WIDTH)
    center_err = abs(bird.y - gap_center) / WIN_HEIGHT
    proximity_weight = 1.0 - dx_norm
    reward += (1.0 - center_err) * 0.8 * proximity_weight
    if action == 1:
        reward -= 0.02
    vel_term = -((bird.y - gap_center) / WIN_HEIGHT) * (bird.vel / 16.0)
    reward += 0.3 * vel_term
    return float(np.clip(reward, -5.0, 5.0))

#TRAIN LOOP
def train_dqn(episodes=12000):
    print("=" * 40)
    agent = DQNAgent()
    scores = []

    for ep in range(episodes):
        bird = Bird(230, random.randint(250, 450))
        bird.jump_strength = -7  #  Smooth hop (based on observations, the bird sometimes jumps too high and too low. very aggressive jumping lol)
        base = Base(FLOOR)
        pipes = [Pipe(700)]
        score, done, steps = 0, False, 0
        state = agent.get_state(bird, pipes)
        clock = pygame.time.Clock()

        while not done and steps < MAX_STEPS:
            steps += 1
            clock.tick(FPS_TRAIN)

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

            reward = arrival_reward(not done, add_pipe, bird, pipes, action)
            next_state = agent.get_state(bird, pipes) if not done else None

            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state

        scores.append(score)
        avg50 = np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores)
        agent.update_epsilon(ep)

        if ep % 50 == 0:
            print(f"Ep {ep:4d} | Score: {score:2d} | Avg50: {avg50:.2f} | ε={agent.epsilon:.3f} | Mem={len(agent.memory)}")

        if ep % 1000 == 0 and ep > 0:
            torch.save(agent.q_net.state_dict(), f"dqn_flappy_v6R_ep{ep}.pth")
            #I added this so you can have a checkpoint of the model at each 1000 episodes, ive had times where my IDE crashed mid run and lost all my weights. 
            print(f" Model saved at episode {ep}")

    torch.save(agent.q_net.state_dict(), "dqn_flappy_v6R_final.pth")
    print("Training complete!")
    print(f"Final Avg50: {np.mean(scores[-50:]):.2f}")

if __name__ == "__main__":
    pygame.init()
    train_dqn(episodes=12000) #<--- this is the number of episodes to train for. You can change it to whatever you want. HIGHER IS BETTER. If we get a CPU, i want to do like 20k or more
    #Dont forget to also change line 216 to the same number of episodes
