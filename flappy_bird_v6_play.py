
import torch
import pygame
import numpy as np
from flappy_bird import Bird, Pipe, Base, WIN_WIDTH, WIN_HEIGHT, FLOOR
from v6 import DQNAgent, device

MODEL_PATH = "dqn_flappy_v6R_final.pth"
FPS = 45
NUM_EPISODES = 50 #This can be changed to whatever 


def get_state(bird, pipes):
    """Return the 6-feature normalized state vector used in v6 training."""
    pipe_idx = 0
    if len(pipes) > 1 and bird.x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
        pipe_idx = 1
    top = pipes[pipe_idx].height
    bottom = pipes[pipe_idx].bottom
    gap_center = 0.5 * (top + bottom)
    dx = pipes[pipe_idx].x - bird.x

    
    #NOTEEE: I originally had 3 states, but the model wasnt training well at all and not passing any pipes.
    #I make changes into 6 states and it worked alot better, this can be changed (more details in v6.py)
    '''
    OG 3 States:
    bird.y
    abs(bird.y - pipes[pipe_ind].height)
    abs(bird.y - pipes[pipe_ind].bottom)
    '''
    return np.array([
        bird.y / WIN_HEIGHT,
        abs(bird.y - top) / WIN_HEIGHT,
        abs(bird.y - bottom) / WIN_HEIGHT,
        (bird.y - gap_center) / WIN_HEIGHT,
        bird.vel / 16.0,
        dx / WIN_WIDTH,
    ], dtype=np.float32)


def run_one_episode(agent, win):
    """Run one Flappy Bird episode with rendering."""
    bird = Bird(230, 350)
    base = Base(FLOOR)
    pipes = [Pipe(700)]
    clock = pygame.time.Clock()
    score, done = 0, False

    while not done:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        state = get_state(bird, pipes)
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = torch.argmax(agent.q_net(state_t)).item()

        if action == 1:
            bird.jump()

        bird.move()
        base.move()

        rem, add_pipe = [], False
        for pipe in pipes:
            pipe.move()
            if pipe.collide(bird, win):
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

        # render visuals
        win.blit(pygame.transform.scale(pygame.image.load("imgs/bg.png"), (WIN_WIDTH, WIN_HEIGHT)), (0, 0))
        for pipe in pipes:
            pipe.draw(win)
        base.draw(win)
        bird.draw(win)

        font = pygame.font.SysFont("comicsans", 40)
        score_text = font.render(f"Score: {score}", True, (255, 255, 255))
        win.blit(score_text, (10, 10))
        pygame.display.update()

    return score


def play_dqn_multi():
    print(" Loading trained model and running playback (20 games)...")
    pygame.init()
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("Flappy Bird - DQN Agent (v6 Auto Runs)")

    agent = DQNAgent()
    agent.q_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    agent.q_net.eval()

    scores = []
    for i in range(NUM_EPISODES):
        print(f"\n Episode {i + 1}/{NUM_EPISODES}")
        score = run_one_episode(agent, win)
        scores.append(score)
        print(f"Run {i + 1} finished with score: {score}")

    avg_score = np.mean(scores)
    best_score = np.max(scores)
    print("\n========================================")
    print(f"Finished {NUM_EPISODES} runs")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Best Score: {best_score}")
    print("========================================")

    pygame.quit()


if __name__ == "__main__":
    play_dqn_multi()
