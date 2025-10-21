"""
Enhanced Flappy Bird with Progressive Difficulty
Features:
- Progressively smaller pipe gaps
- Increasing game speed over time
- Dynamic pipe spacing
- Score multipliers for difficulty
- Variable pipe heights
- Color changes as difficulty increases
"""

import pygame
import random
import os
import time
#import neat
import visualize
import pickle
import math
pygame.font.init()

WIN_WIDTH = 600
WIN_HEIGHT = 800
FLOOR = 730
STAT_FONT = pygame.font.SysFont("comicsans", 50)
END_FONT = pygame.font.SysFont("comicsans", 70)
SMALL_FONT = pygame.font.SysFont("comicsans", 30)
DRAW_LINES = False

WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird - Enhanced")

pipe_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","pipe.png")).convert_alpha())
bg_img = pygame.transform.scale(pygame.image.load(os.path.join("imgs","bg.png")).convert_alpha(), (600, 900))
bird_images = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird" + str(x) + ".png"))) for x in range(1,4)]
base_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","base.png")).convert_alpha())

gen = 0

import os as _os
_HEADLESS = _os.environ.get("SDL_VIDEODRIVER") == "dummy" or _os.environ.get("FB_NO_DISPLAY") == "1"
if not _HEADLESS:
    WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("Flappy Bird - Enhanced")
else:
    WIN = None  # training doesn't render

class Bird:
    """
    Enhanced Bird class with adaptive mechanics
    """
    MAX_ROTATION = 25
    IMGS = bird_images
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        """Enhanced jump with speed consideration"""
        self.vel = -7.5
        self.tick_count = 0
        self.height = self.y

    def move(self, speed_multiplier=1.0):
        """Enhanced move with speed multiplier"""
        self.tick_count += 1

        # Adjust physics based on game speed
        displacement = self.vel*(self.tick_count) + 0.5*(2.5 * speed_multiplier)*(self.tick_count)**2

        # Terminal velocity
        if displacement >= 16:
            displacement = (displacement/abs(displacement)) * 16

        if displacement < 0:
            displacement -= 2

        self.y = self.y + displacement

        if displacement < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        self.img_count += 1

        # Animation cycle
        if self.img_count <= self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count <= self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]
        elif self.img_count <= self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]
        elif self.img_count <= self.ANIMATION_TIME*4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME*4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME*2

        blitRotateCenter(win, self.img, (self.x, self.y), self.tilt)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Pipe():
    """
    Enhanced pipe with progressive difficulty
    """
    INITIAL_GAP = 300
    MIN_GAP = 100
    GAP_DECREASE_RATE = 3  # Pixels to decrease per pipe
    INITIAL_VEL = 5
    MAX_VEL = 12

    def __init__(self, x, pipes_passed=0, base_vel=5):
        self.x = x
        self.height = 0
        self.pipes_passed = pipes_passed
        
        # Progressive gap reduction
        self.GAP = max(self.MIN_GAP, 
                      self.INITIAL_GAP - (self.GAP_DECREASE_RATE * pipes_passed))
        
        # Progressive speed increase
        self.VEL = min(self.MAX_VEL, base_vel + (pipes_passed * 0.15))
        
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(pipe_img, False, True)
        self.PIPE_BOTTOM = pipe_img
        self.passed = False
        
        # Color tinting based on difficulty
        self.difficulty_level = min(pipes_passed / 20, 1.0)
        self.apply_difficulty_tint()
        
        self.set_height()

    def apply_difficulty_tint(self):
        """Apply color tint based on difficulty"""
        if self.difficulty_level > 0.3:
            # Create tinted versions of pipes
            tint_color = (255, 
                         int(255 * (1 - self.difficulty_level * 0.3)),
                         int(255 * (1 - self.difficulty_level * 0.5)))
            
            # Create surface copies for tinting
            top_copy = self.PIPE_TOP.copy()
            bottom_copy = self.PIPE_BOTTOM.copy()
            
            # Apply tint
            top_copy.fill(tint_color, special_flags=pygame.BLEND_MULT)
            bottom_copy.fill(tint_color, special_flags=pygame.BLEND_MULT)
            
            self.PIPE_TOP = top_copy
            self.PIPE_BOTTOM = bottom_copy

    def set_height(self):
        """Enhanced height variation"""
        # More aggressive height variation as game progresses
        height_variance = 50 + int(self.pipes_passed * 2)
        min_height = max(50, 100 - height_variance)
        max_height = min(450, 400 + height_variance)
        
        self.height = random.randrange(min_height, max_height)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird, win):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if b_point or t_point:
            return True
        return False


class Base:
    """
    Enhanced moving floor with variable speed
    """
    INITIAL_VEL = 5
    WIDTH = base_img.get_width()
    IMG = base_img

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH
        self.VEL = self.INITIAL_VEL

    def move(self, speed_multiplier=1.0):
        """Move with speed multiplier"""
        current_vel = self.VEL * speed_multiplier
        
        self.x1 -= current_vel
        self.x2 -= current_vel
        
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


class GameDifficulty:
    """Manages progressive difficulty"""
    def __init__(self):
        self.pipes_passed = 0
        self.base_speed = 5
        self.speed_multiplier = 1.0
        self.score_multiplier = 1.0
        self.pipe_spacing = 600
        self.min_pipe_spacing = 400
        
    def update(self, pipes_passed):
        """Update difficulty based on progress"""
        self.pipes_passed = pipes_passed
        
        # Speed increases every 5 pipes
        self.speed_multiplier = 1.0 + (pipes_passed * 0.03)
        self.speed_multiplier = min(self.speed_multiplier, 2.5)
        
        # Score multiplier increases with difficulty
        self.score_multiplier = 1.0 + (pipes_passed * 0.1)
        
        # Pipe spacing decreases
        self.pipe_spacing = max(self.min_pipe_spacing,
                               600 - (pipes_passed * 5))
        
    def get_pipe_spawn_distance(self):
        """Get dynamic pipe spawn distance"""
        # Add some randomness to pipe spacing
        variance = random.randint(-50, 50)
        return self.pipe_spacing + variance


def blitRotateCenter(surf, image, topleft, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center = image.get_rect(topleft = topleft).center)
    surf.blit(rotated_image, new_rect.topleft)


def draw_window(win, birds, pipes, base, score, gen, pipe_ind, difficulty):
    """Enhanced draw with difficulty indicators"""
    if gen == 0:
        gen = 1
    
    # Dynamic background tint based on difficulty
    win.blit(bg_img, (0,0))
    
    # Add difficulty overlay
    if difficulty.speed_multiplier > 1.5:
        overlay = pygame.Surface((WIN_WIDTH, WIN_HEIGHT))
        overlay.set_alpha(int(20 * (difficulty.speed_multiplier - 1.5)))
        overlay.fill((200, 50, 50))
        win.blit(overlay, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

    base.draw(win)
    
    for bird in birds:
        if DRAW_LINES and pipe_ind < len(pipes):
            try:
                pygame.draw.line(win, (255,0,0), 
                               (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2),
                               (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_TOP.get_width()/2, pipes[pipe_ind].height), 5)
                pygame.draw.line(win, (255,0,0),
                               (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2),
                               (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_BOTTOM.get_width()/2, pipes[pipe_ind].bottom), 5)
            except:
                pass
        bird.draw(win)

    # Score with multiplier
    actual_score = int(score * difficulty.score_multiplier)
    score_label = STAT_FONT.render(f"Score: {actual_score}", 1, (255,255,255))
    win.blit(score_label, (WIN_WIDTH - score_label.get_width() - 15, 10))
    
    # Difficulty indicators
    diff_label = SMALL_FONT.render(f"Speed: {difficulty.speed_multiplier:.1f}x", 1, (255,255,0))
    win.blit(diff_label, (WIN_WIDTH - diff_label.get_width() - 15, 60))
    
    gap_label = SMALL_FONT.render(f"Gap: {pipes[0].GAP if pipes else 200}px", 1, (255,200,0))
    win.blit(gap_label, (WIN_WIDTH - gap_label.get_width() - 15, 90))

    # Generation info
    gen_label = STAT_FONT.render(f"Gen: {gen-1}", 1, (255,255,255))
    win.blit(gen_label, (10, 10))

    alive_label = STAT_FONT.render(f"Alive: {len(birds)}", 1, (255,255,255))
    win.blit(alive_label, (10, 50))

    pygame.display.update()


def eval_genomes(genomes, config):
    """Enhanced evaluation with progressive difficulty"""
    global WIN, gen
    win = WIN
    gen += 1

    nets = []
    birds = []
    ge = []
    
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        ge.append(genome)

    base = Base(FLOOR)
    difficulty = GameDifficulty()
    pipes = [Pipe(700, 0, difficulty.base_speed)]
    score = 0
    pipes_passed = 0
    
    clock = pygame.time.Clock()
    run = True
    
    while run and len(birds) > 0:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1

        for x, bird in enumerate(birds):
            # Fitness increases with difficulty
            ge[x].fitness += 0.1 * difficulty.score_multiplier
            bird.move(difficulty.speed_multiplier)

            # Neural network decision
            if pipe_ind < len(pipes):
                output = nets[birds.index(bird)].activate((
                    bird.y, 
                    abs(bird.y - pipes[pipe_ind].height),
                    abs(bird.y - pipes[pipe_ind].bottom)
                ))

                if output[0] > 0.5:
                    bird.jump()

        base.move(difficulty.speed_multiplier)

        rem = []
        add_pipe = False
        
        for pipe in pipes:
            pipe.move()
            
            # Check collisions
            for bird in birds[:]:
                if pipe.collide(bird, win):
                    ge[birds.index(bird)].fitness -= 1
                    nets.pop(birds.index(bird))
                    ge.pop(birds.index(bird))
                    birds.pop(birds.index(bird))

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            if not pipe.passed and len(birds) > 0 and pipe.x < birds[0].x:
                pipe.passed = True
                add_pipe = True
                pipes_passed += 1
                difficulty.update(pipes_passed)

        if add_pipe:
            score += 1
            # Bonus fitness for passing harder pipes
            for genome in ge:
                genome.fitness += 5 * difficulty.score_multiplier
            
            # Dynamic pipe spawning
            spawn_distance = difficulty.get_pipe_spawn_distance()
            pipes.append(Pipe(pipes[-1].x + spawn_distance, 
                            pipes_passed, 
                            difficulty.base_speed))

        for r in rem:
            pipes.remove(r)

        # Remove birds that hit boundaries
        for bird in birds[:]:
            if bird.y + bird.img.get_height() - 10 >= FLOOR or bird.y < -50:
                if bird in birds:
                    idx = birds.index(bird)
                    nets.pop(idx)
                    ge.pop(idx)
                    birds.pop(idx)

        draw_window(WIN, birds, pipes, base, score, gen, pipe_ind, difficulty)

        # Optional: Stop if score is high enough
        if score > 100:
            pickle.dump(nets[0], open("best.pickle", "wb"))
            break


def run(config_file):
    """Run NEAT algorithm"""
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    # Run for up to 50 generations
    winner = p.run(eval_genomes, 50)
    
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)